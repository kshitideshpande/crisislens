"""Loading CrisisMMD v2.0 splits.

Expected layout (after unpacking the two official downloads):

    <data_dir>/
        CrisisMMD_v2.0/                      # images (CrisisMMD_v2.0.tar.gz)
            data_image/<event>/<date>/<id>.jpg
        crisismmd_datasplit_agreed_label/    # splits (crisismmd_datasplit_agreed_label.zip)
            task_informative_text_img_agreed_lab_{train,dev,test}.tsv
            task_humanitarian_text_img_agreed_lab_{train,dev,test}.tsv
            task_damage_text_img_{train,dev,test}.tsv

Each row is one (tweet text, image) pair. A tweet with several images appears
several times, so every model (text-only, image-only, fusion) is evaluated on
exactly the same rows and the numbers stay comparable.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SPLITS = ("train", "dev", "test")


@dataclass(frozen=True)
class TaskSpec:
    name: str
    file_pattern: str
    expected_labels: tuple[str, ...]
    urgent_labels: tuple[str, ...] = ()


TASKS: dict[str, TaskSpec] = {
    "informative": TaskSpec(
        name="informative",
        file_pattern="task_informative_text_img_agreed_lab_{split}.tsv",
        expected_labels=("informative", "not_informative"),
    ),
    "humanitarian": TaskSpec(
        name="humanitarian",
        file_pattern="task_humanitarian_text_img_agreed_lab_{split}.tsv",
        expected_labels=(
            "affected_individuals",
            "infrastructure_and_utility_damage",
            "not_humanitarian",
            "other_relevant_information",
            "rescue_volunteering_or_donation_effort",
        ),
        # The classes where a missed post is costly for responders.
        urgent_labels=(
            "affected_individuals",
            "infrastructure_and_utility_damage",
            "rescue_volunteering_or_donation_effort",
        ),
    ),
    "damage": TaskSpec(
        name="damage",
        file_pattern="task_damage_text_img_{split}.tsv",
        expected_labels=("little_or_no_damage", "mild_damage", "severe_damage"),
        urgent_labels=("severe_damage",),
    ),
}

SPLIT_DIR_NAMES = ("crisismmd_datasplit_agreed_label", "crisismmd_datasplit_all")


def find_split_dir(data_dir: str | Path) -> Path:
    """Return the folder holding the split TSVs, searching a couple of levels deep.

    Kaggle sometimes nests uploads one folder deeper than expected, so this
    looks for the informative train file instead of trusting a fixed path.
    """
    data_dir = Path(data_dir)
    target = TASKS["informative"].file_pattern.format(split="train")
    for name in SPLIT_DIR_NAMES:
        if (data_dir / name / target).exists():
            return data_dir / name
    if (data_dir / target).exists():
        return data_dir
    matches = sorted(data_dir.rglob(target))
    if matches:
        return matches[0].parent
    raise FileNotFoundError(
        f"Could not find {target} under {data_dir}. "
        "Download crisismmd_datasplit_agreed_label.zip and unzip it there."
    )


def find_image_root(data_dir: str | Path) -> Path | None:
    """Return the folder that image paths in the TSVs are relative to."""
    data_dir = Path(data_dir)
    for candidate in (data_dir / "CrisisMMD_v2.0", data_dir):
        if (candidate / "data_image").is_dir():
            return candidate
    matches = sorted(data_dir.rglob("data_image"))
    return matches[0].parent if matches else None


_URL = re.compile(r"https?://\S+|www\.\S+")
_USER = re.compile(r"@\w+")
_SPACE = re.compile(r"\s+")


def normalize_tweet(text: str) -> str:
    """Light normalization matching BERTweet's conventions.

    URLs become HTTPURL and mentions become @USER, which keeps the signal that
    a link or mention was present without memorizing specific accounts.
    Hashtags are kept: #HurricaneHarvey is real information.
    """
    if not isinstance(text, str):
        return ""
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = _URL.sub("HTTPURL", text)
    text = _USER.sub("@USER", text)
    text = re.sub(r"^RT\s+", "", text)
    return _SPACE.sub(" ", text).strip()


def load_split(data_dir: str | Path, task: str, split: str) -> pd.DataFrame:
    """Load one task/split as a DataFrame with a clean `text` and `label` column."""
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; choose from {sorted(TASKS)}")
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; choose from {SPLITS}")

    path = find_split_dir(data_dir) / TASKS[task].file_pattern.format(split=split)
    df = pd.read_csv(
        path,
        sep="\t",
        quoting=csv.QUOTE_NONE,  # tweets contain stray quote characters
        dtype=str,
        keep_default_na=False,
        on_bad_lines="warn",
    )
    df.columns = [c.strip().lower() for c in df.columns]

    missing = {"tweet_id", "tweet_text", "image", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns {sorted(missing)}; found {list(df.columns)}")

    df["label"] = df["label"].str.strip()
    df["text"] = df["tweet_text"].map(normalize_tweet)
    df["split"] = split
    return df.reset_index(drop=True)


def load_task(data_dir: str | Path, task: str) -> dict[str, pd.DataFrame]:
    """Load train/dev/test for one task."""
    return {split: load_split(data_dir, task, split) for split in SPLITS}


def label_names(train_df: pd.DataFrame) -> list[str]:
    """Fixed, sorted label order taken from the training split."""
    return sorted(train_df["label"].unique())


# ---------------------------------------------------------------------------
# Multi-task table: one row per (tweet, image) with a label column per task.
# ---------------------------------------------------------------------------

SPLIT_RANK = {"train": 0, "dev": 1, "test": 2}


def build_multitask_table(data_dir: str | Path, tasks: tuple[str, ...] = tuple(TASKS)) -> pd.DataFrame:
    """Merge the per-task split files into one table for multi-task training.

    Each task file has its own rows (agreed-label filtering differs by task),
    so a (tweet, image) pair may have a label for one task and not another.
    Missing labels are NaN and are ignored by the loss.

    Columns: row_id, tweet_id, image, event_name, text,
             label_<task>, split_<task>   (one pair per task)
             holdout_split                 (the most held-out split across tasks)

    Leakage guard: a pair that is dev/test for ANY task is never trained on for
    ANY task. `train_mask(df, task)` applies this, so official test sets stay
    exactly as published while the model never sees a test image or tweet.
    """
    keys = ["tweet_id", "image"]
    per_task = {}
    for task in tasks:
        frames = []
        for split in SPLITS:
            df = load_split(data_dir, task, split)
            if "event_name" not in df:
                df["event_name"] = "unknown"
            df = df[keys + ["text", "event_name", "label"]].rename(columns={"label": f"label_{task}"})
            df[f"split_{task}"] = split
            frames.append(df)
        per_task[task] = pd.concat(frames, ignore_index=True).drop_duplicates(keys)

    # Base table: every (tweet, image) pair seen in any task, with its text and event.
    base = pd.concat([d[keys + ["text", "event_name"]] for d in per_task.values()], ignore_index=True)
    merged = base.drop_duplicates(keys).reset_index(drop=True)
    for task, d in per_task.items():
        merged = merged.merge(d[keys + [f"label_{task}", f"split_{task}"]], on=keys, how="left")

    merged["row_id"] = merged["tweet_id"].astype(str) + "|" + merged["image"].astype(str)

    split_cols = [f"split_{t}" for t in tasks]
    ranks = merged[split_cols].apply(lambda col: col.map(SPLIT_RANK))
    merged["holdout_split"] = ranks.max(axis=1).map({v: k for k, v in SPLIT_RANK.items()})
    front = ["row_id", "tweet_id", "image", "event_name", "text"]
    return merged[front + [c for c in merged.columns if c not in front]].reset_index(drop=True)


def train_mask(df: pd.DataFrame, task: str) -> pd.Series:
    """Rows whose `task` label may be used for training (see leakage guard above)."""
    return (df[f"split_{task}"] == "train") & (df["holdout_split"] == "train")


def eval_mask(df: pd.DataFrame, task: str, split: str) -> pd.Series:
    """Rows in the official `split` for `task`."""
    return df[f"split_{task}"] == split


def leakage_report(df: pd.DataFrame, tasks: tuple[str, ...] = tuple(TASKS)) -> dict[str, int]:
    """How many training labels the leakage guard removes, per task."""
    return {t: int(((df[f"split_{t}"] == "train") & ~train_mask(df, t)).sum()) for t in tasks}
