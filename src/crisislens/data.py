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
