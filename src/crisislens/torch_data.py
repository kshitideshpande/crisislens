"""PyTorch datasets for the neural models (text, image, and fusion).

One dataset class serves every model: it always returns both modalities plus
`text_present` / `image_present` flags, and the model decides what to use.
Labels are a vector with one entry per task; -100 means "no label for this
task on this row" and is ignored by the loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from crisislens.data import TASKS, eval_mask, train_mask

IGNORE = -100


def label_maps(tasks: list[str]) -> dict[str, list[str]]:
    """Fixed label order per task, so class indices never change between runs."""
    return {t: sorted(TASKS[t].expected_labels) for t in tasks}


@dataclass
class View:
    """The rows and usable labels for one split (train / dev / test)."""

    df: pd.DataFrame
    labels: np.ndarray  # shape (n_rows, n_tasks), IGNORE where unusable


def make_views(table: pd.DataFrame, tasks: list[str], holdout_event: str | None = None) -> dict[str, View]:
    """Split the multi-task table into train/dev/test views.

    train: labels allowed by the leakage guard (`train_mask`).
    dev/test: the official split for each task.
    With `holdout_event`, that disaster is removed from train and dev and the
    test view contains only that disaster: a true "unseen disaster" test.
    """
    maps = label_maps(tasks)
    views = {}
    for split in ("train", "dev", "test"):
        masks = [train_mask(table, t) if split == "train" else eval_mask(table, t, split) for t in tasks]
        if holdout_event:
            in_event = table["event_name"] == holdout_event
            keep_rows = in_event if split == "test" else ~in_event
            masks = [m & keep_rows for m in masks]

        any_label = np.logical_or.reduce([m.to_numpy() for m in masks])
        df = table[any_label].reset_index(drop=True)
        labels = np.full((len(df), len(tasks)), IGNORE, dtype=np.int64)
        for j, (task, m) in enumerate(zip(tasks, masks)):
            m = m[any_label].to_numpy()
            idx = {name: i for i, name in enumerate(maps[task])}
            labels[m, j] = df.loc[m, f"label_{task}"].map(idx).to_numpy()
        views[split] = View(df=df, labels=labels)
    return views


def class_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    """Inverse-frequency weights, normalized to mean 1, for one task's labels."""
    y = labels[labels != IGNORE]
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    weights = counts.sum() / np.clip(counts, 1, None) / n_classes
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


class CrisisDataset(Dataset):
    def __init__(self, view: View, image_root: str | Path | None, image_processor=None):
        self.df = view.df
        self.labels = view.labels
        self.image_root = Path(image_root) if image_root else None
        self.processor = image_processor
        # Placeholder for missing images, shaped exactly like a processed image.
        if image_processor is not None:
            probe = image_processor(images=Image.new("RGB", (64, 64)), return_tensors="pt")["pixel_values"][0]
            self._blank = torch.zeros_like(probe)
        else:
            self._blank = torch.zeros(3, 224, 224)

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, rel_path: str) -> torch.Tensor | None:
        if self.processor is None or self.image_root is None or not rel_path:
            return None
        try:
            with Image.open(self.image_root / rel_path) as im:
                im = im.convert("RGB")
                return self.processor(images=im, return_tensors="pt")["pixel_values"][0]
        except (OSError, ValueError):
            return None  # missing or corrupt file: treat the post as text-only

    def __getitem__(self, i: int) -> dict:
        row = self.df.iloc[i]
        pixels = self._load_image(row["image"])
        return {
            "row_id": row["row_id"],
            "text": row["text"] or "",
            "pixel_values": pixels if pixels is not None else self._blank,
            "image_present": pixels is not None,
            "labels": torch.from_numpy(self.labels[i]),
        }


class Collator:
    """Tokenizes text per batch (faster than per item) and stacks everything."""

    def __init__(self, tokenizer=None, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, items: list[dict]) -> dict:
        texts = [it["text"] for it in items]
        batch = {
            "row_id": [it["row_id"] for it in items],
            "pixel_values": torch.stack([it["pixel_values"] for it in items]),
            "image_present": torch.tensor([it["image_present"] for it in items]),
            "text_present": torch.tensor([bool(t.strip()) for t in texts]),
            "labels": torch.stack([it["labels"] for it in items]),
        }
        if self.tokenizer is not None:
            enc = self.tokenizer(
                [t if t.strip() else "." for t in texts],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch["input_ids"] = enc["input_ids"]
            batch["attention_mask"] = enc["attention_mask"]
        return batch
