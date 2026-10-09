"""Evaluation metrics shared by every model, so all numbers are computed the same way."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
    urgent_labels: Sequence[str] = (),
) -> dict:
    """Return a JSON-friendly dict of metrics.

    Macro-F1 is the headline metric: it weights every class equally, so the
    model cannot score well by ignoring rare but important classes.
    """
    labels = list(labels)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
        "n": int(len(y_true)),
    }

    urgent = [l for l in urgent_labels if l in labels]
    if urgent:
        # Treat "any urgent class" vs "not urgent" as a binary question:
        # did we catch the posts responders needed to see?
        t = np.isin(np.asarray(y_true), urgent)
        p = np.isin(np.asarray(y_pred), urgent)
        out["urgent_recall"] = float(recall_score(t, p, zero_division=0))
        out["urgent_precision"] = float(precision_score(t, p, zero_division=0))
        out["urgent_labels"] = urgent
    return out


def summary_line(name: str, split: str, m: dict) -> str:
    line = f"{name:<28} {split:<5} acc={m['accuracy']:.4f}  macroF1={m['macro_f1']:.4f}  weightedF1={m['weighted_f1']:.4f}"
    if "urgent_recall" in m:
        line += f"  urgentR={m['urgent_recall']:.4f}"
    return line
