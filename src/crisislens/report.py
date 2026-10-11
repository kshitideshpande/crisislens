"""Collect every finished run into one results table (mean ± std across seeds).

    python -m crisislens.report

Reads outputs/*/seed_*/metrics.json (+ evaluation.json when present) and the
TF-IDF baseline, then writes:
    outputs/results.csv        one row per model x task, ready for the notebook
    outputs/results.md         the table pasted into the README
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ORDER = ["baseline_tfidf_lr", "text_bertweet", "image_clip", "late_fusion", "fusion_concat", "fusion_gated"]
PRETTY = {
    "baseline_tfidf_lr": "TF-IDF + LR (text)",
    "text_bertweet": "BERTweet (text)",
    "image_clip": "CLIP (image)",
    "late_fusion": "Late fusion (avg probs)",
    "fusion_concat": "Concat fusion",
    "fusion_gated": "Gated fusion",
    "fusion_gated_holdout": "Gated fusion, unseen disaster",
}
TASK_ORDER = ["informative", "humanitarian", "damage"]


def collect(outputs: Path) -> pd.DataFrame:
    rows = []
    for metrics_path in sorted(outputs.glob("*/seed_*/metrics.json")):
        m = json.loads(metrics_path.read_text())
        name = metrics_path.parent.parent.name
        seed = int(metrics_path.parent.name.split("_")[1])
        for task, tm in m["test"].items():
            rows.append({"model": name, "seed": seed, "task": task, "macro_f1": tm["macro_f1"],
                         "accuracy": tm["accuracy"], "urgent_recall": tm.get("urgent_recall", np.nan),
                         "train_seconds": m.get("train_seconds", np.nan)})
    # Baseline (single seed, its own layout).
    for res_path in sorted(outputs.glob("baseline_tfidf_lr/*/results.json")):
        r = json.loads(res_path.read_text())
        t = r["test"]
        rows.append({"model": "baseline_tfidf_lr", "seed": 0, "task": r["task"], "macro_f1": t["macro_f1"],
                     "accuracy": t["accuracy"], "urgent_recall": t.get("urgent_recall", np.nan), "train_seconds": np.nan})
    return pd.DataFrame(rows)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    g = raw.groupby(["model", "task"])
    out = g.agg(macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
                urgent_recall_mean=("urgent_recall", "mean"), urgent_recall_std=("urgent_recall", "std"),
                n_seeds=("seed", "nunique")).reset_index()
    return out.fillna({"macro_f1_std": 0.0, "urgent_recall_std": 0.0})


def fmt(mean: float, std: float, n: int) -> str:
    if np.isnan(mean):
        return ""
    return f"{mean:.3f} ± {std:.3f}" if n > 1 else f"{mean:.3f}"


def markdown_table(summary: pd.DataFrame, models: list[str]) -> str:
    header = "| Model | " + " | ".join(t.title() for t in TASK_ORDER) + " | Urgent recall (humanitarian) |"
    lines = [header, "| --- " * (len(TASK_ORDER) + 2) + "|"]
    for model in models:
        s = summary[summary["model"] == model].set_index("task")
        if s.empty:
            continue
        cells = [fmt(s.loc[t, "macro_f1_mean"], s.loc[t, "macro_f1_std"], s.loc[t, "n_seeds"]) if t in s.index else ""
                 for t in TASK_ORDER]
        hum = s.loc["humanitarian"] if "humanitarian" in s.index else None
        urgent = fmt(hum["urgent_recall_mean"], hum["urgent_recall_std"], hum["n_seeds"]) if hum is not None else ""
        lines.append(f"| {PRETTY.get(model, model)} | " + " | ".join(cells) + f" | {urgent} |")
    return "\n".join(lines)


def extras(outputs: Path) -> str:
    """Calibration and robustness for the deployed model, averaged over seeds."""
    evals = [json.loads(p.read_text()) for p in sorted(outputs.glob("fusion_gated/seed_*/evaluation.json"))]
    if not evals:
        return ""
    lines = ["", "**Gated fusion, extra checks (test split)**", ""]
    for task in TASK_ORDER:
        cal = [e["calibration"][task] for e in evals if task in e.get("calibration", {})]
        if cal:
            b = np.mean([c["test_ece_before"] for c in cal])
            a = np.mean([c["test_ece_after"] for c in cal])
            lines.append(f"- Calibration ({task}): ECE {b:.3f} → {a:.3f} after temperature scaling")
    flags = [e["flagging"]["humanitarian"] for e in evals if "humanitarian" in e.get("flagging", {})]
    if flags:
        r = np.mean([f["test"]["recall"] for f in flags])
        p = np.mean([f["test"]["precision"] for f in flags])
        lines.append(f"- Urgent flag (threshold set on dev for {flags[0]['target_recall']:.0%} recall): "
                     f"test recall {r:.3f}, precision {p:.3f}")
    rob = [e["robustness"] for e in evals if e.get("robustness")]
    if rob:
        for cond in ("full", "no_image", "no_text"):
            vals = [r[cond] for r in rob if cond in r]
            if vals:
                cells = ", ".join(f"{t} {np.mean([v[t] for v in vals]):.3f}" for t in TASK_ORDER if t in vals[0])
                lines.append(f"- Macro-F1 with {cond.replace('_', ' ')}: {cells}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    args = ap.parse_args()
    outputs = Path(args.outputs)

    raw = collect(outputs)
    if raw.empty:
        raise SystemExit(f"No finished runs found under {outputs}/")
    summary = summarize(raw)
    summary.to_csv(outputs / "results.csv", index=False)

    md = ["**Main results** (CrisisMMD v2.0 test split, macro-F1, mean ± std over seeds)", "",
          markdown_table(summary, ORDER)]
    if (summary["model"] == "fusion_gated_holdout").any():
        md += ["", "**Unseen disaster** (trained without it; tested only on it)", "",
               markdown_table(summary, ["fusion_gated_holdout"])]
    md.append(extras(outputs))

    g = summary.set_index(["model", "task"])
    if ("fusion_gated", "humanitarian") in g.index and ("text_bertweet", "humanitarian") in g.index:
        gain = g.loc[("fusion_gated", "humanitarian"), "macro_f1_mean"] - g.loc[("text_bertweet", "humanitarian"), "macro_f1_mean"]
        md += ["", f"Adding the image (gated fusion vs. text-only BERTweet) changes humanitarian macro-F1 by {gain:+.3f}."]

    text = "\n".join(md) + "\n"
    (outputs / "results.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
