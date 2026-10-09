"""Train and evaluate the TF-IDF + logistic regression baseline.

    python -m crisislens.train_baseline --config configs/baseline_tfidf.yaml
    python -m crisislens.train_baseline --config configs/baseline_tfidf.yaml --tasks informative --wandb

C is chosen on the dev split by macro-F1; the test split is used once, at the end.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import yaml

from crisislens.data import TASKS, label_names, load_task
from crisislens.metrics import evaluate, summary_line
from crisislens.models.baselines import build_tfidf_lr, top_features


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_confusion_png(metrics: dict, title: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array(metrics["confusion_matrix"], dtype=float)
    norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)  # row = true class
    labels = [l.replace("_", " ") for l in metrics["labels"]]

    fig, ax = plt.subplots(figsize=(1.3 * len(labels) + 3, 1.1 * len(labels) + 2.5))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center",
                    fontsize=8, color="white" if norm[i, j] > 0.55 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_task(task: str, cfg: dict, out_dir: Path, use_wandb: bool) -> dict:
    spec = TASKS[task]
    splits = load_task(cfg["data_dir"], task)
    labels = label_names(splits["train"])
    unexpected = set(labels) - set(spec.expected_labels)
    if unexpected:
        print(f"[warn] {task}: labels not in the expected set: {sorted(unexpected)}")

    X = {s: df["text"].tolist() for s, df in splits.items()}
    y = {s: df["label"].tolist() for s, df in splits.items()}
    print(f"\n=== {task} === train={len(X['train'])} dev={len(X['dev'])} test={len(X['test'])} classes={len(labels)}")

    model_cfg = cfg["model"]
    run = None
    if use_wandb:
        import wandb

        run = wandb.init(
            project=cfg.get("wandb_project", "crisislens"),
            name=f"tfidf_lr-{task}",
            group="baseline_tfidf_lr",
            config={"task": task, **model_cfg, "seed": cfg["seed"]},
            reinit=True,
        )

    # Model selection on dev only.
    sweep = []
    for C in model_cfg["C_grid"]:
        pipe = build_tfidf_lr(
            C=C,
            ngram_max=model_cfg["ngram_max"],
            min_df=model_cfg["min_df"],
            max_features=model_cfg["max_features"],
            seed=cfg["seed"],
        )
        t0 = time.time()
        pipe.fit(X["train"], y["train"])
        dev_m = evaluate(y["dev"], pipe.predict(X["dev"]), labels, spec.urgent_labels)
        sweep.append({"C": C, "dev_macro_f1": dev_m["macro_f1"], "fit_seconds": round(time.time() - t0, 2)})
        print(f"  C={C:<6} dev macroF1={dev_m['macro_f1']:.4f}")
        if run:
            run.log({"C": C, "dev/macro_f1": dev_m["macro_f1"]})

    best_C = max(sweep, key=lambda r: r["dev_macro_f1"])["C"]
    best = build_tfidf_lr(
        C=best_C,
        ngram_max=model_cfg["ngram_max"],
        min_df=model_cfg["min_df"],
        max_features=model_cfg["max_features"],
        seed=cfg["seed"],
    ).fit(X["train"], y["train"])

    results = {"task": task, "model": "tfidf_lr", "best_C": best_C, "sweep": sweep}
    for split in ("dev", "test"):
        m = evaluate(y[split], best.predict(X[split]), labels, spec.urgent_labels)
        results[split] = m
        print(summary_line(f"tfidf_lr (C={best_C})", split, m))
    results["top_features"] = top_features(best)

    task_dir = out_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "results.json").write_text(json.dumps(results, indent=2))
    save_confusion_png(results["test"], f"TF-IDF + LR, {task} (test)", task_dir / "confusion_test.png")

    if run:
        for split in ("dev", "test"):
            m = results[split]
            run.summary[f"{split}/macro_f1"] = m["macro_f1"]
            run.summary[f"{split}/accuracy"] = m["accuracy"]
            run.summary[f"{split}/weighted_f1"] = m["weighted_f1"]
            if "urgent_recall" in m:
                run.summary[f"{split}/urgent_recall"] = m["urgent_recall"]
        run.summary["best_C"] = best_C
        import wandb

        run.log({"test/confusion_matrix": wandb.Image(str(task_dir / "confusion_test.png"))})
        run.finish()
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--tasks", nargs="+", default=None, help="Override the tasks listed in the config")
    ap.add_argument("--data-dir", default=None, help="Override data_dir from the config")
    ap.add_argument("--wandb", action="store_true", help="Log runs to Weights & Biases")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    set_seed(cfg["seed"])

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = [run_task(t, cfg, out_dir, args.wandb) for t in (args.tasks or cfg["tasks"])]

    print("\n=== Test summary ===")
    for r in all_results:
        print(summary_line(r["task"], "test", r["test"]))


if __name__ == "__main__":
    main()
