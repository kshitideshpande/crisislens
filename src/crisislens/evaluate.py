"""Post-training evaluation for one run: calibration, flag threshold, robustness.

    python -m crisislens.evaluate --run outputs/fusion_gated/seed_42                  # calibration + threshold
    python -m crisislens.evaluate --run outputs/fusion_gated/seed_42 --robustness     # + missing-modality tests

Calibration: a model that says "90% sure" should be right about 90% of the time.
Temperature scaling divides the logits by one number T, fit on dev. It never
changes which class wins, only how confident the model claims to be.

Flag threshold: the dashboard flags a post as urgent when P(urgent) is above a
threshold. We pick it on dev to reach a target urgent recall (default 90%),
then report the precision that costs on test.

Writes <run>/evaluation.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crisislens.data import TASKS
from crisislens.torch_data import IGNORE, label_maps

# ---------------------------------------------------------------------------
# Reading prediction files (shared with late_fusion.py and report.py)
# ---------------------------------------------------------------------------


def load_predictions(path: str | Path, tasks: list[str]) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Return (frame, {task: (logits[n, k], labels[n])}) keeping only labeled rows per task."""
    df = pd.read_csv(path)
    out = {}
    for task in tasks:
        cols = [c for c in df.columns if c.startswith(f"logit_{task}_")]
        cols.sort(key=lambda c: int(c.rsplit("_", 1)[1]))
        y = df[f"y_{task}"].to_numpy()
        keep = y != IGNORE
        out[task] = (df.loc[keep, cols].to_numpy(dtype=np.float64), y[keep].astype(int))
    return df, out


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Average gap between confidence and accuracy, weighted by how many predictions fall in each bin."""
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def nll(logits: np.ndarray, y: np.ndarray, temperature: float) -> float:
    p = softmax(logits, temperature)[np.arange(len(y)), y]
    return float(-np.log(np.clip(p, 1e-12, None)).mean())


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Pick T minimizing dev negative log-likelihood (coarse grid, then a fine grid)."""
    grid = np.exp(np.linspace(np.log(0.05), np.log(10), 200))
    best = min(grid, key=lambda t: nll(logits, y, t))
    fine = np.linspace(best * 0.9, best * 1.1, 100)
    return float(min(fine, key=lambda t: nll(logits, y, t)))


# ---------------------------------------------------------------------------
# Urgent-flag threshold
# ---------------------------------------------------------------------------


def urgent_probability(probs: np.ndarray, task: str) -> np.ndarray:
    names = label_maps([task])[task]
    idx = [names.index(l) for l in TASKS[task].urgent_labels]
    return probs[:, idx].sum(axis=1)


def pick_threshold(p_urgent: np.ndarray, is_urgent: np.ndarray, target_recall: float) -> float:
    """Highest threshold whose recall on these rows still reaches the target."""
    if not is_urgent.any():
        return 0.5
    scores = np.sort(p_urgent[is_urgent])[::-1]
    k = int(np.ceil(target_recall * len(scores))) - 1
    return float(scores[max(0, min(k, len(scores) - 1))])


def threshold_metrics(p_urgent: np.ndarray, is_urgent: np.ndarray, threshold: float) -> dict:
    flag = p_urgent >= threshold
    tp = int((flag & is_urgent).sum())
    return {
        "recall": tp / max(1, int(is_urgent.sum())),
        "precision": tp / max(1, int(flag.sum())),
        "flag_rate": float(flag.mean()),
    }


def calibrate_run(run_dir: Path, tasks: list[str], target_recall: float = 0.9) -> dict:
    _, dev = load_predictions(run_dir / "predictions_dev.csv", tasks)
    _, test = load_predictions(run_dir / "predictions_test.csv", tasks)
    result = {"calibration": {}, "flagging": {}}
    for task in tasks:
        (ld, yd), (lt, yt) = dev[task], test[task]
        if len(yd) == 0 or len(yt) == 0:
            continue
        T = fit_temperature(ld, yd)
        result["calibration"][task] = {
            "temperature": T,
            "test_ece_before": expected_calibration_error(softmax(lt), yt),
            "test_ece_after": expected_calibration_error(softmax(lt, T), yt),
            "test_nll_before": nll(lt, yt, 1.0),
            "test_nll_after": nll(lt, yt, T),
        }
        if TASKS[task].urgent_labels:
            names = label_maps([task])[task]
            urgent_idx = [names.index(l) for l in TASKS[task].urgent_labels]
            pd_, pt_ = urgent_probability(softmax(ld, T), task), urgent_probability(softmax(lt, T), task)
            thr = pick_threshold(pd_, np.isin(yd, urgent_idx), target_recall)
            result["flagging"][task] = {
                "target_recall": target_recall,
                "threshold": thr,
                "dev": threshold_metrics(pd_, np.isin(yd, urgent_idx), thr),
                "test": threshold_metrics(pt_, np.isin(yt, urgent_idx), thr),
            }
    return result


# ---------------------------------------------------------------------------
# Missing-modality robustness (needs the saved model)
# ---------------------------------------------------------------------------


def robustness(run_dir: Path, encoders: tuple | None = None, device=None) -> dict:
    """Test macro-F1 with every post's text hidden, and with every image hidden."""
    import torch
    from torch.utils.data import DataLoader

    from crisislens.data import build_multitask_table, find_image_root
    from crisislens.torch_data import Collator, CrisisDataset, make_views
    from crisislens.train import load_trained_model, pick_device, predict, score

    device = device or pick_device()
    model, tokenizer, processor, cfg = load_trained_model(run_dir / "model.pt", encoders, device)
    if model.mode not in ("concat", "gated"):
        return {}
    tasks = cfg["tasks"]
    view = make_views(build_multitask_table(cfg["data_dir"], tuple(tasks)), tasks, cfg.get("holdout_event"))["test"]
    loader = DataLoader(CrisisDataset(view, find_image_root(cfg["data_dir"]), processor),
                        batch_size=cfg["train"]["batch_size"] * 2,
                        collate_fn=Collator(tokenizer, cfg["data"]["max_length"]))

    def hide(flag: str):
        for batch in loader:
            batch[flag] = torch.zeros_like(batch[flag])
            yield batch

    out = {}
    for name, flag in (("no_text", "text_present"), ("no_image", "image_present")):
        out[name] = {t: m["macro_f1"] for t, m in score(predict(model, hide(flag), device), tasks).items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="e.g. outputs/fusion_gated/seed_42")
    ap.add_argument("--target-recall", type=float, default=0.9)
    ap.add_argument("--robustness", action="store_true", help="Also run missing-modality tests (needs model.pt)")
    args = ap.parse_args()

    run_dir = Path(args.run)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    tasks = list(metrics["test"])
    result = calibrate_run(run_dir, tasks, args.target_recall)
    if args.robustness:
        result["robustness"] = robustness(run_dir)
        result["robustness"]["full"] = {t: m["macro_f1"] for t, m in metrics["test"].items()}
    (run_dir / "evaluation.json").write_text(json.dumps(result, indent=2))

    for t, c in result["calibration"].items():
        print(f"{t:<13} T={c['temperature']:.2f}  ECE {c['test_ece_before']:.4f} -> {c['test_ece_after']:.4f}")
    for t, f in result["flagging"].items():
        print(f"{t:<13} flag at P(urgent) >= {f['threshold']:.3f}: test recall={f['test']['recall']:.3f} "
              f"precision={f['test']['precision']:.3f}")
    for name, scores in result.get("robustness", {}).items():
        print(f"{name:<9} " + "  ".join(f"{t}={v:.3f}" for t, v in scores.items()))


if __name__ == "__main__":
    main()
