"""Late fusion: average the text-only and image-only models' probabilities.

    python -m crisislens.late_fusion --seed 42

The cheapest way to combine modalities: no new training, just
    p = (1 - w) * p_text + w * p_image
with one weight w per task, picked on dev. If learned fusion (concat / gated)
can't beat this, its extra complexity isn't earning its keep.

Reads  outputs/text_bertweet/seed_<s>/ and outputs/image_clip/seed_<s>/
Writes outputs/late_fusion/seed_<s>/metrics.json (same format as train.py).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from crisislens.data import TASKS
from crisislens.evaluate import load_predictions, softmax
from crisislens.metrics import evaluate, summary_line
from crisislens.torch_data import label_maps

WEIGHTS = np.round(np.linspace(0, 1, 21), 2)


def _metrics(probs: np.ndarray, y: np.ndarray, task: str) -> dict:
    names = label_maps([task])[task]
    return evaluate([names[i] for i in y], [names[i] for i in probs.argmax(1)], names, TASKS[task].urgent_labels)


def fuse(text_dir: Path, image_dir: Path, tasks: list[str]) -> dict:
    res = {"name": "late_fusion", "weights_image": {}, "dev": {}, "test": {}}
    loaded = {}
    for split in ("dev", "test"):
        ft, pt = load_predictions(text_dir / f"predictions_{split}.csv", tasks)
        fi, pi = load_predictions(image_dir / f"predictions_{split}.csv", tasks)
        if ft["row_id"].tolist() != fi["row_id"].tolist():
            raise ValueError(f"{split}: text and image runs scored different rows; rerun both with the same data")
        loaded[split] = (pt, pi)

    for task in tasks:
        (lt_dev, y_dev), (li_dev, _) = loaded["dev"][0][task], loaded["dev"][1][task]
        p_t, p_i = softmax(lt_dev), softmax(li_dev)
        scores = {w: _metrics((1 - w) * p_t + w * p_i, y_dev, task)["macro_f1"] for w in WEIGHTS}
        w = float(max(scores, key=scores.get))  # ties -> the first (more text) weight
        res["weights_image"][task] = w
        for split in ("dev", "test"):
            (lt, y), (li, _) = loaded[split][0][task], loaded[split][1][task]
            res[split][task] = _metrics((1 - w) * softmax(lt) + w * softmax(li), y, task)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--text-run", default="text_bertweet")
    ap.add_argument("--image-run", default="image_clip")
    args = ap.parse_args()

    root = Path(args.outputs)
    text_dir, image_dir = root / args.text_run / f"seed_{args.seed}", root / args.image_run / f"seed_{args.seed}"
    tasks = list(json.loads((text_dir / "metrics.json").read_text())["test"])
    res = fuse(text_dir, image_dir, tasks)
    res["seed"] = args.seed

    out = root / "late_fusion" / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    for task, m in res["test"].items():
        print(summary_line(f"late_fusion {task} (w_img={res['weights_image'][task]})", "test", m))


if __name__ == "__main__":
    main()
