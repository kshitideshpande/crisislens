"""Sanity-check the CrisisMMD download before training anything.

    python scripts/check_data.py --data-dir /kaggle/input/crisismmd

Prints row counts and label distributions per task/split, and confirms that a
sample of image paths actually exist on disk.
"""

from __future__ import annotations

import argparse
import random

from crisislens.data import SPLITS, TASKS, find_image_root, find_split_dir, load_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--image-sample", type=int, default=200)
    args = ap.parse_args()

    split_dir = find_split_dir(args.data_dir)
    image_root = find_image_root(args.data_dir)
    print(f"Split files : {split_dir}")
    print(f"Image root  : {image_root or 'NOT FOUND (download CrisisMMD_v2.0.tar.gz)'}\n")

    ok = True
    for task, spec in TASKS.items():
        print(f"== {task} ==")
        for split in SPLITS:
            df = load_split(args.data_dir, task, split)
            counts = df["label"].value_counts()
            unknown = set(counts.index) - set(spec.expected_labels)
            print(f"  {split:<5} rows={len(df):>6}  unique tweets={df['tweet_id'].nunique():>6}  events={df['event_name'].nunique() if 'event_name' in df else '?'}")
            for label, n in counts.items():
                print(f"        {label:<42} {n:>6}  ({n / len(df):.1%})")
            if unknown:
                ok = False
                print(f"  [!] unexpected labels: {sorted(unknown)}")
            if (df["text"] == "").any():
                print(f"  [!] {(df['text'] == '').sum()} rows with empty text")

            if image_root is not None and split == "train":
                paths = df["image"].tolist()
                sample = random.Random(0).sample(paths, min(args.image_sample, len(paths)))
                missing = [p for p in sample if not (image_root / p).exists()]
                status = "all found" if not missing else f"{len(missing)} MISSING, e.g. {missing[0]}"
                print(f"  image check ({len(sample)} sampled): {status}")
                ok &= not missing
        print()

    print("Data looks good." if ok else "Some checks failed; see [!] lines above.")


if __name__ == "__main__":
    main()
