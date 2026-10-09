from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from crisislens.data import TASKS, label_names, load_split, load_task, normalize_tweet
from crisislens.metrics import evaluate


def test_normalize_tweet():
    out = normalize_tweet("RT @fema Shelter open at https://t.co/abc &amp; more  #Harvey")
    assert out == "@USER Shelter open at HTTPURL & more #Harvey"
    assert normalize_tweet(None) == ""


@pytest.mark.parametrize("task", sorted(TASKS))
def test_load_task(fake_data_dir: Path, task: str):
    splits = load_task(fake_data_dir, task)
    assert [len(splits[s]) for s in ("train", "dev", "test")] == [60, 20, 20]
    assert label_names(splits["train"]) == sorted(TASKS[task].expected_labels)
    assert "HTTPURL" in splits["train"]["text"].iloc[0]


def test_bad_task_raises(fake_data_dir: Path):
    with pytest.raises(ValueError):
        load_split(fake_data_dir, "nope", "train")


def test_urgent_recall():
    labels = ["a", "b", "c"]
    m = evaluate(["a", "b", "c", "a"], ["a", "c", "c", "b"], labels, urgent_labels=["a", "b"])
    # urgent truth: rows 0,1,3 ; predicted urgent: rows 0,3  -> recall 2/3
    assert m["urgent_recall"] == pytest.approx(2 / 3)
    assert m["n"] == 4 and len(m["confusion_matrix"]) == 3


def test_baseline_end_to_end(fake_data_dir: Path, tmp_path: Path):
    cfg = tmp_path / "cfg.yaml"
    out = tmp_path / "out"
    cfg.write_text(
        f"data_dir: {fake_data_dir}\noutput_dir: {out}\ntasks: [informative, humanitarian, damage]\n"
        "seed: 42\nmodel:\n  C_grid: [1.0]\n  ngram_max: 2\n  min_df: 1\n  max_features: 1000\n"
    )
    subprocess.run([sys.executable, "-m", "crisislens.train_baseline", "--config", str(cfg)], check=True)
    for task in TASKS:
        res = json.loads((out / task / "results.json").read_text())
        assert res["test"]["macro_f1"] > 0.9  # templates are trivially separable
        assert (out / task / "confusion_test.png").exists()
