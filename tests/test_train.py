from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from conftest import make_tiny_encoders
from crisislens.train import load_config, run, warmup_linear

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def tiny_cfg(config: str, data_dir: Path, out_dir: Path, **train) -> dict:
    cfg = load_config(CONFIGS / config)
    cfg.update(data_dir=str(data_dir), output_dir=str(out_dir))
    cfg["model"]["hidden_dim"] = 16
    cfg["data"].update(max_length=24, num_workers=0)
    cfg["train"].update({"epochs": 2, "batch_size": 8, "grad_accum": 1, "lr_text": 1e-3, "lr_image": 1e-3,
                         "lr_head": 5e-3, **train})
    return cfg


@pytest.mark.parametrize("config", ["text_bertweet.yaml", "image_clip.yaml", "fusion_concat.yaml", "fusion_gated.yaml"])
def test_every_config_trains(fake_data_dir, tmp_path, config):
    cfg = tiny_cfg(config, fake_data_dir, tmp_path / "out", max_train_batches=4)
    res = run(cfg, seed=1, encoders=make_tiny_encoders(), keep_model=True, device=torch.device("cpu"))

    seed_dir = tmp_path / "out" / "seed_1"
    for f in ("metrics.json", "predictions_dev.csv", "predictions_test.csv", "model.pt", "config.yaml"):
        assert (seed_dir / f).exists(), f
    saved = json.loads((seed_dir / "metrics.json").read_text())
    assert set(saved["test"]) == {"informative", "humanitarian", "damage"}
    assert len(res["history"]) >= 1

    preds = pd.read_csv(seed_dir / "predictions_test.csv")
    assert len(preds) == 60 and preds["row_id"].is_unique
    assert {"logit_humanitarian_0", "logit_humanitarian_4", "y_damage"} <= set(preds.columns)
    assert ("image_reliance" in preds) == config.startswith("fusion_gated")


def test_gated_model_learns(fake_data_dir, tmp_path):
    """On the trivially separable fake data, a few epochs should clearly beat chance."""
    cfg = tiny_cfg("fusion_gated.yaml", fake_data_dir, tmp_path / "out", epochs=6, patience=6)
    res = run(cfg, seed=0, encoders=make_tiny_encoders(), device=torch.device("cpu"))
    assert res["test"]["informative"]["macro_f1"] > 0.75
    assert not (tmp_path / "out" / "seed_0" / "model.pt").exists()  # not kept without keep_model


def test_holdout_config(fake_data_dir, tmp_path):
    cfg = tiny_cfg("fusion_gated_holdout.yaml", fake_data_dir, tmp_path / "out", epochs=1, max_train_batches=2)
    run(cfg, seed=0, encoders=make_tiny_encoders(), device=torch.device("cpu"))
    preds = pd.read_csv(tmp_path / "out" / "seed_0" / "predictions_test.csv")
    assert (preds["event_name"] == "california_wildfires").all()


def test_warmup_schedule():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = warmup_linear(opt, warmup_steps=2, total_steps=10)
    lrs = []
    for _ in range(10):
        lrs.append(sched.get_last_lr()[0])
        opt.step()
        sched.step()
    assert lrs[0] == 0.5 and lrs[1] == 1.0 and lrs[-1] < 0.2


def test_config_overrides():
    cfg = load_config(CONFIGS / "fusion_gated.yaml", {"train.epochs": 1, "data_dir": "x", "train.batch_size": None})
    assert cfg["train"]["epochs"] == 1 and cfg["data_dir"] == "x" and cfg["train"]["batch_size"] == 16
