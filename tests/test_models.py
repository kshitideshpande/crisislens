from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from crisislens.data import build_multitask_table, find_image_root, leakage_report, train_mask
from crisislens.models.fusion import FusionClassifier
from crisislens.torch_data import IGNORE, Collator, CrisisDataset, class_weights, label_maps, make_views

TASKS = ["informative", "humanitarian", "damage"]
N_CLASSES = {t: len(v) for t, v in label_maps(TASKS).items()}


def test_multitask_table(fake_data_dir):
    table = build_multitask_table(fake_data_dir)
    assert table["row_id"].is_unique
    assert {f"label_{t}" for t in TASKS} <= set(table.columns)
    # Every row has a label for at least one task.
    assert table[[f"label_{t}" for t in TASKS]].notna().any(axis=1).all()


def test_leakage_guard():
    # Row "b" is train for informative but test for humanitarian -> never trained on.
    table = pd.DataFrame({
        "split_informative": ["train", "train"],
        "split_humanitarian": ["train", "test"],
        "holdout_split": ["train", "test"],
    })
    assert train_mask(table, "informative").tolist() == [True, False]
    assert leakage_report(table, ("informative", "humanitarian")) == {"informative": 1, "humanitarian": 0}


def test_views_and_holdout(fake_data_dir):
    table = build_multitask_table(fake_data_dir)
    views = make_views(table, TASKS)
    assert len(views["train"].df) == 180 and len(views["test"].df) == 60
    assert views["train"].labels.shape == (180, 3)
    assert ((views["train"].labels != IGNORE).sum(axis=1) >= 1).all()

    held = make_views(table, TASKS, holdout_event="california_wildfires")
    assert (held["test"].df["event_name"] == "california_wildfires").all()
    assert not (held["train"].df["event_name"] == "california_wildfires").any()
    assert not (held["dev"].df["event_name"] == "california_wildfires").any()


def test_class_weights():
    w = class_weights(np.array([0, 0, 0, 1, IGNORE]), 2)
    assert w[1] > w[0] and torch.isclose(w.mean(), torch.tensor(1.0))


def _batch(fake_data_dir, tokenizer, processor, n=8):
    table = build_multitask_table(fake_data_dir)
    view = make_views(table, TASKS)["train"]
    ds = CrisisDataset(view, find_image_root(fake_data_dir), processor)
    return next(iter(DataLoader(ds, batch_size=n, collate_fn=Collator(tokenizer, 32))))


def test_dataset_handles_missing_images(fake_data_dir, tiny_encoders):
    _, tok, _, proc = tiny_encoders
    batch = _batch(fake_data_dir, tok, proc, n=8)
    assert batch["pixel_values"].shape == (8, 3, 32, 32)
    assert batch["image_present"].any() and not batch["image_present"].all()  # fixture leaves some missing
    assert batch["input_ids"].shape[0] == 8


@pytest.mark.parametrize("mode", ["text", "image", "concat", "gated"])
def test_forward_all_modes(fake_data_dir, tiny_encoders, mode):
    text, tok, image, proc = tiny_encoders
    model = FusionClassifier(mode, N_CLASSES, text, image, hidden_dim=16).eval()
    out = model(_batch(fake_data_dir, tok, proc))
    for t in TASKS:
        assert out[t].shape == (8, N_CLASSES[t])
    assert ("image_reliance" in out) == (mode == "gated")


def test_gate_respects_missing_modalities(fake_data_dir, tiny_encoders):
    text, tok, image, proc = tiny_encoders
    model = FusionClassifier("gated", N_CLASSES, text, image, hidden_dim=16).eval()
    batch = _batch(fake_data_dir, tok, proc)
    r = model(batch)["image_reliance"]
    assert ((r >= 0) & (r <= 1)).all()
    assert torch.allclose(r[~batch["image_present"]], torch.tensor(0.0))  # no image -> zero image weight

    batch["text_present"] = torch.zeros(8, dtype=torch.bool)
    batch["image_present"] = torch.ones(8, dtype=torch.bool)
    assert torch.allclose(model(batch)["image_reliance"], torch.tensor(1.0))  # no text -> all image


def test_modality_dropout_never_drops_both(tiny_encoders):
    text, _, image, _ = tiny_encoders
    model = FusionClassifier("gated", N_CLASSES, text, image, hidden_dim=16, modality_dropout=1.0).train()
    n = 2000
    batch = {"labels": torch.zeros(n, 3), "text_present": torch.ones(n, dtype=torch.bool),
             "image_present": torch.ones(n, dtype=torch.bool)}
    t_ok, i_ok = model._presence(batch, torch.device("cpu"))
    assert (t_ok | i_ok).all()
    assert 0.4 < (~t_ok).float().mean() < 0.6  # half lose text, half lose image


def test_param_groups_cover_everything_once(tiny_encoders):
    text, _, image, _ = tiny_encoders
    model = FusionClassifier("gated", N_CLASSES, text, image, hidden_dim=16)
    groups = model.param_groups(1e-5, 1e-5, 1e-3, 0.01)
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == sum(1 for _ in model.parameters())


def test_wrong_mode_raises(tiny_encoders):
    with pytest.raises(ValueError):
        FusionClassifier("nope", N_CLASSES)
    with pytest.raises(ValueError):
        FusionClassifier("text", N_CLASSES, text_encoder=None)
