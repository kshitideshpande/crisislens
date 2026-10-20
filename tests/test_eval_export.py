from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from conftest import make_tiny_encoders
from crisislens.evaluate import (calibrate_run, expected_calibration_error, fit_temperature, pick_threshold, robustness,
                                 softmax)
from crisislens.explain import Explainer, merge_subwords
from crisislens.inference import OnnxPredictor, occlude_image, occlude_words
from crisislens.late_fusion import fuse
from crisislens.train import load_trained_model

TASK_LIST = ["informative", "humanitarian", "damage"]


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def test_ece_perfect_and_overconfident():
    y = np.array([0, 1, 0, 1])
    assert expected_calibration_error(np.eye(2)[y], y) == pytest.approx(0.0)
    wrong = np.array([[0.99, 0.01]] * 4)  # always says class 0 with 99%, right half the time
    assert expected_calibration_error(wrong, y) == pytest.approx(0.49, abs=1e-6)


def test_temperature_softens_overconfident_logits():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 400)
    logits = rng.normal(size=(400, 3))
    logits[np.arange(400), y] += 1.0
    T = fit_temperature(logits * 8, y)  # 8x too confident
    assert T > 3
    assert (softmax(logits * 8, T).argmax(1) == (logits * 8).argmax(1)).all()  # never changes predictions


def test_pick_threshold_hits_target_recall():
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.1])
    urgent = np.array([True, True, True, True, False, False])
    thr = pick_threshold(p, urgent, 0.75)
    assert thr == pytest.approx(0.7)
    assert ((p >= thr) & urgent).sum() / urgent.sum() >= 0.75


def test_merge_subwords_bertweet_and_gpt_styles():
    s = np.array([0.0, 1.0, 2.0, 3.0, 0.0])
    assert merge_subwords(["<s>", "Flo@@", "od", "now", "</s>"], s, {"<s>", "</s>"}) == [("Flood", 3.0), ("now", 3.0)]
    assert merge_subwords(["<s>", "ĠFlo", "od", "Ġnow", "</s>"], s, {"<s>", "</s>"}) == [("Flood", 3.0), ("now", 3.0)]


def test_occlusion_finds_the_important_patch():
    def prob(pixels):  # "model" that only looks at the top-left corner
        return pixels[:, 0, :8, :8].mean(axis=(1, 2))

    heat = occlude_image(prob, np.ones((3, 32, 32), dtype=np.float32), grid=4)
    assert heat.argmax() == 0 and heat[0, 0] > 0 and np.allclose(heat.ravel()[1:], 0)

    words = occlude_words(lambda texts: np.array([float("help" in t) for t in texts]), "please help us")
    assert max(words, key=lambda w: w[1])[0] == "help"


# ---------------------------------------------------------------------------
# End to end on tiny models: train -> calibrate -> fuse -> explain -> export
# ---------------------------------------------------------------------------

def test_calibration_and_robustness(trained):
    run_dir = trained / "fusion_gated" / "seed_42"
    ev = calibrate_run(run_dir, TASK_LIST)
    assert set(ev["calibration"]) == set(TASK_LIST)
    assert 0 < ev["flagging"]["humanitarian"]["threshold"] <= 1
    rob = robustness(run_dir, make_tiny_encoders(), torch.device("cpu"))
    assert set(rob) == {"no_text", "no_image"}


def test_late_fusion(trained):
    res = fuse(trained / "text_bertweet" / "seed_42", trained / "image_clip" / "seed_42", TASK_LIST)
    assert set(res["test"]) == set(TASK_LIST)
    assert all(0 <= w <= 1 for w in res["weights_image"].values())
    # Picked on dev, so dev score is at least as good as either model alone (weights 0 and 1 are in the grid).
    text_dev = json.loads((trained / "text_bertweet" / "seed_42" / "metrics.json").read_text())["dev"]
    for t in TASK_LIST:
        assert res["dev"][t]["macro_f1"] >= text_dev[t]["macro_f1"] - 1e-9


def test_explainer(trained):
    model, tok, proc, _ = load_trained_model(trained / "fusion_gated" / "seed_42" / "model.pt", make_tiny_encoders())
    exp = Explainer(model, tok, proc, 24).explain("power lines down and the bridge", Image.new("RGB", (40, 40), "red"))
    assert [w for w, _ in exp["words"]] == ["power", "lines", "down", "and", "the", "bridge"]
    assert exp["heatmap"].shape == (7, 7)


def test_export_and_onnx_inference(trained, tmp_path):
    from crisislens.export_onnx import export

    run_dir = trained / "fusion_gated" / "seed_42"
    model, tok, proc, cfg = load_trained_model(run_dir / "model.pt", make_tiny_encoders())
    report = export(model, tok, proc, cfg, tmp_path / "exported", calibrate_run(run_dir, TASK_LIST))
    for t in TASK_LIST:
        assert report["parity"]["model.onnx"][t]["max_abs_diff"] < 1e-4
        assert report["parity"]["model.int8.onnx"][t]["argmax_agreement"] >= 0.75

    pred = OnnxPredictor(tmp_path / "exported")
    img = Image.new("RGB", (50, 30), (200, 40, 40))
    # Batch sizes differ from the export batch: checks the batch axis is truly dynamic.
    out = pred.predict(["Volunteers needed at the shelter", "Bridge collapsed", "no image here"], [img, img, None])
    assert len(out) == 3 and {"flagged", "p_urgent", "image_reliance"} <= set(out[0])
    assert out[2]["image_reliance"] == pytest.approx(0.0, abs=1e-5)  # no image -> no image weight
    assert abs(sum(out[0]["humanitarian"]["probs"].values()) - 1) < 1e-3

    exp = pred.explain("Volunteers needed at the shelter", img)
    assert len(exp["words"]) == 5 and exp["heatmap"].shape == (7, 7)


def test_onnx_matches_pytorch_on_new_shapes(trained, exported_model):
    """The export used one batch shape; outputs must still match PyTorch for other batch sizes and lengths."""
    model, *_ = load_trained_model(trained / "fusion_gated" / "seed_42" / "model.pt", make_tiny_encoders())
    pred = OnnxPredictor(exported_model, "model.onnx")
    img = Image.new("RGB", (50, 40), (10, 200, 30))
    for texts in (["power lines down"],
                  ["volunteers needed at the shelter donate water", "rt", "the bridge collapsed on main st and homes flattened"]):
        feeds = pred._feeds(texts, [img, None, img][:len(texts)])
        onnx_out = pred._run(feeds)
        with torch.no_grad():
            ref = model({k: torch.from_numpy(v) for k, v in feeds.items()})
        for t in TASK_LIST:
            assert np.abs(onnx_out[f"logits_{t}"] - ref[t].numpy()).max() < 1e-4


def test_posts_without_image_go_to_text_fallback(trained, exported_model, tmp_path):
    import shutil

    from crisislens.export_onnx import export

    exported = shutil.copytree(exported_model, tmp_path / "exported")  # leave the shared fixture untouched

    text_run = trained / "text_bertweet" / "seed_42"
    model, tok, proc, cfg = load_trained_model(text_run / "model.pt", make_tiny_encoders())
    export(model, tok, proc, cfg, exported / "text_fallback", calibrate_run(text_run, TASK_LIST))

    pred = OnnxPredictor(exported)
    assert pred.fallback is not None
    img = Image.new("RGB", (50, 30), (200, 40, 40))
    texts = ["Bridge collapsed", "no image here", "Volunteers needed at the shelter"]
    out = pred.predict(texts, [img, None, img])
    assert [r["model"] for r in out] == ["fusion_gated", "text_bertweet", "fusion_gated"]
    assert out[1]["image_reliance"] == 0.0
    # Same answers as calling each model on its own posts.
    assert out[1]["humanitarian"] == pred.fallback.predict([texts[1]])[0]["humanitarian"]
    alone = OnnxPredictor(exported, use_fallback=False).predict([texts[0], texts[2]], [img, img])
    assert [out[0]["humanitarian"], out[2]["humanitarian"]] == [a["humanitarian"] for a in alone]

    exp = pred.explain("no image here", None)
    assert exp["prediction"]["model"] == "text_bertweet" and exp["heatmap"] is None


def test_predictor_normalizes_text_like_training(exported_model):
    pred = OnnxPredictor(exported_model)
    raw = pred.encode_texts(["RT @fema Shelter open at https://t.co/abc &amp; more"])
    clean = pred.encode_texts(["@USER Shelter open at HTTPURL & more"])
    assert raw["input_ids"].tolist() == clean["input_ids"].tolist()
