"""A tiny fake CrisisMMD folder, so tests run in CI without the real 1.8 GB dataset."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from crisislens.data import TASKS

HEADER = "event_name\ttweet_id\timage_id\ttweet_text\timage\tlabel\tlabel_text\tlabel_image\tlabel_text_image\n"

TEMPLATES = {
    "informative": "Flood waters rising near the bridge, roads closed",
    "not_informative": "Thinking of everyone today, stay safe friends",
    "affected_individuals": "Two people injured and a family missing after the quake",
    "infrastructure_and_utility_damage": "Power lines down and the bridge collapsed on Main St",
    "not_humanitarian": "Great game last night, what a finish",
    "other_relevant_information": "Storm expected to make landfall tomorrow evening",
    "rescue_volunteering_or_donation_effort": "Volunteers needed, donate water and blankets at the shelter",
    "little_or_no_damage": "Some leaves on the road but houses are fine",
    "mild_damage": "A few roof shingles torn off on our street",
    "severe_damage": "Entire block destroyed, homes flattened",
}


EVENTS = ("hurricane_harvey", "california_wildfires")


def _write_split(path: Path, labels: tuple[str, ...], n: int, rng: random.Random, image_root: Path | None = None) -> None:
    rows = [HEADER]
    for i in range(n):
        label = labels[i % len(labels)]
        event = EVENTS[(i // len(labels)) % len(EVENTS)]
        tid = 900000000000000000 + rng.randint(0, 10**9)
        text = f"RT @someone {TEMPLATES[label]} #Harvey https://t.co/x{i}"
        image = f"data_image/{event}/1_9_2017/{tid}_0.jpg"
        if image_root is not None and i % 4 != 3:  # every 4th image is "missing" on disk
            _write_image(image_root / image, labels.index(label), rng)
        rows.append(f"{event}\t{tid}\t{tid}_0\t{text}\t{image}\t{label}\t{label}\t{label}\tPositive\n")
    path.write_text("".join(rows))


def _write_image(path: Path, label_idx: int, rng: random.Random) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    color = [(200, 40, 40), (40, 200, 40), (40, 40, 200), (200, 200, 40), (40, 200, 200)][label_idx % 5]
    Image.new("RGB", (48 + rng.randint(0, 16), 40), color).save(path)


@pytest.fixture
def fake_data_dir(tmp_path: Path) -> Path:
    rng = random.Random(0)
    split_dir = tmp_path / "crisismmd_datasplit_agreed_label"
    split_dir.mkdir()
    image_root = tmp_path / "CrisisMMD_v2.0"
    for spec in TASKS.values():
        for split, n in (("train", 60), ("dev", 20), ("test", 20)):
            _write_split(split_dir / spec.file_pattern.format(split=split), spec.expected_labels, n, rng, image_root)
    return tmp_path


# ---------------------------------------------------------------------------
# Tiny, randomly initialized stand-ins for BERTweet and CLIP (no downloads).
# ---------------------------------------------------------------------------

def make_tiny_tokenizer():
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = sorted({w.lower() for t in TEMPLATES.values() for w in t.split()} | {"httpurl", "@user", "#harvey", "rt", "."})
    vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, **{w: i + 4 for i, w in enumerate(words)}}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.normalizer = normalizers.Lowercase()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, bos_token="<s>", eos_token="</s>", pad_token="<pad>", unk_token="<unk>"
    )


def make_tiny_encoders():
    """Return (text_model, tokenizer, image_model, image_processor), all tiny."""
    from transformers import CLIPVisionConfig, CLIPVisionModel, RobertaConfig, RobertaModel

    from crisislens.models.encoders import clip_processor_class

    tokenizer = make_tiny_tokenizer()
    text = RobertaModel(RobertaConfig(
        vocab_size=len(tokenizer), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=64, max_position_embeddings=80, pad_token_id=1,
    ))
    image = CLIPVisionModel(CLIPVisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, image_size=32, patch_size=16,
    ))
    processor = clip_processor_class()(size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32})
    return text, tokenizer, image, processor


@pytest.fixture
def tiny_encoders():
    return make_tiny_encoders()


# ---------------------------------------------------------------------------
# Session-wide: tiny text/image/gated runs, plus an exported ONNX gated model.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    import torch

    from crisislens.train import load_config, run

    configs = Path(__file__).resolve().parents[1] / "configs"
    root = tmp_path_factory.mktemp("e2e")
    data = root / "data"
    split_dir = data / "crisismmd_datasplit_agreed_label"
    split_dir.mkdir(parents=True)
    rng = random.Random(0)
    for spec in TASKS.values():
        for split, n in (("train", 60), ("dev", 20), ("test", 20)):
            _write_split(split_dir / spec.file_pattern.format(split=split), spec.expected_labels, n, rng,
                         data / "CrisisMMD_v2.0")
    out = root / "outputs"
    for name in ("text_bertweet", "image_clip", "fusion_gated"):
        cfg = load_config(configs / f"{name}.yaml")
        cfg.update(data_dir=str(data), output_dir=str(out / name))
        cfg["model"]["hidden_dim"] = 16
        cfg["data"].update(max_length=24, num_workers=0)
        cfg["train"].update({"epochs": 3, "batch_size": 8, "grad_accum": 1, "lr_text": 1e-3, "lr_image": 1e-3,
                             "lr_head": 5e-3})
        run(cfg, seed=42, encoders=make_tiny_encoders(), keep_model=True, device=torch.device("cpu"))
    return out


@pytest.fixture(scope="session")
def exported_model(trained, tmp_path_factory) -> Path:
    from crisislens.evaluate import calibrate_run
    from crisislens.export_onnx import export
    from crisislens.train import load_trained_model

    run_dir = trained / "fusion_gated" / "seed_42"
    model, tok, proc, cfg = load_trained_model(run_dir / "model.pt", make_tiny_encoders())
    out = tmp_path_factory.mktemp("exported")
    export(model, tok, proc, cfg, out, calibrate_run(run_dir, ["informative", "humanitarian", "damage"]))
    return out
