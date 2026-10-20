"""Export a trained model to ONNX, quantize it to int8, and check nothing broke.

    python -m crisislens.export_onnx --run outputs/fusion_gated/seed_42 --out exported/

Writes to --out:
    model.onnx          float32 graph
    model.int8.onnx     dynamically quantized weights (about 4x smaller, faster on CPU)
    meta.json           tasks, label names, temperatures and flag threshold (from evaluation.json)
    tokenizer + preprocessor files, so inference needs no internet and no PyTorch
    export_report.json  parity with PyTorch, test-split macro-F1 per file, file sizes, CPU latency
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from crisislens.inference import OnnxPredictor
from crisislens.torch_data import label_maps

SAMPLE_TEXTS = [
    "Bridge on Route 9 collapsed, cars stuck, please send help #flood",
    "Shelter at Lincoln High needs water and blankets, volunteers welcome",
    "Thinking of everyone affected tonight",
    "Power lines down across Oak Ave, avoid the area",
]


class ExportWrapper(nn.Module):
    """Plain-tensor signature for ONNX: (ids, mask, pixels, text_present, image_present) -> logits..."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, pixel_values, text_present, image_present):
        out = self.model({"input_ids": input_ids, "attention_mask": attention_mask, "pixel_values": pixel_values,
                          "text_present": text_present, "image_present": image_present})
        logits = tuple(out[t] for t in self.model.tasks)
        return logits + ((out["image_reliance"],) if "image_reliance" in out else ())


def sample_batch(tokenizer, processor, max_length: int, n: int = 4) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(0)
    texts = (SAMPLE_TEXTS * n)[:n]
    if tokenizer is not None:
        enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        ids, mask = enc["input_ids"], enc["attention_mask"]
    else:
        ids, mask = torch.zeros(n, 4, dtype=torch.long), torch.ones(n, 4, dtype=torch.long)
    if processor is not None:
        imgs = [Image.fromarray(rng.integers(0, 255, (96, 128, 3), dtype=np.uint8)) for _ in range(n)]
        pixels = processor(images=imgs, return_tensors="pt")["pixel_values"]
    else:
        pixels = torch.zeros(n, 3, 224, 224)
    image_present = torch.tensor([True] * (n - 1) + [False])  # include a no-image post
    return {"input_ids": ids, "attention_mask": mask, "pixel_values": pixels.float(),
            "text_present": torch.ones(n, dtype=torch.bool), "image_present": image_present}


def export(model: nn.Module, tokenizer, processor, cfg: dict, out_dir: str | Path,
           evaluation: dict | None = None, quantize: bool = True, test_check: bool = False) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    wrapper = ExportWrapper(model).eval()
    tasks = model.tasks
    batch = sample_batch(tokenizer, processor, cfg["data"]["max_length"])
    args = tuple(batch[k] for k in ("input_ids", "attention_mask", "pixel_values", "text_present", "image_present"))
    input_names = ["input_ids", "attention_mask", "pixel_values", "text_present", "image_present"]
    output_names = [f"logits_{t}" for t in tasks] + (["image_reliance"] if model.mode == "gated" else [])
    dynamic = {"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
               "pixel_values": {0: "batch"}, "text_present": {0: "batch"}, "image_present": {0: "batch"},
               **{o: {0: "batch"} for o in output_names}}

    fp32 = out_dir / "model.onnx"
    # The TorchScript exporter (dynamo=False) handles dynamic batch/sequence axes for Hugging Face
    # encoders reliably; tests check ONNX matches PyTorch on new batch sizes and sequence lengths.
    torch.onnx.export(wrapper, args, str(fp32), input_names=input_names, output_names=output_names,
                      dynamic_axes=dynamic, opset_version=17, dynamo=False)

    int8 = out_dir / "model.int8.onnx"
    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        # Per-channel scales matter: per-tensor int8 collapses BERTweet to one class. Embeddings stay
        # fp32 (MatMul only), which keeps the text model within ~0.01 macro-F1 of fp32 on the test split.
        quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8, per_channel=True, reduce_range=True,
                         op_types_to_quantize=["MatMul"])

    # Everything inference needs, saved next to the model.
    if tokenizer is not None:
        save_tokenizer(tokenizer, out_dir)
    if processor is not None:
        processor.save_pretrained(out_dir)
    evaluation = evaluation or {}
    flag = evaluation.get("flagging", {}).get("humanitarian")
    from crisislens.data import TASKS

    meta = {
        "name": cfg["name"], "mode": model.mode, "tasks": tasks, "label_maps": label_maps(tasks),
        "max_length": cfg["data"]["max_length"], "image_size": int(batch["pixel_values"].shape[-1]),
        "temperature": {t: c["temperature"] for t, c in evaluation.get("calibration", {}).items()},
        "flag": {"task": "humanitarian", "urgent_labels": list(TASKS["humanitarian"].urgent_labels),
                 "threshold": flag["threshold"] if flag else 0.5} if "humanitarian" in tasks else None,
        "default_model": int8.name if quantize else fp32.name,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # Parity: do ONNX models agree with PyTorch on the same inputs?
    with torch.no_grad():
        ref = {f"logits_{t}": v.numpy() for t, v in zip(tasks, wrapper(*args))}
    report = {"files_mb": {}, "parity": {}, "latency_ms": {}}
    feeds = {k: v.numpy() for k, v in batch.items()}
    for path in [fp32] + ([int8] if quantize else []):
        pred = OnnxPredictor(out_dir, path.name, use_fallback=False)
        outs = pred._run({k: v for k, v in feeds.items() if k in pred.input_names})
        report["files_mb"][path.name] = round(path.stat().st_size / 1e6, 2)
        report["parity"][path.name] = {
            t: {"max_abs_diff": float(np.abs(outs[f"logits_{t}"] - ref[f"logits_{t}"]).max()),
                "argmax_agreement": float((outs[f"logits_{t}"].argmax(1) == ref[f"logits_{t}"].argmax(1)).mean())}
            for t in tasks}
        one = {k: v[:1] for k, v in feeds.items() if k in pred.input_names}
        report["latency_ms"][path.name] = _median_ms(lambda: pred._run(one))

    one_args = tuple(a[:1] for a in args)
    with torch.no_grad():
        report["latency_ms"]["pytorch_fp32"] = _median_ms(lambda: wrapper(*one_args))
    if test_check:
        report["test_macro_f1"] = test_scores(out_dir, [p.name for p in [fp32] + ([int8] if quantize else [])], cfg)
    (out_dir / "export_report.json").write_text(json.dumps(report, indent=2))
    return report


def test_scores(out_dir: Path, files: list[str], cfg: dict, batch_size: int = 32) -> dict:
    """Macro-F1 of each exported file on the real test split, run end to end through OnnxPredictor.

    The parity check above feeds both models the same token ids, so it can't catch a broken saved
    tokenizer or image preprocessing; this goes through the exact path the demo uses.
    """
    from sklearn.metrics import f1_score

    from crisislens.data import build_multitask_table, find_image_root
    from crisislens.torch_data import IGNORE, make_views

    data_dir = Path(cfg["data_dir"])
    if not data_dir.exists():
        print(f"[warn] {data_dir} not found: skipping the test-split check")
        return {}
    tasks = cfg["tasks"]
    view = make_views(build_multitask_table(data_dir, tuple(tasks)), tasks, cfg.get("holdout_event"))["test"]
    texts, paths, root = list(view.df["text"].fillna("")), list(view.df["image"]), find_image_root(data_dir)

    def load(rel):
        try:
            with Image.open(root / rel) as im:
                return im.convert("RGB")
        except (OSError, ValueError, TypeError):
            return None

    scores = {}
    for name in files:
        pred = OnnxPredictor(out_dir, name, use_fallback=False)
        uses_image = "pixel_values" in pred.input_names
        labels_out = {t: [] for t in tasks}
        for s in range(0, len(texts), batch_size):
            batch = texts[s:s + batch_size]
            images = [load(p) for p in paths[s:s + batch_size]] if uses_image else [None] * len(batch)
            for r in pred.predict(batch, images):
                for t in tasks:
                    labels_out[t].append(pred.label_maps[t].index(r[t]["label"]))
        scores[name] = {}
        for j, t in enumerate(tasks):
            keep = view.labels[:, j] != IGNORE
            y_pred = np.array(labels_out[t])[keep]
            scores[name][t] = round(float(f1_score(view.labels[keep, j], y_pred, average="macro")), 4)
        print(f"test macro-F1 {name}: {scores[name]}")
    return scores


def save_tokenizer(tokenizer, out_dir: Path) -> None:
    """save_pretrained, then check the saved copy tokenizes exactly like the original.

    BERTweet's slow tokenizer writes bpe.codes without the merge counts, and reloading
    that file keeps only ~10k of its 64k merges (text comes out as single characters).
    Copying the original merges file fixes it; the round-trip check catches anything else.
    """
    import shutil

    from transformers import AutoTokenizer

    tokenizer.save_pretrained(out_dir)
    merges = getattr(tokenizer, "merges_file", None)
    if merges and Path(merges).exists():
        shutil.copyfile(merges, out_dir / Path(merges).name)
    reloaded = AutoTokenizer.from_pretrained(out_dir)
    for text in SAMPLE_TEXTS:
        if reloaded(text)["input_ids"] != tokenizer(text)["input_ids"]:
            raise RuntimeError(f"Saved tokenizer in {out_dir} does not match the training tokenizer on {text!r}")


def _median_ms(fn, warmup: int = 3, runs: int = 15) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return round(statistics.median(times), 2)


def main() -> None:
    from crisislens.train import load_trained_model

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="Run folder with model.pt (and ideally evaluation.json)")
    ap.add_argument("--out", default="exported")
    ap.add_argument("--no-quantize", action="store_true")
    ap.add_argument("--skip-test-check", action="store_true", help="Don't score the exported files on the test split")
    args = ap.parse_args()

    run_dir = Path(args.run)
    model, tok, proc, cfg = load_trained_model(run_dir / "model.pt")
    eval_path = run_dir / "evaluation.json"
    evaluation = json.loads(eval_path.read_text()) if eval_path.exists() else None
    if evaluation is None:
        print("[warn] no evaluation.json: run `python -m crisislens.evaluate --run ...` first for calibrated probabilities")
    report = export(model, tok, proc, cfg, args.out, evaluation, quantize=not args.no_quantize,
                    test_check=not args.skip_test_check)
    print(json.dumps(report, indent=2))
    scores = report.get("test_macro_f1", {})
    if "model.onnx" in scores and "model.int8.onnx" in scores:
        drop = max(scores["model.onnx"][t] - scores["model.int8.onnx"][t] for t in scores["model.onnx"])
        if drop > 0.02:
            print(f"[warn] int8 loses up to {drop:.3f} macro-F1 on the test split; consider serving model.onnx")


if __name__ == "__main__":
    main()
