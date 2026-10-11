"""Torch-free inference with the exported ONNX model (used by the demo app and the API).

    from crisislens.inference import OnnxPredictor
    p = OnnxPredictor("exported/")
    p.predict(["Bridge collapsed on Main St, people trapped"], [PIL.Image.open("photo.jpg")])

Only needs numpy, pillow, onnxruntime and transformers (for the tokenizer and
image preprocessing), so the free Hugging Face Space stays small and starts fast.

Explanations here use occlusion, which only needs forward passes:
  words: drop one word at a time and measure the change in probability
  image: gray out one patch at a time and measure the change in probability
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Generic occlusion (no model dependency; explain.py reuses occlude_image)
# ---------------------------------------------------------------------------


def occlude_image(prob_fn: Callable[[np.ndarray], np.ndarray], pixels: np.ndarray, grid: int = 7,
                  batch_size: int = 16) -> np.ndarray:
    """Importance of each grid cell = how much the target probability drops when it is hidden.

    pixels: one preprocessed image, shape (C, H, W). Hidden cells are set to 0,
    which is the dataset mean after CLIP's normalization (a neutral gray).
    """
    c, h, w = pixels.shape
    ys, xs = np.linspace(0, h, grid + 1).astype(int), np.linspace(0, w, grid + 1).astype(int)
    variants = []
    for i in range(grid):
        for j in range(grid):
            v = pixels.copy()
            v[:, ys[i]:ys[i + 1], xs[j]:xs[j + 1]] = 0.0
            variants.append(v)
    base = float(prob_fn(pixels[None])[0])
    probs = np.concatenate([prob_fn(np.stack(variants[k:k + batch_size])) for k in range(0, len(variants), batch_size)])
    return (base - probs).reshape(grid, grid)


def occlude_words(prob_fn: Callable[[list[str]], np.ndarray], text: str, max_words: int = 60) -> list[tuple[str, float]]:
    """Importance of each word = drop in target probability when that word is removed."""
    words = text.split()[:max_words]
    if not words:
        return []
    variants = [text] + [" ".join(words[:i] + words[i + 1:]) or "." for i in range(len(words))]
    probs = prob_fn(variants)
    return [(w, float(probs[0] - p)) for w, p in zip(words, probs[1:])]


def softmax(z: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = z / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# ONNX predictor
# ---------------------------------------------------------------------------


class OnnxPredictor:
    """Loads exported/{model.int8.onnx, meta.json, tokenizer files, preprocessor_config.json}."""

    def __init__(self, model_dir: str | Path, model_file: str | None = None):
        import onnxruntime as ort

        self.dir = Path(model_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        model_file = model_file or self.meta.get("default_model", "model.int8.onnx")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(self.dir / model_file), opts, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.tasks: list[str] = self.meta["tasks"]
        self.label_maps: dict[str, list[str]] = self.meta["label_maps"]
        self.temperature: dict[str, float] = self.meta.get("temperature", {})
        self.flag = self.meta.get("flag")
        self.tokenizer, self.processor = self._load_preprocessing()

    def _load_preprocessing(self):
        tokenizer = processor = None
        if {"input_ids", "attention_mask"} & self.input_names:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.dir)
        if "pixel_values" in self.input_names:
            try:  # pure-PIL preprocessing, identical to training
                from transformers import CLIPImageProcessorPil as Processor
            except ImportError:
                from transformers import CLIPImageProcessor as Processor
            processor = Processor.from_pretrained(self.dir)
        return tokenizer, processor

    # -- preprocessing ------------------------------------------------------
    def encode_texts(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        if self.tokenizer is None:
            return {}
        enc = self.tokenizer([t if t and t.strip() else "." for t in texts], padding=True, truncation=True,
                             max_length=self.meta.get("max_length", 128), return_tensors="np")
        return {"input_ids": enc["input_ids"].astype(np.int64), "attention_mask": enc["attention_mask"].astype(np.int64)}

    def encode_images(self, images: Sequence) -> np.ndarray:
        size = self.meta.get("image_size", 224)
        out = np.zeros((len(images), 3, size, size), dtype=np.float32)
        for i, im in enumerate(images):
            if im is not None and self.processor is not None:
                out[i] = self.processor(images=im.convert("RGB"), return_tensors="np")["pixel_values"][0]
        return out

    def _feeds(self, texts: Sequence[str], images: Sequence, pixels: np.ndarray | None = None) -> dict:
        n = len(texts)
        feeds = self.encode_texts(texts)
        feeds["pixel_values"] = pixels if pixels is not None else self.encode_images(images)
        feeds["text_present"] = np.array([bool(t and t.strip()) for t in texts])
        feeds["image_present"] = np.array([im is not None for im in images]) if images is not None else np.ones(n, bool)
        return {k: v for k, v in feeds.items() if k in self.input_names}

    def _run(self, feeds: dict) -> dict[str, np.ndarray]:
        return dict(zip(self.output_names, self.session.run(None, feeds)))

    def _probs(self, outputs: dict, task: str) -> np.ndarray:
        return softmax(outputs[f"logits_{task}"], self.temperature.get(task, 1.0))

    # -- public API ---------------------------------------------------------
    def predict(self, texts: Sequence[str], images: Sequence | None = None) -> list[dict]:
        """One result per post: label, confidence and probabilities per task, plus the urgent flag."""
        images = list(images) if images is not None else [None] * len(texts)
        outputs = self._run(self._feeds(list(texts), images))
        results = [dict() for _ in texts]
        for task in self.tasks:
            probs = self._probs(outputs, task)
            names = self.label_maps[task]
            for r, p in zip(results, probs):
                r[task] = {"label": names[int(p.argmax())], "confidence": float(p.max()),
                           "probs": {n: round(float(v), 4) for n, v in zip(names, p)}}
        reliance = outputs.get("image_reliance")
        for i, r in enumerate(results):
            r["image_reliance"] = float(reliance[i]) if reliance is not None else None
            if self.flag and self.flag["task"] in r:
                p_urgent = sum(r[self.flag["task"]]["probs"][l] for l in self.flag["urgent_labels"])
                r["p_urgent"] = round(float(p_urgent), 4)
                r["flagged"] = bool(p_urgent >= self.flag["threshold"])
        return results

    def explain(self, text: str, image=None, task: str = "humanitarian", grid: int = 7) -> dict:
        """Prediction plus word and image-region importance for its predicted class."""
        pred = self.predict([text], [image])[0]
        target = self.label_maps[task].index(pred[task]["label"])

        def text_prob(variants: list[str]) -> np.ndarray:
            pixels = np.repeat(self.encode_images([image]), len(variants), axis=0)
            out = self._run(self._feeds(variants, [image] * len(variants), pixels))
            return self._probs(out, task)[:, target]

        words = occlude_words(text_prob, text) if self.tokenizer is not None and text.strip() else []
        heatmap = None
        if image is not None and self.processor is not None:
            def image_prob(pixels: np.ndarray) -> np.ndarray:
                out = self._run(self._feeds([text] * len(pixels), [image] * len(pixels), pixels.astype(np.float32)))
                return self._probs(out, task)[:, target]

            heatmap = occlude_image(image_prob, self.encode_images([image])[0], grid=grid)
        return {"prediction": pred, "task": task, "target": pred[task]["label"], "words": words, "heatmap": heatmap}
