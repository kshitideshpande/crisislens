"""REST API around the exported model, for integrating CrisisLens into other tools.

    MODEL_DIR=exported uvicorn service.api:app --port 8000
    curl -X POST localhost:8000/classify -H 'Content-Type: application/json' \
         -d '{"posts": [{"text": "Bridge collapsed on Main St, people trapped"}]}'

Endpoints
    GET  /health      liveness + whether the model loaded
    GET  /meta        tasks, labels, flag threshold
    POST /classify    batch of posts (text and/or base64 image) -> predictions
    POST /explain     one post -> prediction + word and image-region importance

The demo Space does not depend on this; it exists so the model can be served
behind any dashboard or alerting pipeline.
"""

from __future__ import annotations

import base64
import binascii
import io
import os
import time
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, model_validator

from crisislens.inference import OnnxPredictor

MAX_BATCH = 32
MAX_IMAGE_BYTES = 5_000_000

app = FastAPI(title="CrisisLens API", version="0.1.0",
              description="Multimodal triage of disaster social media posts (text + image).")


class Post(BaseModel):
    text: str = Field("", max_length=1000, description="Post text; may be empty if an image is given")
    image_base64: str | None = Field(None, description="JPEG or PNG, base64-encoded")

    @model_validator(mode="after")
    def needs_content(self):
        if not self.text.strip() and not self.image_base64:
            raise ValueError("each post needs text or an image")
        return self


class ClassifyRequest(BaseModel):
    posts: list[Post] = Field(..., min_length=1, max_length=MAX_BATCH)


class ExplainRequest(Post):
    task: str = "humanitarian"


@lru_cache(maxsize=1)
def get_predictor() -> OnnxPredictor:
    return OnnxPredictor(os.environ.get("MODEL_DIR", "exported"))


def decode_image(b64: str | None) -> Image.Image | None:
    if not b64:
        return None
    try:
        raw = base64.b64decode(b64.split(",", 1)[-1], validate=True)  # tolerate "data:image/...;base64," prefixes
    except (binascii.Error, ValueError):
        raise HTTPException(422, "image_base64 is not valid base64")
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(413, f"image larger than {MAX_IMAGE_BYTES // 1_000_000} MB")
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(422, "image_base64 is not a readable JPEG/PNG")


@app.get("/health")
def health() -> dict:
    try:
        get_predictor()
        return {"status": "ok", "model_loaded": True}
    except Exception as e:
        return {"status": "degraded", "model_loaded": False, "error": f"{type(e).__name__}: {e}"}


@app.get("/meta")
def meta() -> dict:
    m = get_predictor().meta
    return {k: m.get(k) for k in ("name", "mode", "tasks", "label_maps", "flag", "temperature")}


@app.post("/classify")
def classify(req: ClassifyRequest) -> dict:
    t0 = time.perf_counter()
    images = [decode_image(p.image_base64) for p in req.posts]
    results = get_predictor().predict([p.text for p in req.posts], images)
    return {"results": results, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


@app.post("/explain")
def explain(req: ExplainRequest) -> dict:
    pred = get_predictor()
    if req.task not in pred.tasks:
        raise HTTPException(422, f"task must be one of {pred.tasks}")
    exp = pred.explain(req.text, decode_image(req.image_base64), task=req.task)
    exp["heatmap"] = exp["heatmap"].round(4).tolist() if exp["heatmap"] is not None else None
    return exp
