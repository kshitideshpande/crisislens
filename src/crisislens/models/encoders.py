"""Loading the pretrained encoders.

Text:  BERTweet (RoBERTa architecture, pretrained on 850M English tweets),
       so hashtags, slang, and tweet structure are already familiar to it.
Image: the vision half of CLIP ViT-B/32, pretrained on 400M image-caption
       pairs, so its features already line up with language concepts.
"""

from __future__ import annotations

import torch
from torch import nn


def load_text_encoder(name: str):
    """Return (model, tokenizer) for a Hugging Face text encoder."""
    from transformers import AutoModel, AutoTokenizer

    kwargs = {"normalization": False} if "bertweet" in name else {}  # we normalize in data.py
    tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
    return AutoModel.from_pretrained(name), tokenizer


def clip_processor_class():
    """The pure-PIL CLIP preprocessor, so training (Mac) and serving (no PyTorch) resize images identically."""
    try:
        from transformers import CLIPImageProcessorPil

        return CLIPImageProcessorPil
    except ImportError:  # older transformers
        from transformers import CLIPImageProcessor

        return CLIPImageProcessor


def load_image_encoder(name: str):
    """Return (vision_model, image_processor) for a CLIP checkpoint."""
    from transformers import CLIPVisionModel

    return CLIPVisionModel.from_pretrained(name), clip_processor_class().from_pretrained(name)


def text_features(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Sentence vector = the first token's final hidden state (<s>, RoBERTa's [CLS])."""
    return model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]


def image_features(model: nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
    """CLIP's pooled image vector (layer-normed class token)."""
    return model(pixel_values=pixel_values).pooler_output


def hidden_size(model: nn.Module | None) -> int:
    return int(model.config.hidden_size) if model is not None else 0


def set_trainable(model: nn.Module | None, trainable: bool) -> None:
    if model is not None:
        for p in model.parameters():
            p.requires_grad = trainable
