"""Tweet normalization shared by training and inference (stdlib only, so the demo Space can ship it)."""

from __future__ import annotations

import re

_URL = re.compile(r"https?://\S+|www\.\S+")
_USER = re.compile(r"@\w+")
_SPACE = re.compile(r"\s+")


def normalize_tweet(text: str) -> str:
    """Light normalization matching BERTweet's conventions.

    URLs become HTTPURL and mentions become @USER, which keeps the signal that
    a link or mention was present without memorizing specific accounts.
    Hashtags are kept: #HurricaneHarvey is real information.
    """
    if not isinstance(text, str):
        return ""
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = _URL.sub("HTTPURL", text)
    text = _USER.sub("@USER", text)
    text = re.sub(r"^RT\s+", "", text)
    return _SPACE.sub(" ", text).strip()
