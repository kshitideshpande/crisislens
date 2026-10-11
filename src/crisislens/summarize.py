"""Short, grounded situation summaries per area, for responders.

The LLM sees ONLY the flagged posts for one area and must cite every claim
with post ids like [p12]. Afterwards we check the citations: sentences citing
ids that were not provided are dropped, and if nothing valid is left we fall
back to a deterministic template. So a summary can be incomplete, but it
can't cite posts that don't exist.

Providers (first that works): Gemini -> Groq -> template. Keys come from the
environment (GEMINI_API_KEY, GROQ_API_KEY); with no keys, the template is used.
Model names change often, so they're overridable: GEMINI_MODEL, GROQ_MODEL.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from collections import Counter
from typing import Callable

PROMPT = """You are helping disaster responders. Below are social media posts from one area ({area}),
already flagged by a classifier as possibly urgent. Write a situation summary of 2 to 4 sentences.

Rules:
- Use only facts stated in the posts. Do not guess numbers, causes, or locations.
- End every sentence with the ids of the posts it is based on, like [p03] or [p03, p07].
- Prioritize people needing rescue or medical help, then damage to roads, power and buildings, then resources offered.
- Do not include personal names or descriptions that identify individuals.

Posts:
{posts}

Summary:"""

PRETTY = {
    "affected_individuals": "people affected or injured",
    "infrastructure_and_utility_damage": "infrastructure damage",
    "rescue_volunteering_or_donation_effort": "rescue, volunteering or donations",
    "other_relevant_information": "other updates",
    "not_humanitarian": "other posts",
}

_CITE = re.compile(r"\[(p\d+(?:\s*,\s*p\d+)*)\]")


def _post_lines(posts: list[dict]) -> str:
    return "\n".join(f"[{p['post_id']}] ({PRETTY.get(p['category'], p['category'])}) {p['text']}" for p in posts)


def validate_citations(summary: str, valid_ids: set[str]) -> str:
    """Keep only sentences whose citations all refer to provided posts (and that cite something)."""
    sentences = re.split(r"(?<=[.!?\]])\s+", summary.strip())
    kept = []
    for s in sentences:
        cited = {pid.strip() for group in _CITE.findall(s) for pid in group.split(",")}
        if cited and cited <= valid_ids:
            kept.append(s.strip())
    return " ".join(kept)


def template_summary(area: str, posts: list[dict]) -> str:
    """No-LLM fallback: counts by category plus the most urgent post, with citations."""
    counts = Counter(p["category"] for p in posts)
    parts = [f"{n} {PRETTY.get(c, c)}" for c, n in counts.most_common()]
    top = max(posts, key=lambda p: p.get("p_urgent", 0))
    quote = top["text"] if len(top["text"]) <= 140 else top["text"][:137] + "..."
    ids = ", ".join(p["post_id"] for p in posts[:6])
    where = "with no mappable location" if area == "Unlocated" else f"near {area}"
    return (f"{len(posts)} flagged post{'s' if len(posts) != 1 else ''} {where}: {', '.join(parts)} [{ids}]. "
            f"Most urgent: \"{quote}\" [{top['post_id']}].")


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

def _post_json(url: str, payload: dict, headers: dict, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def gemini(prompt: str) -> str:
    key = os.environ["GEMINI_API_KEY"]
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    data = _post_json(url, {"contents": [{"parts": [{"text": prompt}]}],
                            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 300}},
                      {"x-goog-api-key": key})
    return data["candidates"][0]["content"]["parts"][0]["text"]


def groq(prompt: str) -> str:
    key = os.environ["GROQ_API_KEY"]
    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    data = _post_json("https://api.groq.com/openai/v1/chat/completions",
                      {"model": model, "temperature": 0.2, "max_tokens": 300,
                       "messages": [{"role": "user", "content": prompt}]},
                      {"Authorization": f"Bearer {key}"})
    return data["choices"][0]["message"]["content"]


def available_providers() -> list[tuple[str, Callable[[str], str]]]:
    out = []
    if os.environ.get("GEMINI_API_KEY"):
        out.append(("gemini", gemini))
    if os.environ.get("GROQ_API_KEY"):
        out.append(("groq", groq))
    return out


def summarize_area(area: str, posts: list[dict], providers: list | None = None) -> dict:
    """posts: dicts with post_id, text, category, p_urgent. Returns {'summary', 'source', 'post_ids'}."""
    ranked = sorted(posts, key=lambda p: -p.get("p_urgent", 0))
    posts = ranked[:15]  # most urgent first, so the prompt stays short
    ids = {p["post_id"] for p in posts}
    prompt = PROMPT.format(area=area, posts=_post_lines(posts))
    for name, call in providers if providers is not None else available_providers():
        try:
            text = validate_citations(call(prompt), ids)
            if text:
                return {"summary": text, "source": name, "post_ids": sorted(ids)}
        except Exception as e:  # network errors, quota limits, schema changes: try the next one
            print(f"[summarize] {name} failed for {area}: {type(e).__name__}: {e}")
    return {"summary": template_summary(area, ranked), "source": "template", "post_ids": sorted(ids)}
