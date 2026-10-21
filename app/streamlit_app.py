"""CrisisLens demo: replay a simulated hurricane, or try the model on your own post.

Local:  MODEL_DIR=exported streamlit run app/streamlit_app.py
Space:  set secrets MODEL_REPO (e.g. kshitid/crisislens-model) and HF_TOKEN (read token).

The Replay tab only needs demo/scenario/replay.json (predictions were computed
ahead of time), so it works even if the model can't be loaded. The Try-it tab
runs the int8 ONNX model live on CPU.
"""

from __future__ import annotations

import csv
import html
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pydeck as pdk
import streamlit as st
from PIL import Image

HERE = Path(__file__).resolve().parent
for candidate in (HERE, HERE.parent / "src"):  # Space layout, then repo layout
    if (candidate / "crisislens" / "inference.py").exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from crisislens.inference import OnnxPredictor  # noqa: E402

SCENARIO_DIR = next((p for p in (Path(os.environ.get("SCENARIO_DIR", "")), HERE / "scenario",
                                 HERE.parent / "demo" / "scenario") if (p / "replay.json").exists()), None)

CATEGORIES = {  # label -> (short name, RGB)
    "affected_individuals": ("People affected", (214, 39, 40)),
    "rescue_volunteering_or_donation_effort": ("Rescue / donations", (255, 127, 14)),
    "infrastructure_and_utility_damage": ("Infrastructure damage", (148, 103, 189)),
    "other_relevant_information": ("Other updates", (31, 119, 180)),
    "not_humanitarian": ("Not humanitarian", (127, 127, 127)),
}
DAMAGE = {"severe_damage": "Severe", "mild_damage": "Mild", "little_or_no_damage": "Little or none"}

st.set_page_config(page_title="CrisisLens", page_icon="🛰️", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 1.6rem; max-width: 1400px;}
.post {border: 1px solid rgba(128,128,128,.25); border-radius: 10px; padding: .7rem .9rem; margin-bottom: .6rem;}
.badge {display: inline-block; padding: .05rem .5rem; border-radius: 999px; font-size: .78rem; margin-right: .3rem;
        color: white;}
.meta {font-size: .8rem; opacity: .7;}
.word {padding: 0 .12rem; border-radius: 3px;}
.summary {border-left: 3px solid #ff7f0e; padding: .2rem .7rem; margin-bottom: .6rem;}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_replay() -> dict | None:
    return json.loads((SCENARIO_DIR / "replay.json").read_text()) if SCENARIO_DIR else None


@st.cache_data
def load_credits() -> list[dict]:
    path = SCENARIO_DIR / "sources.csv" if SCENARIO_DIR else None
    if not path or not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def setting(name: str) -> str | None:
    """Environment variable (local, HF Space), else a Streamlit secret (Streamlit Community Cloud)."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        if name in st.secrets:
            return str(st.secrets[name])
        for section in st.secrets.values():  # also accept the keys under a [section] header
            if hasattr(section, "get") and section.get(name):
                return str(section[name])
    except Exception:  # no secrets file
        pass
    return None


@st.cache_resource(show_spinner="Loading the model…")
def load_predictor() -> tuple[OnnxPredictor | None, str]:
    local = setting("MODEL_DIR")
    try:
        if local:
            return OnnxPredictor(local), ""
        repo = setting("MODEL_REPO")
        if repo:
            from huggingface_hub import snapshot_download

            path = snapshot_download(repo, token=setting("HF_TOKEN"), local_dir="/tmp/crisislens-model")
            return OnnxPredictor(path), ""
        return None, ("No model configured. Locally, set MODEL_DIR=exported. When hosted, add the secrets "
                      'MODEL_REPO = "<user>/crisislens-model" and HF_TOKEN = "<read token>".')
    except Exception as e:  # keep the Replay tab alive even if the model fails
        return None, f"Model failed to load: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def badge(text: str, rgb: tuple[int, int, int]) -> str:
    return f'<span class="badge" style="background: rgb{rgb}">{html.escape(text)}</span>'


def highlighted(words: list, fallback: str) -> str:
    """Words shaded by importance: orange pushed toward the prediction, blue pushed away."""
    if not words:
        return html.escape(fallback)
    top = max(abs(s) for _, s in words) or 1.0
    spans = []
    for w, s in words:
        a = min(1.0, abs(s) / top) * 0.55
        color = f"rgba(255,127,14,{a:.2f})" if s > 0 else f"rgba(31,119,180,{a:.2f})"
        spans.append(f'<span class="word" style="background:{color}">{html.escape(w)}</span>')
    return " ".join(spans)


def with_heatmap(image: Image.Image, heatmap) -> Image.Image:
    """Blend an orange overlay where hiding the region lowered the prediction most."""
    if heatmap is None:
        return image
    h = np.clip(np.asarray(heatmap, dtype=np.float32), 0, None)
    if h.max() <= 0:
        return image
    mask = Image.fromarray((255 * h / h.max()).astype(np.uint8)).resize(image.size, Image.BILINEAR)
    overlay = Image.new("RGB", image.size, (255, 127, 14))
    alpha = Image.fromarray((np.asarray(mask, dtype=np.float32) * 0.6).astype(np.uint8))
    return Image.composite(overlay, image.convert("RGB"), alpha)


def clock(minute: int) -> str:
    return f"Day {minute // 1440 + 1}, {(minute % 1440) // 60:02d}:{minute % 60:02d}"


def post_card(post: dict, show_heatmap: bool) -> None:
    name, rgb = CATEGORIES[post["category"]]
    badges = badge(name, rgb)
    if post["flagged"]:
        badges += badge(f"Urgent {post['p_urgent']:.0%}", (200, 30, 30))
    badges += badge(f"Damage: {DAMAGE.get(post['damage'], post['damage'])}", (90, 90, 90))
    loc = post["location"]["name"] if post.get("location") else "location unknown"
    reliance = f" · image weight {post['image_reliance']:.0%}" if post.get("image_reliance") is not None else ""
    cols = st.columns([3, 2]) if post.get("image") else [st.container()]
    with cols[0]:
        st.markdown(
            f'<div class="post">{badges}<div style="margin:.45rem 0">{highlighted(post["words"], post["text"])}</div>'
            f'<div class="meta">{post["post_id"]} · {clock(post["minute"])} · {html.escape(loc)}{reliance}</div></div>',
            unsafe_allow_html=True)
    if post.get("image"):
        with cols[1]:
            img = Image.open(SCENARIO_DIR / post["image"]).convert("RGB")
            st.image(with_heatmap(img, post.get("heatmap")) if show_heatmap else img, width="stretch")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

def replay_tab(replay: dict) -> None:
    sc = replay["scenario"]
    st.info(f"**{sc['name']}**: {sc['description']} The model's predictions are real; the event is not.", icon="ℹ️")
    duration = sc["duration_minutes"]
    st.session_state.setdefault("minute", min(1440, duration))
    st.session_state.setdefault("playing", False)
    if st.session_state.playing:  # advance before the slider is drawn
        st.session_state.minute = min(duration, st.session_state.minute + 90)
        if st.session_state.minute >= duration:
            st.session_state.playing = False

    c1, c2, c3 = st.columns([1, 1, 6])
    if c1.button("▶ Play" if not st.session_state.playing else "⏸ Pause", width="stretch"):
        st.session_state.playing = not st.session_state.playing
        if st.session_state.playing and st.session_state.minute >= duration:
            st.session_state.minute = 0
        st.rerun()
    if c2.button("⟲ Restart", width="stretch"):
        st.session_state.minute, st.session_state.playing = 0, False
        st.rerun()
    c3.slider("Time since the first post", 0, duration, key="minute", step=30, format="%d min",
              help="Drag to scrub through the storm")

    now = st.session_state.minute
    f1, f2, f3 = st.columns([2, 1, 1])
    chosen = f1.multiselect("Categories", list(CATEGORIES), default=[c for c in CATEGORIES if c != "not_humanitarian"],
                            format_func=lambda c: CATEGORIES[c][0])
    urgent_only = f2.toggle("Urgent only", value=True)
    show_heat = f3.toggle("Image heatmaps", value=True)

    seen = [p for p in replay["posts"] if p["minute"] <= now]
    shown = [p for p in seen if p["category"] in chosen and (p["flagged"] or not urgent_only)]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Clock", clock(now))
    m2.metric("Posts so far", len(seen))
    m3.metric("Flagged urgent", sum(p["flagged"] for p in seen))
    m4.metric("Mapped", sum(1 for p in seen if p.get("location")))

    left, right = st.columns([3, 2], gap="large")
    with left:
        mapped = [p for p in shown if p.get("location")]
        region = sc["region"]
        points = [{"lat": p["location"]["lat"], "lon": p["location"]["lon"], "color": [*CATEGORIES[p["category"]][1], 200],
                   "radius": 900 + 1800 * p["p_urgent"], "text": p["text"], "place": p["location"]["name"],
                   "category": CATEGORIES[p["category"]][0]} for p in mapped]
        st.pydeck_chart(pdk.Deck(
            map_provider="carto", map_style="light",
            initial_view_state=pdk.ViewState(latitude=(region["south"] + region["north"]) / 2,
                                             longitude=(region["west"] + region["east"]) / 2, zoom=8),
            layers=[pdk.Layer("ScatterplotLayer", data=points, get_position="[lon, lat]", get_fill_color="color",
                              get_radius="radius", pickable=True, opacity=0.8, stroked=True,
                              get_line_color=[255, 255, 255], line_width_min_pixels=1)],
            tooltip={"html": "<b>{category}</b> · {place}<br/>{text}", "style": {"maxWidth": "320px"}},
        ), width="stretch")
        legend = " ".join(badge(CATEGORIES[c][0], CATEGORIES[c][1]) for c in chosen)
        st.markdown(f"{legend} <span class='meta'>dot size = urgency</span>", unsafe_allow_html=True)

        st.subheader("Area summaries")
        latest = {}
        for s in replay["summaries"]:
            if s["minute"] <= now:
                latest[s["area"]] = s
        if not latest:
            st.caption("The first summaries appear after 12 hours of posts.")
        ranked = sorted(latest.items(), key=lambda kv: -len(kv[1]["post_ids"]))

        def show(container, area: str, s: dict) -> None:
            container.markdown(f'<div class="summary"><b>{html.escape(area)}</b> <span class="meta">as of '
                               f'{clock(s["minute"])} · {s["source"]}</span><br/>{html.escape(s["summary"])}</div>',
                               unsafe_allow_html=True)

        for area, s in ranked[:6]:
            show(st, area, s)
        if len(ranked) > 6:
            with st.expander(f"{len(ranked) - 6} more areas"):
                for area, s in ranked[6:]:
                    show(st, area, s)

    with right:
        st.subheader(f"Feed ({len(shown)})")
        st.caption("Highlighted words drove the category; orange image regions mattered most.")
        for post in sorted(shown, key=lambda p: (-p["flagged"], -p["minute"]))[:25]:
            post_card(post, show_heat)

    if st.session_state.playing:
        time.sleep(0.8)
        st.rerun()


def try_tab() -> None:
    predictor, error = load_predictor()
    if predictor is None:
        load_predictor.clear()  # don't cache a failure: retry on the next run (e.g. after secrets are added)
        st.warning(error)
        return
    st.write("Write a post like one you might see during a disaster. Add a photo or leave it text-only.")
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        text = st.text_area("Post text", "Water is coming into our house in Dickinson, two kids with us, need a boat",
                            height=110, max_chars=400)
        upload = st.file_uploader("Photo (optional)", type=["jpg", "jpeg", "png"])
        go = st.button("Classify", type="primary")
    image = Image.open(io.BytesIO(upload.getvalue())).convert("RGB") if upload else None
    if image is not None:
        c2.image(image, caption="your photo", width="stretch")
    if not go:
        return
    if not text.strip() and image is None:
        st.error("Add some text or a photo.")
        return
    with st.spinner("Running the model and computing explanations…"):
        t0 = time.perf_counter()
        exp = predictor.explain(text, image, task="humanitarian")
        ms = (time.perf_counter() - t0) * 1000
    pred = exp["prediction"]
    name, rgb = CATEGORIES[pred["humanitarian"]["label"]]
    head = badge(f"{name} {pred['humanitarian']['confidence']:.0%}", rgb)
    if pred.get("flagged") is not None:
        head += badge(f"{'Urgent' if pred['flagged'] else 'Not urgent'} ({pred['p_urgent']:.0%})",
                      (200, 30, 30) if pred["flagged"] else (90, 90, 90))
    head += badge(f"Informative: {pred['informative']['label'].replace('_', ' ')}", (60, 60, 60))
    head += badge(f"Damage: {DAMAGE.get(pred['damage']['label'])}", (90, 90, 90))
    with c1:
        st.markdown(head, unsafe_allow_html=True)
        st.markdown(f'<div class="post">{highlighted(exp["words"], text)}</div>', unsafe_allow_html=True)
        if pred.get("image_reliance") is not None and image is not None:
            st.progress(pred["image_reliance"], text=f"Image weight in the decision: {pred['image_reliance']:.0%}")
        st.caption(f"{ms:.0f} ms on CPU, including {len(exp['words']) + 1 + (50 if image else 0)} model runs for the explanation")
        st.bar_chart({CATEGORIES[k][0]: v for k, v in pred["humanitarian"]["probs"].items()}, horizontal=True)
    if image is not None:
        c2.image(with_heatmap(image, exp["heatmap"]), caption="orange = regions that mattered", width="stretch")


def about_tab() -> None:
    results = next((p for p in (HERE / "results.md", HERE.parent / "outputs" / "results.md") if p.exists()), None)
    st.markdown("""
### How it works
Each post's **text** goes through BERTweet (a language model pretrained on tweets) and its **photo** through CLIP's
vision encoder. A learned **gate** decides, for every post, how much to trust each one, and three heads predict:
is it informative, what kind of need is it, and how severe is the damage. Probabilities are calibrated with
temperature scaling, and a post is flagged **urgent** when P(people affected, rescue, or damage) passes a threshold
chosen to catch 90% of urgent posts on held-out data.

### Data and responsible use
The model was trained on [CrisisMMD](https://crisisnlp.qcri.org/crisismmd.html), human-labeled tweets and images from
seven 2017 disasters. Its terms restrict the data to research, so no CrisisMMD posts appear here: the replay is a
fictional storm with posts written for this demo and public-domain photos. CrisisLens is a triage aid. It ranks what
people look at first; it never decides who gets help.
""")
    if results:
        st.markdown("### Results\n" + results.read_text())
    credits = load_credits()
    if credits:
        with st.expander(f"Photo credits ({len(credits)} public-domain images, Wikimedia Commons)"):
            for c in credits:
                st.markdown(f"- {c['post_id']}: [{c['title']}]({c['commons_url']}) · {c['author']} · {c['license']}")
    st.markdown("Code: [github.com/kshitideshpande/crisislens](https://github.com/kshitideshpande/crisislens)")


def main() -> None:
    st.title("CrisisLens")
    st.caption("Multimodal triage of disaster social media posts: text + image → urgency, need type, damage, location")
    replay = load_replay()
    tabs = st.tabs(["Replay a storm", "Try it", "About"])
    with tabs[0]:
        if replay:
            replay_tab(replay)
        else:
            st.warning("No replay found. Build it with `python scripts/build_scenario.py build`.")
    with tabs[1]:
        try_tab()
    with tabs[2]:
        about_tab()


main()
