from __future__ import annotations

import base64
import importlib.util
import io
import json
import shutil
import sys
from pathlib import Path

import pytest
from PIL import Image

from crisislens.geo import HOUSTON, Geocoder, extract_places
from crisislens.summarize import summarize_area, template_summary, validate_citations

ROOT = Path(__file__).resolve().parents[1]


def _png_b64(color=(200, 40, 40)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (40, 30), color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Geocoding (network replaced by a fake)
# ---------------------------------------------------------------------------

def test_extract_places_gazetteer_order_and_dedup():
    text = "Flooding in Meyerland and meyerland again, also Katy. Thanks @Houston"
    assert extract_places(text, None, ["Katy", "Meyerland"]) == ["Meyerland", "Katy"]


def test_geocoder_bounds_and_cache(tmp_path):
    calls = []

    def fake(query):
        calls.append(query)
        if query.startswith("Katy"):
            return [{"lat": "45.0", "lon": "-100.0"}, {"lat": "29.79", "lon": "-95.82"}]  # first is outside Texas
        return []

    g = Geocoder(tmp_path / "cache.json", HOUSTON, fetch=fake)
    assert g.locate("Shelter open in Katy", gazetteer=["Katy"]) == {"name": "Katy", "lat": 29.79, "lon": -95.82}
    assert g.geocode("Nowhere") is None
    Geocoder(tmp_path / "cache.json", HOUSTON, fetch=fake).geocode("Katy")  # served from the cache file
    assert calls == ["Katy, Texas", "Nowhere, Texas"]


# ---------------------------------------------------------------------------
# Summaries: citations are checked, failures fall back to the template
# ---------------------------------------------------------------------------

POSTS = [
    {"post_id": "p01", "text": "Family trapped in attic", "category": "affected_individuals", "p_urgent": 0.95},
    {"post_id": "p02", "text": "Bridge washed out", "category": "infrastructure_and_utility_damage", "p_urgent": 0.8},
]


def test_validate_citations_drops_invented_ids():
    llm = "A family is trapped [p01]. A hospital collapsed [p99]. Roads are closed. The bridge is out [p02, p01]."
    assert validate_citations(llm, {"p01", "p02"}) == "A family is trapped [p01]. The bridge is out [p02, p01]."


def test_summary_uses_llm_then_falls_back():
    ok = summarize_area("Dickinson", POSTS, [("fake", lambda prompt: "People are trapped in an attic [p01].")])
    assert ok["source"] == "fake" and ok["summary"].endswith("[p01].")

    def broken(prompt):
        raise TimeoutError("no network")

    bad = summarize_area("Dickinson", POSTS, [("broken", broken), ("liar", lambda p: "Made up [p77].")])
    assert bad["source"] == "template" and "[p01]" in bad["summary"]
    assert template_summary("Unlocated", POSTS).startswith("2 flagged posts with no mappable location")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@pytest.fixture
def client(exported_model, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("MODEL_DIR", str(exported_model))
    from service import api

    api.get_predictor.cache_clear()
    return TestClient(api.app)


def test_api_health_meta(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    meta = client.get("/meta").json()
    assert meta["tasks"] == ["informative", "humanitarian", "damage"] and meta["flag"]["task"] == "humanitarian"


def test_api_classify_batch(client):
    r = client.post("/classify", json={"posts": [
        {"text": "Volunteers needed at the shelter", "image_base64": _png_b64()},
        {"text": "Bridge collapsed"},
        {"image_base64": "data:image/png;base64," + _png_b64((40, 40, 200))},
    ]})
    assert r.status_code == 200
    res = r.json()["results"]
    assert len(res) == 3 and all("flagged" in x for x in res)
    assert res[1]["image_reliance"] == pytest.approx(0.0, abs=1e-5)


def test_api_rejects_bad_input(client):
    assert client.post("/classify", json={"posts": []}).status_code == 422
    assert client.post("/classify", json={"posts": [{"text": "  "}]}).status_code == 422
    assert client.post("/classify", json={"posts": [{"text": "x", "image_base64": "%%%"}]}).status_code == 422
    assert client.post("/classify", json={"posts": [{"text": "x"}] * 33}).status_code == 422
    assert client.post("/explain", json={"text": "x", "task": "nope"}).status_code == 422


def test_api_explain(client):
    r = client.post("/explain", json={"text": "power lines down near the bridge", "image_base64": _png_b64()})
    body = r.json()
    assert r.status_code == 200 and len(body["words"]) == 6 and len(body["heatmap"]) == 7


# ---------------------------------------------------------------------------
# Scenario builder + Streamlit app
# ---------------------------------------------------------------------------

def _load_build_scenario():
    spec = importlib.util.spec_from_file_location("build_scenario", ROOT / "scripts" / "build_scenario.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def built_scenario(exported_model, tmp_path_factory) -> Path:
    bs = _load_build_scenario()
    scen = tmp_path_factory.mktemp("scenario")
    shutil.copy(ROOT / "demo" / "scenario" / "posts.csv", scen / "posts.csv")
    (scen / "images").mkdir()
    for pid in ("p03", "p05", "p10"):
        Image.new("RGB", (120, 90), (90, 120, 160)).save(scen / "images" / f"{pid}.jpg")

    def fake(query):  # every place lands in central Houston
        return [{"lat": "29.76", "lon": "-95.37"}]

    geocoder = Geocoder(scen / "cache.json", HOUSTON, fetch=fake, min_interval=0)
    bs.build(exported_model, summary_every=720, no_llm=True, scenario=scen, geocoder=geocoder)
    return scen


def test_scenario_replay(built_scenario):
    replay = json.loads((built_scenario / "replay.json").read_text())
    posts = replay["posts"]
    assert len(posts) == 62 and replay["scenario"]["simulated"] is True
    with_img = [p for p in posts if p["image"]]
    assert {p["post_id"] for p in with_img} == {"p03", "p05", "p10"} and len(with_img[0]["heatmap"]) == 7
    assert all(p["location"] for p in posts if "Galveston" in p["text"])
    assert replay["summaries"] and all(s["source"] == "template" for s in replay["summaries"])


def test_streamlit_app_renders(built_scenario, exported_model, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("SCENARIO_DIR", str(built_scenario))
    monkeypatch.setenv("MODEL_DIR", str(exported_model))
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    assert [t.label for t in at.tabs] == ["Replay a storm", "Try it", "About"]
    assert at.metric[1].value == str(sum(1 for p in json.loads((built_scenario / "replay.json").read_text())["posts"]
                                         if p["minute"] <= 1440))

    at.slider[0].set_value(3000).run()  # scrub forward
    assert not at.exception
    at.button[2].click().run()  # "Classify" in the Try-it tab
    assert not at.exception, at.exception
