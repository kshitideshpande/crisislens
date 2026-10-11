"""Build the simulated disaster replay for the demo.

Step 1 (once): download public-domain photos for the posts that have one.
    python scripts/build_scenario.py fetch-images
    -> demo/scenario/images/<post_id>.jpg, demo/scenario/sources.csv (credits + licenses),
       demo/scenario/contact_sheet.jpg (check every photo before publishing!)
    Re-fetch one photo you don't like:  ... fetch-images --only p12 --skip 1

Step 2: score every post with the exported model, explain it, map it, summarize areas.
    python scripts/build_scenario.py build --model exported/
    -> demo/scenario/replay.json   (everything the app plays back)

The scenario ("Hurricane Delphine") is fictional and the posts were written for
this demo. No CrisisMMD data is used here, so the result is safe to publish.
Photos come only from Wikimedia Commons files marked public domain or CC0
(mostly US government photos, e.g. FEMA, US Coast Guard, National Guard).
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "demo" / "scenario"
USER_AGENT = "CrisisLens-portfolio/0.1 (https://github.com/kshitideshpande/crisislens)"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
SENSITIVE = re.compile(r"\b(body|bodies|dead|death|deceased|corpse|funeral|morgue|casualt)", re.IGNORECASE)

# Local names spaCy often misses; matched exactly in the post text.
GAZETTEER = [
    "Galveston", "Galveston Island", "Pearland", "Katy", "League City", "Dickinson", "Friendswood", "Meyerland",
    "Port Arthur", "Pasadena", "Kingwood", "Deer Park", "Clear Lake", "Baytown", "Sugar Land", "Beaumont", "Kemah",
    "Freeport", "Columbia Lakes", "Braeswood", "Washington Ave", "Westpark", "Bellfort Ave", "Telephone Rd",
    "Spencer Hwy", "Brays Bayou", "Kirby", "Fuqua", "Main and Holcombe", "Beltway 8", "Houston",
]
CATEGORY_SHORT = {
    "affected_individuals": "People affected",
    "infrastructure_and_utility_damage": "Infrastructure damage",
    "rescue_volunteering_or_donation_effort": "Rescue / donations",
    "other_relevant_information": "Other updates",
    "not_humanitarian": "Not humanitarian",
}


def read_posts(scenario: Path = SCENARIO) -> list[dict]:
    with open(scenario / "posts.csv", newline="") as f:
        return [{**r, "minute": int(r["minute"])} for r in csv.DictReader(f)]


# ---------------------------------------------------------------------------
# Step 1: public-domain photos from Wikimedia Commons
# ---------------------------------------------------------------------------

def commons_search(query: str, limit: int = 25) -> list[dict]:
    params = {"action": "query", "format": "json", "generator": "search", "gsrnamespace": 6,
              "gsrsearch": f"{query} filetype:bitmap", "gsrlimit": limit, "prop": "imageinfo",
              "iiprop": "url|extmetadata|mime", "iiurlwidth": 800}
    req = urllib.request.Request(f"{COMMONS_API}?{urllib.parse.urlencode(params)}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        pages = json.loads(resp.read()).get("query", {}).get("pages", {})
    return sorted(pages.values(), key=lambda p: p.get("index", 0))


def _plain(value: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", value or "")).strip()


def usable(page: dict) -> dict | None:
    """Return credit info if the file is a public-domain/CC0 JPEG or PNG with no sensitive keywords."""
    info = (page.get("imageinfo") or [{}])[0]
    meta = info.get("extmetadata", {})
    short = _plain(meta.get("LicenseShortName", {}).get("value", ""))
    code = _plain(meta.get("License", {}).get("value", "")).lower()
    if not (code.startswith("pd") or code == "cc0" or "public domain" in short.lower() or short.upper() == "CC0"):
        return None
    if info.get("mime") not in ("image/jpeg", "image/png"):
        return None
    text = f"{page.get('title', '')} {_plain(meta.get('ImageDescription', {}).get('value', ''))}"
    if SENSITIVE.search(text):
        return None
    return {
        "title": page["title"].removeprefix("File:"),
        "commons_url": info.get("descriptionurl", ""),
        "download_url": info.get("thumburl") or info.get("url"),
        "author": _plain(meta.get("Artist", {}).get("value", "")) or "unknown",
        "license": short or code,
        "license_url": _plain(meta.get("LicenseUrl", {}).get("value", "")),
    }


def pick_image(query: str, used: set[str], search=commons_search, skip: int = 0) -> dict | None:
    """First usable, unused file for the query, trying a simpler query if needed."""
    queries = [query, " ".join(query.split()[:3])]
    for q in queries:
        candidates = [c for c in (usable(p) for p in search(q)) if c and c["title"] not in used]
        if len(candidates) > skip:
            return candidates[skip]
    return None


def contact_sheet(rows: list[dict], path: Path) -> None:
    from PIL import Image, ImageDraw

    thumbs = []
    for r in rows:
        try:
            im = Image.open(SCENARIO / r["image_file"]).convert("RGB")
            im.thumbnail((240, 180))
            canvas = Image.new("RGB", (240, 200), "white")
            canvas.paste(im, ((240 - im.width) // 2, 0))
            ImageDraw.Draw(canvas).text((4, 184), r["post_id"], fill="black")
            thumbs.append(canvas)
        except OSError:
            pass
    if not thumbs:
        return
    cols = 6
    sheet = Image.new("RGB", (240 * cols, 200 * ((len(thumbs) + cols - 1) // cols)), "white")
    for i, t in enumerate(thumbs):
        sheet.paste(t, ((i % cols) * 240, (i // cols) * 200))
    sheet.save(path, quality=85)


def fetch_images(only: list[str] | None, skip: int) -> None:
    images_dir = SCENARIO / "images"
    images_dir.mkdir(exist_ok=True)
    sources_path = SCENARIO / "sources.csv"
    existing = {}
    if sources_path.exists():
        with open(sources_path, newline="") as f:
            existing = {r["post_id"]: r for r in csv.DictReader(f)}

    for post in read_posts():
        pid, query = post["post_id"], post["image_query"].strip()
        if not query or (only and pid not in only) or (not only and pid in existing):
            continue
        used = {r["title"] for k, r in existing.items() if k != pid}
        choice = pick_image(query, used, skip=skip)
        if choice is None:
            print(f"{pid}: nothing public-domain found for {query!r}; edit image_query in posts.csv")
            continue
        req = urllib.request.Request(choice["download_url"], headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as resp:
            (images_dir / f"{pid}.jpg").write_bytes(resp.read())
        existing[pid] = {"image_file": f"images/{pid}.jpg", "post_id": pid, **{k: choice[k] for k in
                         ("title", "commons_url", "author", "license", "license_url")}}
        print(f"{pid}: {choice['title']}  ({choice['license']})")
        time.sleep(1)  # be polite to Wikimedia

    rows = sorted(existing.values(), key=lambda r: r["post_id"])
    with open(sources_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_file", "post_id", "title", "commons_url", "author", "license",
                                          "license_url"])
        w.writeheader()
        w.writerows(rows)
    contact_sheet(rows, SCENARIO / "contact_sheet.jpg")
    print(f"\n{len(rows)} photos credited in {sources_path.relative_to(ROOT)}. "
          f"Open demo/scenario/contact_sheet.jpg and re-fetch any photo that doesn't fit its post.")


# ---------------------------------------------------------------------------
# Step 2: score, explain, map, summarize
# ---------------------------------------------------------------------------

def build(model_dir: Path, summary_every: int, no_llm: bool, scenario: Path = SCENARIO, geocoder=None) -> dict:
    from PIL import Image

    from crisislens.geo import HOUSTON, Geocoder, load_spacy
    from crisislens.inference import OnnxPredictor
    from crisislens.summarize import available_providers, summarize_area

    predictor = OnnxPredictor(model_dir)
    nlp = load_spacy()
    geocoder = geocoder or Geocoder(scenario / "geocode_cache.json", HOUSTON)
    out_posts = []
    for post in read_posts(scenario):
        img_path = scenario / "images" / f"{post['post_id']}.jpg"
        image = Image.open(img_path).convert("RGB") if post["image_query"] and img_path.exists() else None
        exp = predictor.explain(post["text"], image, task="humanitarian")
        pred = exp["prediction"]
        out_posts.append({
            "post_id": post["post_id"], "minute": post["minute"], "text": post["text"],
            "image": f"images/{post['post_id']}.jpg" if image is not None else None,
            "category": pred["humanitarian"]["label"], "category_confidence": pred["humanitarian"]["confidence"],
            "informative": pred["informative"]["label"], "damage": pred["damage"]["label"],
            "p_urgent": pred.get("p_urgent", 0.0), "flagged": pred.get("flagged", False),
            "image_reliance": pred["image_reliance"],
            "probs": {t: pred[t]["probs"] for t in predictor.tasks},
            "words": [[w, round(s, 4)] for w, s in exp["words"]],
            "heatmap": exp["heatmap"].round(4).tolist() if exp["heatmap"] is not None else None,
            "location": geocoder.locate(post["text"], nlp, GAZETTEER),
        })
        loc = out_posts[-1]["location"]
        print(f"{post['post_id']} {'FLAG' if out_posts[-1]['flagged'] else '    '} "
              f"{CATEGORY_SHORT[out_posts[-1]['category']]:<22} {loc['name'] if loc else '-'}")

    providers = [] if no_llm else available_providers()
    print(f"\nsummaries via: {[n for n, _ in providers] or ['template']}")
    summaries = []
    end = max(p["minute"] for p in out_posts)
    for checkpoint in range(summary_every, end + summary_every, summary_every):
        by_area: dict[str, list[dict]] = {}
        for p in out_posts:
            if p["flagged"] and p["minute"] <= checkpoint:
                by_area.setdefault(p["location"]["name"] if p["location"] else "Unlocated", []).append(p)
        for area, posts in sorted(by_area.items()):
            s = summarize_area(area, posts, providers)
            summaries.append({"minute": checkpoint, "area": area, **s})

    meta = predictor.meta
    replay = {
        "scenario": {
            "name": "Hurricane Delphine (simulated)",
            "description": "A fictional Category 3 hurricane making landfall near Galveston, Texas. Posts were "
                           "written for this demo; photos are public-domain images from past real storms.",
            "region": vars(HOUSTON), "simulated": True, "duration_minutes": end,
        },
        "model": {"name": meta.get("name"), "flag_threshold": (meta.get("flag") or {}).get("threshold"),
                  "label_maps": meta["label_maps"]},
        "posts": out_posts,
        "summaries": summaries,
    }
    (scenario / "replay.json").write_text(json.dumps(replay, indent=1))
    n_flag = sum(p["flagged"] for p in out_posts)
    print(f"\nwrote {scenario / 'replay.json'}: {len(out_posts)} posts, {n_flag} flagged, {len(summaries)} summaries")
    return replay


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch-images")
    f.add_argument("--only", nargs="+", help="post ids to (re)fetch, e.g. p12 p30")
    f.add_argument("--skip", type=int, default=0, help="skip the first N matches (to get a different photo)")
    b = sub.add_parser("build")
    b.add_argument("--model", default="exported", type=Path)
    b.add_argument("--summary-every", type=int, default=720, help="minutes between summary checkpoints")
    b.add_argument("--no-llm", action="store_true", help="template summaries only")
    b.add_argument("--scenario-dir", type=Path, default=SCENARIO)
    args = ap.parse_args()
    if args.cmd == "fetch-images":
        fetch_images(args.only, args.skip)
    else:
        build(args.model, args.summary_every, args.no_llm, args.scenario_dir)


if __name__ == "__main__":
    main()
