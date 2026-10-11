"""Find where a post is talking about, and turn that place into map coordinates.

1. Candidates: place names spaCy's NER tags (GPE, LOC, FAC), plus exact
   matches from an optional gazetteer of local names it tends to miss
   (neighborhoods like "Meyerland").
2. Geocode each candidate with OpenStreetMap's Nominatim, bounded to the
   disaster region so "Katy" lands in Texas, not somewhere else.
3. Cache every lookup on disk. Nominatim's usage policy: at most 1 request
   per second and a real User-Agent; repeated runs hit the cache instead.

Used only when building the demo scenario (on your laptop), never live.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

USER_AGENT = "CrisisLens-portfolio/0.1 (https://github.com/kshitideshpande/crisislens)"
NOMINATIM = "https://nominatim.openstreetmap.org/search"


@dataclass
class Region:
    """Bounding box (west, south, east, north) that results must fall inside."""

    name: str
    west: float
    south: float
    east: float
    north: float
    suffix: str = ""  # appended to queries, e.g. "Texas"

    def contains(self, lat: float, lon: float) -> bool:
        return self.south <= lat <= self.north and self.west <= lon <= self.east


HOUSTON = Region("Greater Houston and the upper Texas coast", west=-96.3, south=28.8, east=-93.7, north=30.4,
                 suffix="Texas")


def extract_places(text: str, nlp=None, gazetteer: list[str] | None = None) -> list[str]:
    """Place-name candidates in reading order, without duplicates."""
    found: list[tuple[int, str]] = []
    if nlp is not None:
        for ent in nlp(text).ents:
            if ent.label_ in ("GPE", "LOC", "FAC"):
                found.append((ent.start_char, ent.text.strip(" #.,")))
    for name in gazetteer or []:
        for m in re.finditer(rf"\b{re.escape(name)}\b", text, flags=re.IGNORECASE):
            found.append((m.start(), name))
    seen, out = set(), []
    for _, name in sorted(found):
        key = name.lower()
        if name and key not in seen and not name.startswith("@"):
            seen.add(key)
            out.append(name)
    return out


def load_spacy(model: str = "en_core_web_sm"):
    """Load spaCy if installed; return None (gazetteer-only) otherwise."""
    try:
        import spacy

        return spacy.load(model)
    except (ImportError, OSError):
        print(f"[geo] spaCy model {model!r} not available; using the gazetteer only. "
              f"Install with: pip install spacy && python -m spacy download {model}")
        return None


class Geocoder:
    def __init__(self, cache_path: str | Path, region: Region = HOUSTON,
                 fetch: Callable[[str], list[dict]] | None = None, min_interval: float = 1.1):
        self.cache_path = Path(cache_path)
        self.cache: dict[str, dict | None] = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        self.region = region
        self.fetch = fetch or self._nominatim
        self.min_interval = min_interval
        self._last = 0.0

    def _nominatim(self, query: str) -> list[dict]:
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        r = self.region
        params = {"q": query, "format": "jsonv2", "limit": 3, "bounded": 1,
                  "viewbox": f"{r.west},{r.north},{r.east},{r.south}"}
        req = urllib.request.Request(f"{NOMINATIM}?{urllib.parse.urlencode(params)}", headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read())
        except OSError as e:
            print(f"[geo] lookup failed for {query!r}: {e}")
            return []
        finally:
            self._last = time.time()

    def geocode(self, place: str) -> dict | None:
        """{'name', 'lat', 'lon'} inside the region, or None."""
        key = place.lower()
        if key not in self.cache:
            query = f"{place}, {self.region.suffix}" if self.region.suffix else place
            hit = None
            for res in self.fetch(query):
                lat, lon = float(res["lat"]), float(res["lon"])
                if self.region.contains(lat, lon):
                    hit = {"name": place, "lat": round(lat, 5), "lon": round(lon, 5)}
                    break
            self.cache[key] = hit
            self.save()
        return self.cache[key]

    def locate(self, text: str, nlp=None, gazetteer: list[str] | None = None) -> dict | None:
        """Most specific place we can map: the first candidate that geocodes inside the region."""
        for place in extract_places(text, nlp, gazetteer):
            hit = self.geocode(place)
            if hit:
                return hit
        return None

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, indent=1, sort_keys=True))
