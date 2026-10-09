"""A tiny fake CrisisMMD folder, so tests run in CI without the real 1.8 GB dataset."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from crisislens.data import TASKS

HEADER = "event_name\ttweet_id\timage_id\ttweet_text\timage\tlabel\tlabel_text\tlabel_image\tlabel_text_image\n"

TEMPLATES = {
    "informative": "Flood waters rising near the bridge, roads closed",
    "not_informative": "Thinking of everyone today, stay safe friends",
    "affected_individuals": "Two people injured and a family missing after the quake",
    "infrastructure_and_utility_damage": "Power lines down and the bridge collapsed on Main St",
    "not_humanitarian": "Great game last night, what a finish",
    "other_relevant_information": "Storm expected to make landfall tomorrow evening",
    "rescue_volunteering_or_donation_effort": "Volunteers needed, donate water and blankets at the shelter",
    "little_or_no_damage": "Some leaves on the road but houses are fine",
    "mild_damage": "A few roof shingles torn off on our street",
    "severe_damage": "Entire block destroyed, homes flattened",
}


def _write_split(path: Path, labels: tuple[str, ...], n: int, rng: random.Random) -> None:
    rows = [HEADER]
    for i in range(n):
        label = labels[i % len(labels)]
        tid = 900000000000000000 + rng.randint(0, 10**9)
        text = f"RT @someone {TEMPLATES[label]} #Harvey https://t.co/x{i}"
        image = f"data_image/hurricane_harvey/1_9_2017/{tid}_0.jpg"
        rows.append(f"hurricane_harvey\t{tid}\t{tid}_0\t{text}\t{image}\t{label}\t{label}\t{label}\tPositive\n")
    path.write_text("".join(rows))


@pytest.fixture
def fake_data_dir(tmp_path: Path) -> Path:
    rng = random.Random(0)
    split_dir = tmp_path / "crisismmd_datasplit_agreed_label"
    split_dir.mkdir()
    for spec in TASKS.values():
        for split, n in (("train", 60), ("dev", 20), ("test", 20)):
            _write_split(split_dir / spec.file_pattern.format(split=split), spec.expected_labels, n, rng)
    return tmp_path
