# tests/test_discover_validate.py
from bud.stages.discover_validate import find_near_duplicates, dedup_observations


def test_finds_near_duplicates():
    items = [
        "I need a moment with this",
        "I need a moment with this (Processing lag/Shock)",
        "Something completely different",
    ]
    dupes = find_near_duplicates(items, threshold=0.65)
    assert len(dupes) == 1
    assert dupes[0][0] == 0  # idx_a
    assert dupes[0][1] == 1  # idx_b
    assert dupes[0][2] > 0.65  # similarity


def test_no_false_positives():
    items = ["fractal narrowing", "the corrective pivot", "medium rupture"]
    dupes = find_near_duplicates(items, threshold=0.7)
    assert len(dupes) == 0


def test_dedup_observations_keeps_highest_confidence():
    observations = [
        {"pattern_type": "boundary", "name": "medium rupture",
         "description": "v1", "confidence": 0.8},
        {"pattern_type": "boundary", "name": "medium rupture",
         "description": "v2 better", "confidence": 0.9},
        {"pattern_type": "geometric", "name": "fractal narrowing",
         "description": "zoom", "confidence": 0.85},
    ]
    deduped = dedup_observations(observations)
    assert len(deduped) == 2
    medium = [o for o in deduped if o["name"] == "medium rupture"][0]
    assert medium["confidence"] == 0.9
    assert medium["description"] == "v2 better"


def test_dedup_observations_empty():
    assert dedup_observations([]) == []
