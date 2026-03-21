"""Validation, compaction, and chunk-feedback analysis for discovery maps."""

from difflib import SequenceMatcher


def find_near_duplicates(
    items: list[str], threshold: float = 0.7
) -> list[tuple[int, int, float]]:
    """Find near-duplicate string pairs using sequence similarity.

    Args:
        items: List of strings to compare pairwise.
        threshold: Minimum similarity ratio (0-1) to flag as duplicate.

    Returns:
        List of (idx_a, idx_b, similarity) tuples above threshold.
    """
    duplicates = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            ratio = SequenceMatcher(None, items[i], items[j]).ratio()
            if ratio >= threshold:
                duplicates.append((i, j, round(ratio, 4)))
    return duplicates


def dedup_observations(observations: list[dict]) -> list[dict]:
    """Deduplicate observations by pattern_type + name, keeping highest confidence.

    Args:
        observations: List of observation dicts with pattern_type, name, confidence.

    Returns:
        Deduplicated list, one entry per unique (pattern_type, name) pair.
    """
    best: dict[tuple[str, str], dict] = {}
    for obs in observations:
        key = (obs.get("pattern_type", ""), obs.get("name", ""))
        existing = best.get(key)
        if existing is None or obs.get("confidence", 0) > existing.get("confidence", 0):
            best[key] = obs
    return list(best.values())
