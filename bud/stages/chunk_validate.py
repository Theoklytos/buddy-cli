"""Structural validation and repair for chunked conversation output."""

from collections import Counter


def validate_chunks(chunks: list[dict], num_turns: int) -> dict:
    """Check structural integrity of chunks for a single conversation.

    Args:
        chunks: List of chunk dicts, each with a "turns" key (list[int]).
        num_turns: Total number of turns in the source conversation.

    Returns:
        Dict with is_valid, coverage_ratio, missing_turns, overlapping_turns,
        out_of_bounds, non_contiguous_chunks.
    """
    all_turns: list[int] = []
    non_contiguous: list[int] = []

    for idx, chunk in enumerate(chunks):
        turns = sorted(chunk.get("turns", []))
        all_turns.extend(turns)
        if turns and turns != list(range(min(turns), max(turns) + 1)):
            non_contiguous.append(idx)

    counter = Counter(all_turns)
    expected = set(range(num_turns))
    covered = set(counter.keys()) & expected
    missing = sorted(expected - covered)
    overlapping = {t: c for t, c in counter.items() if c > 1 and t < num_turns}
    oob = sorted(t for t in counter if t >= num_turns)

    coverage = len(covered) / num_turns if num_turns > 0 else 0.0
    is_valid = (
        not missing
        and not overlapping
        and not oob
        and not non_contiguous
        and coverage == 1.0
    )

    return {
        "is_valid": is_valid,
        "coverage_ratio": coverage,
        "missing_turns": missing,
        "overlapping_turns": overlapping,
        "out_of_bounds": oob,
        "non_contiguous_chunks": non_contiguous,
    }
