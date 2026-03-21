"""Structural validation and repair for chunked conversation output."""

import copy
import uuid as uuid_lib
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


def compute_structural_score(validation: dict) -> float:
    """Convert a validation report into a 0-1 structural quality score.

    Args:
        validation: Dict from validate_chunks().

    Returns:
        Float between 0.0 and 1.0.
    """
    coverage = validation["coverage_ratio"]
    overlap_penalty = min(1.0, len(validation["overlapping_turns"]) * 0.05)
    gap_penalty = min(1.0, len(validation["missing_turns"]) * 0.02)
    oob_penalty = min(1.0, len(validation["out_of_bounds"]) * 0.1)
    return round(max(0.0, coverage - overlap_penalty - gap_penalty - oob_penalty), 4)


def repair_chunks(
    chunks: list[dict], num_turns: int, conversation: dict
) -> list[dict]:
    """Repair structural issues in chunks for a single conversation.

    Steps: remove OOB indices, resolve overlaps by proximity, split
    non-contiguous chunks, fill gaps with stasis-pulse chunks.

    Args:
        chunks: List of chunk dicts with "turns" key.
        num_turns: Total turns in the conversation.
        conversation: The source conversation dict (for text assembly).

    Returns:
        New list of structurally valid chunk dicts.
    """
    if not chunks:
        runs = _split_into_runs(list(range(num_turns)))
        return [_make_gap_chunk(run, conversation) for run in runs]

    working = copy.deepcopy(chunks)

    # Step 1: Remove out-of-bounds indices
    for chunk in working:
        chunk["turns"] = [t for t in chunk["turns"] if 0 <= t < num_turns]

    # Step 2: Resolve overlaps by proximity
    turn_to_chunks: dict[int, list[int]] = {}
    for idx, chunk in enumerate(working):
        for t in chunk["turns"]:
            turn_to_chunks.setdefault(t, []).append(idx)

    for turn, chunk_indices in turn_to_chunks.items():
        if len(chunk_indices) <= 1:
            continue
        best_idx = chunk_indices[0]
        best_dist = float("inf")
        for ci in chunk_indices:
            other_turns = [t for t in working[ci]["turns"] if t != turn]
            if other_turns:
                avg_dist = sum(abs(t - turn) for t in other_turns) / len(other_turns)
            else:
                avg_dist = float("inf")
            if avg_dist < best_dist:
                best_dist = avg_dist
                best_idx = ci
        for ci in chunk_indices:
            if ci != best_idx:
                working[ci]["turns"] = [t for t in working[ci]["turns"] if t != turn]

    # Step 3: Sort turns and split non-contiguous chunks
    split_chunks = []
    for chunk in working:
        turns = sorted(chunk["turns"])
        if not turns:
            continue
        runs = _split_into_runs(turns)
        for run in runs:
            new_chunk = dict(chunk)
            new_chunk["turns"] = run
            new_chunk["chunk_id"] = str(uuid_lib.uuid4())
            new_chunk["text"] = " ".join(
                conversation["turns"][t]["text"] for t in run if t < len(conversation["turns"])
            )
            split_chunks.append(new_chunk)

    # Step 4: Fill gaps
    covered = set()
    for chunk in split_chunks:
        covered.update(chunk["turns"])
    missing = sorted(set(range(num_turns)) - covered)

    if missing:
        gap_runs = _split_into_runs(missing)
        for run in gap_runs:
            split_chunks.append(_make_gap_chunk(run, conversation))

    # Step 5: Sort chunks by first turn index
    split_chunks.sort(key=lambda c: c["turns"][0] if c["turns"] else 0)

    return split_chunks


def _split_into_runs(sorted_indices: list[int]) -> list[list[int]]:
    """Split a sorted list of ints into contiguous runs."""
    if not sorted_indices:
        return []
    runs = [[sorted_indices[0]]]
    for i in range(1, len(sorted_indices)):
        if sorted_indices[i] == sorted_indices[i - 1] + 1:
            runs[-1].append(sorted_indices[i])
        else:
            runs.append([sorted_indices[i]])
    return runs


def _make_gap_chunk(turn_indices: list[int], conversation: dict) -> dict:
    """Create a stasis-pulse gap-fill chunk."""
    turns = conversation["turns"]
    text = " ".join(
        turns[t]["text"] for t in turn_indices if t < len(turns)
    )
    return {
        "chunk_id": str(uuid_lib.uuid4()),
        "conversation_id": conversation.get("id", ""),
        "source_file": conversation.get("source_file", ""),
        "text": text,
        "turns": turn_indices,
        "tags": {
            "geometry": "linear",
            "coherence": "fragmented",
            "texture": "sparse",
            "terrain": "relational",
            "motifs": ["stasis-pulse"],
        },
        "chunk_type": "stasis-pulse",
        "split_rationale": "auto-fill: turns not assigned by LLM",
        "schema_version": 0,
        "llm_failure": False,
        "prompt_preset": "",
        "schema_proposals": [],
    }
