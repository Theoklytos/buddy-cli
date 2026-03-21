"""Validation, compaction, and chunk-feedback analysis for discovery maps."""

import statistics
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


def validate_discovery_map(map_data: dict) -> dict:
    """Check structural integrity of a discovery map."""
    list_keys = ("boundary_signals", "coherence_anchors", "chunk_archetypes", "anti_patterns")
    sizes = {k: len(map_data.get(k, [])) for k in list_keys}
    total = sum(sizes.values())

    # Near-duplicate ratio across all lists
    all_dupes = 0
    for key in list_keys:
        items = map_data.get(key, [])
        dupes = find_near_duplicates(items, threshold=0.7)
        all_dupes += len(dupes)
    redundancy = all_dupes / max(1, total)

    # Balance: coefficient of variation mapped to 0-1
    size_values = list(sizes.values())
    if len(size_values) > 1 and sum(size_values) > 0:
        mean = statistics.mean(size_values)
        stdev = statistics.stdev(size_values)
        cv = stdev / mean if mean > 0 else 0
        balance = round(max(0.0, 1.0 - cv), 4)
    else:
        balance = 1.0

    observations = map_data.get("observations", [])
    unique_names = len(set(
        (o.get("pattern_type", ""), o.get("name", "")) for o in observations
    ))

    return {
        "total_items": total,
        "list_sizes": sizes,
        "redundancy_ratio": round(redundancy, 4),
        "balance_score": balance,
        "observation_count": len(observations),
        "observation_unique_names": unique_names,
        "has_chunk_evidence": "chunk_evidence_score" in map_data,
    }


def compute_discovery_score(
    validation: dict, chunk_evidence_score: float | None = None
) -> float:
    """Convert validation report into a 0-1 discovery quality score."""
    base = (1 - validation["redundancy_ratio"]) * 0.3 + validation["balance_score"] * 0.2
    if chunk_evidence_score is not None:
        return round(max(0.0, base * 0.5 + chunk_evidence_score * 0.5), 4)
    return round(max(0.0, base), 4)


def analyze_chunk_feedback(chunks: list[dict], map_data: dict) -> dict:
    """Analyze chunk output to measure discovery map effectiveness.

    Args:
        chunks: List of chunk dicts (from chunks.jsonl).
        map_data: Discovery map data dict.

    Returns:
        Dict with archetype_usage, signal_effectiveness, gap_analysis,
        motif_distribution, evidence_score.
    """
    if not chunks:
        return {
            "archetype_usage": {"used": {}, "unused": [], "unknown": {}},
            "signal_effectiveness": {"referenced": {}, "dead": []},
            "gap_analysis": {"total_gap_chunks": 0, "gap_ratio": 0.0},
            "motif_distribution": {},
            "evidence_score": 0.0,
        }

    # Archetype usage
    type_counts: dict[str, int] = {}
    for chunk in chunks:
        ct = chunk.get("chunk_type", "exchange")
        type_counts[ct] = type_counts.get(ct, 0) + 1

    map_archetypes = set(map_data.get("chunk_archetypes", []))
    used = {k: v for k, v in type_counts.items() if k in map_archetypes}
    unused = sorted(map_archetypes - set(type_counts.keys()))
    unknown = {k: v for k, v in type_counts.items()
               if k not in map_archetypes and k != "stasis-pulse"}

    # Signal effectiveness
    signals = map_data.get("boundary_signals", [])
    rationales = " ".join(
        chunk.get("split_rationale", "").lower() for chunk in chunks
    )
    referenced: dict[str, int] = {}
    for signal in signals:
        if signal.lower() in rationales:
            count = rationales.count(signal.lower())
            referenced[signal] = count
    dead = [s for s in signals if s not in referenced]

    # Gap analysis
    gap_chunks = [c for c in chunks if c.get("chunk_type") == "stasis-pulse"]
    gap_ratio = len(gap_chunks) / max(1, len(chunks))

    # Motif distribution
    motif_counts: dict[str, int] = {}
    for chunk in chunks:
        motifs = chunk.get("tags", {}).get("motifs", [])
        if isinstance(motifs, list):
            for m in motifs:
                motif_counts[m] = motif_counts.get(m, 0) + 1

    # Evidence score
    archetype_ratio = len(used) / max(1, len(used) + len(unused))
    signal_ratio = len(referenced) / max(1, len(signals))
    gap_penalty = 1 - gap_ratio
    evidence = round(
        0.4 * archetype_ratio + 0.4 * signal_ratio + 0.2 * gap_penalty, 4
    )

    return {
        "archetype_usage": {"used": used, "unused": unused, "unknown": unknown},
        "signal_effectiveness": {"referenced": referenced, "dead": dead},
        "gap_analysis": {"total_gap_chunks": len(gap_chunks), "gap_ratio": round(gap_ratio, 4)},
        "motif_distribution": motif_counts,
        "evidence_score": evidence,
    }


def compact_map(map_data: dict, similarity_threshold: float = 0.7) -> dict:
    """Merge near-duplicates and deduplicate observations in place.

    Args:
        map_data: Discovery map data dict (mutated in place).
        similarity_threshold: Minimum similarity to merge.

    Returns:
        Report dict with duplicates_merged, observations_deduped,
        items_before, items_after.
    """
    list_keys = ("boundary_signals", "coherence_anchors", "chunk_archetypes", "anti_patterns")
    items_before = sum(len(map_data.get(k, [])) for k in list_keys)
    total_merged = 0

    for key in list_keys:
        items = map_data.get(key, [])
        if len(items) < 2:
            continue
        dupes = find_near_duplicates(items, threshold=similarity_threshold)
        if not dupes:
            continue
        # Collect indices to remove (keep shorter/more general string)
        to_remove: set[int] = set()
        for idx_a, idx_b, _sim in dupes:
            if idx_a in to_remove or idx_b in to_remove:
                continue
            # Keep the shorter one (more general)
            if len(items[idx_a]) <= len(items[idx_b]):
                to_remove.add(idx_b)
            else:
                to_remove.add(idx_a)
        map_data[key] = [item for i, item in enumerate(items) if i not in to_remove]
        total_merged += len(to_remove)

    # Dedup observations
    observations = map_data.get("observations", [])
    deduped = dedup_observations(observations)
    obs_removed = len(observations) - len(deduped)
    map_data["observations"] = deduped

    items_after = sum(len(map_data.get(k, [])) for k in list_keys)

    return {
        "duplicates_merged": total_merged,
        "observations_deduped": obs_removed,
        "items_before": items_before,
        "items_after": items_after,
    }
