# tests/test_discover_validate.py
from bud.stages.discover_validate import find_near_duplicates, dedup_observations
from bud.stages.discover_validate import validate_discovery_map, compute_discovery_score
from bud.stages.discover_validate import compact_map
from bud.stages.discover_validate import analyze_chunk_feedback

SAMPLE_MAP_DATA = {
    "boundary_signals": ["topic change", "long silence", "hard no"],
    "coherence_anchors": ["shared metaphor"],
    "chunk_archetypes": ["exchange", "deep-dive", "meta-reflection"],
    "anti_patterns": ["splitting mid-thought"],
    "observations": [],
}

SAMPLE_CHUNKS = [
    {"chunk_type": "exchange", "tags": {"motifs": ["identity"]},
     "split_rationale": "topic change detected"},
    {"chunk_type": "exchange", "tags": {"motifs": ["identity", "becoming"]},
     "split_rationale": "natural pause"},
    {"chunk_type": "deep-dive", "tags": {"motifs": ["resonance"]},
     "split_rationale": "long silence marks shift"},
    {"chunk_type": "stasis-pulse", "tags": {"motifs": ["stasis-pulse"]},
     "split_rationale": "auto-fill: turns not assigned by LLM"},
]


def test_archetype_usage():
    feedback = analyze_chunk_feedback(SAMPLE_CHUNKS, SAMPLE_MAP_DATA)
    assert "exchange" in feedback["archetype_usage"]["used"]
    assert "meta-reflection" in feedback["archetype_usage"]["unused"]


def test_signal_effectiveness():
    feedback = analyze_chunk_feedback(SAMPLE_CHUNKS, SAMPLE_MAP_DATA)
    assert "topic change" in feedback["signal_effectiveness"]["referenced"]
    assert "hard no" in feedback["signal_effectiveness"]["dead"]


def test_gap_analysis():
    feedback = analyze_chunk_feedback(SAMPLE_CHUNKS, SAMPLE_MAP_DATA)
    assert feedback["gap_analysis"]["total_gap_chunks"] == 1


def test_evidence_score_range():
    feedback = analyze_chunk_feedback(SAMPLE_CHUNKS, SAMPLE_MAP_DATA)
    assert 0.0 <= feedback["evidence_score"] <= 1.0


def test_empty_chunks():
    feedback = analyze_chunk_feedback([], SAMPLE_MAP_DATA)
    assert feedback["evidence_score"] == 0.0


_SIGNAL_POOL = [
    "fractal narrowing at boundary",
    "corrective pivot under pressure",
    "medium rupture with rebound",
    "slow drift toward equilibrium",
    "sharp inversion cascade",
    "gradual compression event",
    "recursive echo pattern",
    "lateral tension release",
    "oscillating threshold breach",
    "abrupt coherence collapse",
    "micro-fracture propagation",
    "delayed feedback loop",
    "phase transition onset",
    "emergent cluster formation",
    "entropy spike followed by decay",
    "sustained resonance burst",
    "asymmetric load distribution",
    "nonlinear growth inflection",
    "boundary dissolution event",
    "synchronized pulse dropout",
    "threshold saturation plateau",
    "counter-cyclical rebound",
    "nested hierarchy collapse",
    "diffuse signal attenuated",
    "polarization front advance",
    "stochastic resonance peak",
    "modal shift under load",
    "cascading failure initiation",
    "attractor basin escape",
    "bifurcation point crossing",
    "coherence island formation",
    "turbulent wake stabilization",
    "critical slowing down detected",
    "spontaneous symmetry breaking",
    "long-range correlation decay",
    "vortex shedding onset",
    "regime change inflection",
    "tipping point proximity signal",
    "dissipative structure emergence",
    "self-organized criticality marker",
    "hysteresis loop closure",
    "frequency locking event",
    "amplitude death precursor",
    "chimera state formation",
    "transient chaos detection",
    "percolation threshold crossing",
    "soliton wave propagation",
    "topological defect nucleation",
    "jamming transition onset",
    "quasiperiodic orbit detected",
]


def _make_map_data(signals=10, anchors=10, archetypes=10, anti=10, obs=5):
    """Helper to build a discovery map data dict using distinct strings."""
    pool = _SIGNAL_POOL
    # Each list draws from a different offset within the pool to stay distinct
    sig = pool[0:signals]
    anc = pool[signals:signals + anchors]
    arc = pool[signals + anchors:signals + anchors + archetypes]
    ant = pool[signals + anchors + archetypes:signals + anchors + archetypes + anti]
    return {
        "version": 1,
        "iterations_completed": 5,
        "stability_score": 0.7,
        "boundary_signals": sig,
        "coherence_anchors": anc,
        "chunk_archetypes": arc,
        "anti_patterns": ant,
        "observations": [
            {"pattern_type": "geometric", "name": f"observation-item-number-{i:03d}",
             "description": "test", "confidence": 0.8}
            for i in range(obs)
        ],
    }


def test_validate_balanced_map():
    data = _make_map_data()
    validation = validate_discovery_map(data)
    assert validation["total_items"] == 40
    assert validation["balance_score"] > 0.9
    assert validation["redundancy_ratio"] == 0.0


def test_validate_map_with_duplicates():
    data = _make_map_data(signals=3)
    data["boundary_signals"] = [
        "fractal narrowing at the system boundary",
        "fractal narrowing at the system boundary edge",
        "something completely and utterly unrelated",
    ]
    validation = validate_discovery_map(data)
    assert validation["redundancy_ratio"] > 0


def test_validate_imbalanced_map():
    data = _make_map_data(signals=50, anchors=5, archetypes=5, anti=5)
    validation = validate_discovery_map(data)
    assert validation["balance_score"] < 0.7


def test_discovery_score_perfect():
    validation = {
        "redundancy_ratio": 0.0,
        "balance_score": 1.0,
    }
    assert compute_discovery_score(validation) > 0.4


def test_discovery_score_with_evidence():
    validation = {
        "redundancy_ratio": 0.0,
        "balance_score": 1.0,
    }
    score_without = compute_discovery_score(validation)
    score_with = compute_discovery_score(validation, chunk_evidence_score=0.9)
    assert score_with > score_without


def test_discovery_score_floors_at_zero():
    validation = {
        "redundancy_ratio": 1.0,
        "balance_score": 0.0,
    }
    assert compute_discovery_score(validation) == 0.0


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


def test_compact_merges_near_duplicates():
    map_data = _make_map_data()
    map_data["boundary_signals"] = [
        "fractal narrowing at the system boundary",
        "fractal narrowing at the system boundary edge",
        "something completely and utterly unrelated",
    ]
    report = compact_map(map_data)
    assert report["duplicates_merged"] > 0
    assert len(map_data["boundary_signals"]) == 2  # merged to 2


def test_compact_deduplicates_observations():
    map_data = _make_map_data()
    map_data["observations"] = [
        {"pattern_type": "boundary", "name": "rupture", "description": "v1", "confidence": 0.7},
        {"pattern_type": "boundary", "name": "rupture", "description": "v2", "confidence": 0.9},
    ]
    report = compact_map(map_data)
    assert report["observations_deduped"] == 1
    assert len(map_data["observations"]) == 1
    assert map_data["observations"][0]["confidence"] == 0.9


def test_compact_noop_on_clean_map():
    map_data = _make_map_data()
    report = compact_map(map_data)
    assert report["duplicates_merged"] == 0
    assert report["items_before"] == report["items_after"]
