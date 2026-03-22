from bud.stages.chunk_validate import validate_chunks, compute_structural_score, repair_chunks


def test_perfect_coverage():
    chunks = [
        {"turns": [0, 1, 2]},
        {"turns": [3, 4]},
    ]
    result = validate_chunks(chunks, num_turns=5)
    assert result["is_valid"] is True
    assert result["coverage_ratio"] == 1.0
    assert result["missing_turns"] == []
    assert result["overlapping_turns"] == {}
    assert result["out_of_bounds"] == []


def test_detects_overlap():
    chunks = [
        {"turns": [0, 1, 2]},
        {"turns": [2, 3, 4]},
    ]
    result = validate_chunks(chunks, num_turns=5)
    assert result["is_valid"] is False
    assert 2 in result["overlapping_turns"]


def test_detects_gap():
    chunks = [
        {"turns": [0, 1]},
        {"turns": [4, 5]},
    ]
    result = validate_chunks(chunks, num_turns=6)
    assert result["is_valid"] is False
    assert result["missing_turns"] == [2, 3]


def test_detects_out_of_bounds():
    chunks = [{"turns": [0, 1, 10]}]
    result = validate_chunks(chunks, num_turns=5)
    assert result["is_valid"] is False
    assert 10 in result["out_of_bounds"]


def test_detects_non_contiguous():
    chunks = [{"turns": [0, 2, 4]}]
    result = validate_chunks(chunks, num_turns=5)
    assert result["is_valid"] is False
    assert 0 in result["non_contiguous_chunks"]


def test_empty_chunks():
    result = validate_chunks([], num_turns=5)
    assert result["is_valid"] is False
    assert result["coverage_ratio"] == 0.0


def test_perfect_score():
    validation = {
        "coverage_ratio": 1.0,
        "overlapping_turns": {},
        "missing_turns": [],
        "out_of_bounds": [],
    }
    assert compute_structural_score(validation) == 1.0


def test_gaps_reduce_score():
    validation = {
        "coverage_ratio": 0.8,
        "overlapping_turns": {},
        "missing_turns": [3, 4, 5, 6, 7],
        "out_of_bounds": [],
    }
    score = compute_structural_score(validation)
    assert 0.5 < score < 0.8


def test_overlaps_reduce_score():
    validation = {
        "coverage_ratio": 1.0,
        "overlapping_turns": {2: 2, 5: 3},
        "missing_turns": [],
        "out_of_bounds": [],
    }
    score = compute_structural_score(validation)
    assert score < 1.0


def test_score_floors_at_zero():
    validation = {
        "coverage_ratio": 0.1,
        "overlapping_turns": {i: 2 for i in range(30)},
        "missing_turns": list(range(80)),
        "out_of_bounds": list(range(10)),
    }
    assert compute_structural_score(validation) == 0.0


SAMPLE_TURNS = [{"text": f"Turn {i}", "sender": "user"} for i in range(10)]
SAMPLE_CONV = {"id": "c1", "source_file": "test.json", "turns": SAMPLE_TURNS}


def test_repair_fixes_overlaps():
    chunks = [
        {"turns": [0, 1, 2], "tags": {}, "chunk_type": "exchange",
         "split_rationale": "a", "conversation_id": "c1", "source_file": "test.json",
         "chunk_id": "x1", "text": "", "schema_version": 1,
         "llm_failure": False, "prompt_preset": "conversational", "schema_proposals": []},
        {"turns": [2, 3, 4], "tags": {}, "chunk_type": "exchange",
         "split_rationale": "b", "conversation_id": "c1", "source_file": "test.json",
         "chunk_id": "x2", "text": "", "schema_version": 1,
         "llm_failure": False, "prompt_preset": "conversational", "schema_proposals": []},
    ]
    repaired = repair_chunks(chunks, 5, SAMPLE_CONV)
    result = validate_chunks(repaired, 5)
    assert result["is_valid"] is True
    assert result["overlapping_turns"] == {}


def test_repair_fills_gaps():
    chunks = [
        {"turns": [0, 1], "tags": {}, "chunk_type": "exchange",
         "split_rationale": "a", "conversation_id": "c1", "source_file": "test.json",
         "chunk_id": "x1", "text": "", "schema_version": 1,
         "llm_failure": False, "prompt_preset": "conversational", "schema_proposals": []},
        {"turns": [4, 5], "tags": {}, "chunk_type": "exchange",
         "split_rationale": "b", "conversation_id": "c1", "source_file": "test.json",
         "chunk_id": "x2", "text": "", "schema_version": 1,
         "llm_failure": False, "prompt_preset": "conversational", "schema_proposals": []},
    ]
    repaired = repair_chunks(chunks, 6, SAMPLE_CONV)
    result = validate_chunks(repaired, 6)
    assert result["is_valid"] is True
    gap_chunks = [c for c in repaired if c["chunk_type"] == "stasis-pulse"]
    assert len(gap_chunks) >= 1
    assert gap_chunks[0]["turns"] == [2, 3]


def test_repair_noop_on_valid():
    chunks = [
        {"turns": [0, 1, 2], "tags": {}, "chunk_type": "exchange",
         "split_rationale": "a", "conversation_id": "c1", "source_file": "test.json",
         "chunk_id": "x1", "text": "Turn 0 Turn 1 Turn 2", "schema_version": 1,
         "llm_failure": False, "prompt_preset": "conversational", "schema_proposals": []},
        {"turns": [3, 4], "tags": {}, "chunk_type": "exchange",
         "split_rationale": "b", "conversation_id": "c1", "source_file": "test.json",
         "chunk_id": "x2", "text": "Turn 3 Turn 4", "schema_version": 1,
         "llm_failure": False, "prompt_preset": "conversational", "schema_proposals": []},
    ]
    repaired = repair_chunks(chunks, 5, SAMPLE_CONV)
    assert len(repaired) == 2


def test_repair_removes_oob():
    chunks = [
        {"turns": [0, 1, 99], "tags": {}, "chunk_type": "exchange",
         "split_rationale": "a", "conversation_id": "c1", "source_file": "test.json",
         "chunk_id": "x1", "text": "", "schema_version": 1,
         "llm_failure": False, "prompt_preset": "conversational", "schema_proposals": []},
    ]
    repaired = repair_chunks(chunks, 5, SAMPLE_CONV)
    for c in repaired:
        for t in c["turns"]:
            assert t < 5


def test_repair_result_always_valid():
    chunks = [
        {"turns": [0, 3, 7], "tags": {}, "chunk_type": "exchange",
         "split_rationale": "a", "conversation_id": "c1", "source_file": "test.json",
         "chunk_id": "x1", "text": "", "schema_version": 1,
         "llm_failure": False, "prompt_preset": "conversational", "schema_proposals": []},
        {"turns": [3, 5, 9], "tags": {}, "chunk_type": "exchange",
         "split_rationale": "b", "conversation_id": "c1", "source_file": "test.json",
         "chunk_id": "x2", "text": "", "schema_version": 1,
         "llm_failure": False, "prompt_preset": "conversational", "schema_proposals": []},
    ]
    repaired = repair_chunks(chunks, 10, SAMPLE_CONV)
    result = validate_chunks(repaired, 10)
    assert result["is_valid"] is True
