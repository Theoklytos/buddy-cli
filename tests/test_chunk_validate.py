from bud.stages.chunk_validate import validate_chunks, compute_structural_score


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
