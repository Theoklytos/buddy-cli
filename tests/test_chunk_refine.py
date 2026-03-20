"""Tests for bud.stages.chunk_refine."""

import json
from unittest.mock import MagicMock, patch

import pytest

from bud.lib.errors import LLMError
from bud.stages.chunk_refine import (
    ChunkRefinementState,
    run_iterative_chunking,
    _sample_chunks,
    _format_chunks_for_review,
)


SAMPLE_SCHEMA = {
    "version": 1,
    "dimensions": {
        "geometry": ["linear", "recursive"],
        "coherence": ["tight", "loose"],
        "texture": ["dense", "raw"],
        "terrain": ["conceptual", "procedural"],
        "motifs": ["identity", "threshold"],
    },
    "chunk_types": ["exchange", "monologue"],
    "multi_value_dimensions": ["motifs"],
    "candidates": {},
    "evolution_log": [],
}

SAMPLE_CONVERSATION = {
    "id": "conv-1",
    "source_file": "conversations_001.json",
    "conversation_name": "Test Chat",
    "turns": [
        {"sender": "human", "text": "Hello"},
        {"sender": "assistant", "text": "Hi there, how can I help?"},
    ],
}

PIPELINE_CONFIG = {
    "pipeline": {
        "chunk_min_tokens": 1,
        "chunk_max_tokens": 2000,
    }
}

VALID_CHUNK_RESPONSE = json.dumps({
    "chunks": [
        {
            "turns": [0, 1],
            "tags": {
                "geometry": "linear",
                "coherence": "tight",
                "texture": "raw",
                "terrain": "conceptual",
                "motifs": ["identity"],
            },
            "chunk_type": "exchange",
            "split_rationale": "single exchange",
        }
    ],
    "schema_proposals": [],
})

VALID_REVIEW_RESPONSE = json.dumps({
    "feedback": {
        "boundary_issues": ["some chunks split mid-thought"],
        "coherence_issues": [],
        "tag_corrections": ["geometry should be recursive not linear"],
        "missed_patterns": ["call-and-response rhythm"],
        "good_decisions": ["chunk sizes are consistent"],
    },
    "refinement_guidance": "Focus on call-and-response rhythm.",
    "stability_score": 0.7,
})


class TestChunkRefinementState:

    def test_initial_state(self):
        state = ChunkRefinementState()
        assert state.pass_count == 0
        assert state.stability_score == 0.0
        assert state.guidance == ""

    def test_apply_review(self):
        state = ChunkRefinementState()
        review = json.loads(VALID_REVIEW_RESPONSE)
        state.apply_review(review)
        assert state.pass_count == 1
        assert state.stability_score > 0.0
        assert "call-and-response" in state.guidance

    def test_stability_ema(self):
        state = ChunkRefinementState()
        state.apply_review({"stability_score": 0.5})
        state.apply_review({"stability_score": 0.9})
        # EMA: 0.4 * 0.9 + 0.6 * 0.5 = 0.66
        assert abs(state.stability_score - 0.66) < 0.01

    def test_feedback_summary_first_pass(self):
        state = ChunkRefinementState()
        summary = state.to_feedback_summary()
        assert "first pass" in summary

    def test_feedback_summary_after_review(self):
        state = ChunkRefinementState()
        review = json.loads(VALID_REVIEW_RESPONSE)
        state.apply_review(review)
        summary = state.to_feedback_summary()
        assert "mid-thought" in summary
        assert "call-and-response" in summary

    def test_to_report(self):
        state = ChunkRefinementState()
        review = json.loads(VALID_REVIEW_RESPONSE)
        state.apply_review(review)
        report = state.to_report()
        assert report["total_passes"] == 1
        assert report["stability_score"] > 0.0
        assert "stability_history" in report
        assert "accumulated_feedback" in report


class TestSampleChunks:

    def test_returns_all_when_under_limit(self):
        chunks = [{"chunk_id": str(i)} for i in range(3)]
        result = _sample_chunks(chunks, n=10)
        assert len(result) == 3

    def test_samples_down_to_n(self):
        chunks = [{"chunk_id": str(i)} for i in range(20)]
        result = _sample_chunks(chunks, n=5)
        assert len(result) == 5


class TestFormatChunksForReview:

    def test_formats_basic_chunk(self):
        chunks = [{
            "text": "Hello world",
            "tags": {"geometry": "linear"},
            "chunk_type": "exchange",
            "turns": [0, 1],
            "split_rationale": "test",
        }]
        result = _format_chunks_for_review(chunks)
        assert "chunk 1" in result
        assert "Hello world" in result
        assert "exchange" in result


class TestRunIterativeChunking:

    def test_single_iteration(self):
        """Single iteration should chunk without review."""
        llm = MagicMock()
        llm.complete.return_value = VALID_CHUNK_RESPONSE

        chunks, state = run_iterative_chunking(
            conversations=[SAMPLE_CONVERSATION],
            schema=SAMPLE_SCHEMA,
            llm=llm,
            config=PIPELINE_CONFIG,
            system_prompt="test",
            max_iterations=1,
        )
        assert len(chunks) >= 1
        assert state.pass_count == 0  # no reviews on single pass

    def test_multi_iteration_calls_review(self):
        """Multiple iterations should include review between passes."""
        llm = MagicMock()
        # Alternates: chunk response, review response, chunk response
        llm.complete.side_effect = [
            VALID_CHUNK_RESPONSE,       # pass 1 chunk
            VALID_REVIEW_RESPONSE,       # pass 1 review
            VALID_CHUNK_RESPONSE,        # pass 2 chunk
        ]

        chunks, state = run_iterative_chunking(
            conversations=[SAMPLE_CONVERSATION],
            schema=SAMPLE_SCHEMA,
            llm=llm,
            config=PIPELINE_CONFIG,
            system_prompt="test",
            max_iterations=2,
        )
        assert len(chunks) >= 1
        assert state.pass_count == 1  # one review happened

    def test_stops_at_stability(self):
        """Should stop early when stability is reached."""
        high_stability_review = json.dumps({
            "feedback": {},
            "refinement_guidance": "Looks great.",
            "stability_score": 0.95,
        })
        llm = MagicMock()
        llm.complete.side_effect = [
            VALID_CHUNK_RESPONSE,
            high_stability_review,
            # Should NOT be called — stability threshold reached
        ]

        chunks, state = run_iterative_chunking(
            conversations=[SAMPLE_CONVERSATION],
            schema=SAMPLE_SCHEMA,
            llm=llm,
            config=PIPELINE_CONFIG,
            system_prompt="test",
            max_iterations=5,
            stability_threshold=0.75,
        )
        assert state.stability_score >= 0.75
        # Only 2 LLM calls: chunk + review (stopped before pass 2)
        assert llm.complete.call_count == 2

    def test_handles_review_failure(self):
        """If review LLM call fails, should return chunks from that pass."""
        llm = MagicMock()
        llm.complete.side_effect = [
            VALID_CHUNK_RESPONSE,
            LLMError("review failed"),
        ]

        chunks, state = run_iterative_chunking(
            conversations=[SAMPLE_CONVERSATION],
            schema=SAMPLE_SCHEMA,
            llm=llm,
            config=PIPELINE_CONFIG,
            system_prompt="test",
            max_iterations=3,
        )
        assert len(chunks) >= 1
        assert state.pass_count == 0  # review never completed

    def test_callbacks_called(self):
        llm = MagicMock()
        llm.complete.return_value = VALID_CHUNK_RESPONSE

        pass_starts = []
        pass_completes = []

        chunks, state = run_iterative_chunking(
            conversations=[SAMPLE_CONVERSATION],
            schema=SAMPLE_SCHEMA,
            llm=llm,
            config=PIPELINE_CONFIG,
            system_prompt="test",
            max_iterations=1,
            on_pass_start=lambda p, m: pass_starts.append(p),
            on_pass_complete=lambda p, nc, nv: pass_completes.append((p, nc)),
        )
        assert pass_starts == [1]
        assert len(pass_completes) == 1

    def test_concept_map_injected(self):
        """When concept_map_summary is provided, it should appear in the LLM call."""
        llm = MagicMock()
        llm.complete.return_value = VALID_CHUNK_RESPONSE

        chunks, state = run_iterative_chunking(
            conversations=[SAMPLE_CONVERSATION],
            schema=SAMPLE_SCHEMA,
            llm=llm,
            config=PIPELINE_CONFIG,
            system_prompt="test",
            concept_map_summary='{"boundary_signals": ["topic shift"]}',
            max_iterations=1,
        )
        call_args = llm.complete.call_args
        system_sent = call_args[1].get("system", call_args[0][0] if call_args[0] else "")
        assert "Archive Pattern Map" in system_sent
        assert "topic shift" in system_sent
