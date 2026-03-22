"""Iterative chunk refinement for Bud RAG Pipeline.

Runs multiple passes of chunking, reviewing the output after each pass and
feeding refinement insights back into the next pass.  Produces a stability
score that indicates how settled the chunking has become.
"""

import json
import random
from typing import Callable, Optional

from bud.lib.errors import LLMError

REVIEW_SYSTEM_PROMPT = """You are a chunking quality analyst reviewing how an AI conversation archive was split into chunks.

Your job is to evaluate the chunking decisions and provide feedback that will improve the next pass.

Analyze the sample chunks for:
- BOUNDARY QUALITY: Are chunks cut at natural breakpoints, or do they split mid-thought?
- COHERENCE: Does each chunk hold together as a self-contained unit?
- TAG ACCURACY: Do the assigned tags (geometry, coherence, texture, terrain, motifs) match the content?
- SIZE BALANCE: Are chunks roughly similar in information density, or are some bloated/starved?
- MISSED PATTERNS: Are there structural patterns in the data that the chunking is ignoring?

Return ONLY valid JSON:
{
  "feedback": {
    "boundary_issues": ["specific problems with where chunks are cut"],
    "coherence_issues": ["chunks that don't hold together well"],
    "tag_corrections": ["tags that seem wrong and why"],
    "missed_patterns": ["patterns the chunker should look for"],
    "good_decisions": ["things the chunker is doing well — keep these"]
  },
  "refinement_guidance": "A concise paragraph of guidance for the next chunking pass.",
  "stability_score": 0.6
}

stability_score: 0.0 = chunking needs major rework, 1.0 = chunking is excellent and stable.

Structural integrity (turn coverage, gaps, overlaps) is validated automatically and repaired before you see the chunks. Focus your evaluation on boundary quality, coherence, tag accuracy, and pattern matching."""

REVIEW_USER_TEMPLATE = """Chunking pass {pass_number} results.

Discovery map context:
{discovery_summary}

Previous refinement feedback (accumulated):
{prior_feedback}

Structural validation (automated — do NOT re-evaluate these):
{structural_summary}

Sample chunks to review ({n_samples} of {total_chunks} total):
{chunk_samples}

Evaluate these chunks and provide feedback for the next pass. Return only the JSON object."""


class ChunkRefinementState:
    """Tracks accumulated refinement feedback across chunking passes."""

    def __init__(self):
        self._passes: list[dict] = []
        self._stability_scores: list[float] = []
        self._structural_scores: list[float] = []
        self._feedback_accumulator: dict = {
            "boundary_issues": [],
            "coherence_issues": [],
            "tag_corrections": [],
            "missed_patterns": [],
            "good_decisions": [],
        }
        self._guidance: str = ""

    @property
    def pass_count(self) -> int:
        return len(self._passes)

    @property
    def stability_score(self) -> float:
        if not self._stability_scores:
            return 0.0
        # EMA with alpha=0.4 for quality scores
        quality = self._stability_scores[0]
        for s in self._stability_scores[1:]:
            quality = 0.4 * s + 0.6 * quality
        # Composite: blend structural and quality if structural data exists
        if self._structural_scores:
            structural = self._structural_scores[-1]
            return round(0.5 * structural + 0.5 * quality, 4)
        return round(quality, 4)

    @property
    def guidance(self) -> str:
        return self._guidance

    def to_feedback_summary(self) -> str:
        """Compact summary for injection into the chunking system prompt."""
        if not self._passes:
            return "(no prior feedback — this is the first pass)"
        parts = []
        for key, items in self._feedback_accumulator.items():
            if items:
                # Keep the most recent observations (deduplicated)
                unique = list(dict.fromkeys(items))[-8:]
                parts.append(f"  {key}: {json.dumps(unique)}")
        if self._guidance:
            parts.append(f"\n  Guidance: {self._guidance}")
        return "\n".join(parts) if parts else "(no actionable feedback yet)"

    def apply_structural_score(self, score: float) -> None:
        """Record a structural validation score for the current pass."""
        self._structural_scores.append(score)

    def apply_review(self, review: dict) -> None:
        """Merge a review response into accumulated state."""
        self._passes.append(review)

        feedback = review.get("feedback", {})
        for key in self._feedback_accumulator:
            new_items = feedback.get(key, [])
            if isinstance(new_items, list):
                self._feedback_accumulator[key].extend(new_items)

        self._guidance = review.get("refinement_guidance", self._guidance)
        self._stability_scores.append(
            float(review.get("stability_score", 0.0))
        )

    def to_report(self) -> dict:
        """Final summary for persistence or display."""
        return {
            "total_passes": self.pass_count,
            "stability_score": self.stability_score,
            "stability_history": self._stability_scores,
            "structural_score": self._structural_scores[-1] if self._structural_scores else None,
            "structural_history": self._structural_scores,
            "final_guidance": self._guidance,
            "accumulated_feedback": {
                k: list(dict.fromkeys(v))[-8:]
                for k, v in self._feedback_accumulator.items()
            },
        }


def _sample_chunks(chunks: list[dict], n: int = 8) -> list[dict]:
    """Pick a diverse sample of chunks for review."""
    if len(chunks) <= n:
        return chunks
    return random.sample(chunks, n)


def _format_chunks_for_review(chunks: list[dict]) -> str:
    """Format chunks for the review LLM prompt."""
    from bud.stages.chunk import _truncate_at_boundary
    parts = []
    for i, c in enumerate(chunks, 1):
        text_preview = _truncate_at_boundary(c.get("text", ""), 400)
        tags = c.get("tags", {})
        parts.append(
            f"--- chunk {i} ---\n"
            f"type: {c.get('chunk_type', '?')}  |  "
            f"turns: {c.get('turns', [])}  |  "
            f"rationale: {c.get('split_rationale', '?')}\n"
            f"tags: {json.dumps(tags)}\n"
            f"text: {text_preview}\n"
        )
    return "\n".join(parts)


def run_iterative_chunking(
    conversations: list[dict],
    schema: dict,
    llm,
    config: dict,
    system_prompt: str,
    prompt_preset: str = "conversational",
    schema_version: int = 1,
    concept_map_summary: str | None = None,
    max_iterations: int = 3,
    stability_threshold: float = 0.75,
    on_pass_start: Optional[Callable] = None,
    on_pass_complete: Optional[Callable] = None,
    on_review_complete: Optional[Callable] = None,
) -> tuple[list[dict], ChunkRefinementState]:
    """Run iterative chunking with refinement feedback.

    Each iteration:
      1. Chunk all conversations (using accumulated feedback)
      2. Sample the output and ask the LLM to review
      3. Accumulate feedback and check stability
      4. If not stable, re-chunk with the feedback injected

    Args:
        conversations: List of parsed conversation dicts.
        schema: Current schema dict.
        llm: LLMClient instance.
        config: Pipeline config.
        system_prompt: Base system prompt for chunking.
        prompt_preset: Prompt preset name.
        schema_version: Current schema version.
        concept_map_summary: Optional discovery map summary JSON.
        max_iterations: Maximum chunking passes.
        stability_threshold: Stop when stability >= this.
        on_pass_start: callback(pass_num, max_iterations)
        on_pass_complete: callback(pass_num, n_chunks, n_conversations)
        on_review_complete: callback(pass_num, stability_score, state)

    Returns:
        Tuple of (final_chunks, refinement_state).
    """
    from bud.stages.chunk import chunk_conversation

    state = ChunkRefinementState()
    best_chunks: list[dict] = []
    discovery_summary = concept_map_summary or "(no discovery map)"

    for pass_num in range(1, max_iterations + 1):
        if on_pass_start:
            on_pass_start(pass_num, max_iterations)

        # Build the augmented system prompt with refinement feedback
        augmented_prompt = system_prompt
        if concept_map_summary:
            augmented_prompt += (
                "\n\n## Archive Pattern Map\n"
                "Use these discovered patterns to guide your chunking decisions:\n"
                + concept_map_summary
            )
        if state.pass_count > 0:
            augmented_prompt += (
                "\n\n## Chunking Refinement Feedback (from prior passes)\n"
                "Apply this feedback to improve your chunking:\n"
                + state.to_feedback_summary()
            )

        # Chunk all conversations — concurrent when llm.concurrency > 1
        all_chunks: list[dict] = []
        _conc = getattr(llm, "concurrency", 1)
        use_batch = isinstance(_conc, int) and _conc > 1

        if use_batch:
            from bud.stages.chunk import chunk_conversations_batch
            batch_results = chunk_conversations_batch(
                conversations, schema, llm, config, augmented_prompt,
                prompt_preset=prompt_preset,
                schema_version=schema_version,
                concept_map_summary=None,
            )
            for chunks in batch_results:
                all_chunks.extend(chunks)
        else:
            for conv in conversations:
                try:
                    chunks = chunk_conversation(
                        conv, schema, llm, config, augmented_prompt,
                        prompt_preset=prompt_preset,
                        schema_version=schema_version,
                        concept_map_summary=None,
                    )
                    all_chunks.extend(chunks)
                except Exception:
                    pass

        if on_pass_complete:
            on_pass_complete(pass_num, len(all_chunks), len(conversations))

        best_chunks = all_chunks

        # Compute per-conversation structural scores
        from bud.stages.chunk_validate import validate_chunks, compute_structural_score
        chunks_by_conv: dict[str, list[dict]] = {}
        for chunk in all_chunks:
            conv_id = chunk["conversation_id"]
            chunks_by_conv.setdefault(conv_id, []).append(chunk)

        structural_scores = []
        total_missing = 0
        total_overlapping = 0
        n_with_gaps = 0
        n_with_overlaps = 0
        for conv in conversations:
            conv_chunks = chunks_by_conv.get(conv["id"], [])
            validation = validate_chunks(conv_chunks, len(conv["turns"]))
            structural_scores.append(compute_structural_score(validation))
            if validation["missing_turns"]:
                n_with_gaps += 1
                total_missing += len(validation["missing_turns"])
            if validation["overlapping_turns"]:
                n_with_overlaps += 1
                total_overlapping += len(validation["overlapping_turns"])

        avg_structural = sum(structural_scores) / max(1, len(structural_scores))
        state.apply_structural_score(avg_structural)

        # Last iteration: skip review — just use the chunks
        if pass_num == max_iterations:
            break

        # Review the chunks
        sample_n = min(30, max(10, len(all_chunks) // 20))
        sample = _sample_chunks(all_chunks, n=min(sample_n, len(all_chunks)))
        structural_summary = (
            f"Structural score: {avg_structural:.2f}/1.00\n"
            f"Conversations with gaps: {n_with_gaps}/{len(conversations)}\n"
            f"Conversations with overlaps: {n_with_overlaps}/{len(conversations)}\n"
            f"Total missing turns: {total_missing}\n"
            f"Total overlapping turns: {total_overlapping}"
        )
        review_prompt = REVIEW_USER_TEMPLATE.format(
            pass_number=pass_num,
            discovery_summary=discovery_summary,
            prior_feedback=state.to_feedback_summary(),
            structural_summary=structural_summary,
            chunk_samples=_format_chunks_for_review(sample),
            n_samples=len(sample),
            total_chunks=len(all_chunks),
        )

        try:
            response_text = llm.complete(
                system=REVIEW_SYSTEM_PROMPT,
                user=review_prompt,
            )
            text = response_text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            review = json.loads(text)
            state.apply_review(review)
        except (LLMError, json.JSONDecodeError, OSError):
            # If review fails, keep the chunks and move on
            break

        if on_review_complete:
            on_review_complete(pass_num, state.stability_score, state)

        if state.stability_score >= stability_threshold:
            break

    return best_chunks, state
