"""Chunking stage for Bud RAG Pipeline."""

import json
import uuid as uuid_lib

from bud.stages.chunk_validate import validate_chunks, repair_chunks

CHUNK_USER_TEMPLATE = """Conversation: {name}
Total turns: {num_turns} (indices 0 to {max_turn_index})

CRITICAL STRUCTURAL RULES:
1. Every turn index from 0 to {max_turn_index} must appear in EXACTLY one chunk. No gaps. No overlaps.
2. Turn indices within each chunk must be CONTIGUOUS (e.g., [3,4,5] not [3,5,7]).
3. Chunks must follow the linear order of the conversation. Do not group by topic across distant turns.

Turns:
{turns_text}

Respond with JSON only:
{{
  "chunks": [
    {{
      "turns": [0, 1, 2],
      "tags": {{
        "geometry": "one of {geometry}",
        "coherence": "one of {coherence}",
        "texture": "one of {texture}",
        "terrain": "one of {terrain}",
        "motifs": ["one or more of {motifs}"]
      }},
      "chunk_type": "one of {chunk_types}",
      "split_rationale": "why cut here"
    }}
  ],
  "schema_proposals": [
    {{"dimension": "terrain", "value": "new_value", "rationale": "why"}}
  ]
}}"""


def estimate_tokens(text: str) -> int:
    """Estimate token count from text.

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    return max(1, int(len(text.split()) * 1.3))


def _truncate_at_boundary(text: str, max_chars: int) -> str:
    """Truncate text at the nearest sentence or word boundary.

    Tries sentence boundaries first (. ! ? followed by space or end),
    then falls back to the last whitespace before max_chars.
    Never cuts mid-word.
    """
    if len(text) <= max_chars:
        return text

    # Look for the last sentence-ending punctuation within the limit
    window = text[:max_chars]
    for end_char in (". ", "! ", "? ", ".\n", "!\n", "?\n"):
        pos = window.rfind(end_char)
        if pos > max_chars // 3:  # don't go below ~1/3 of the budget
            return window[:pos + 1].rstrip()

    # Check for sentence end at the very end of window
    if window.rstrip()[-1:] in ".!?":
        return window.rstrip()

    # Fall back to last word boundary
    pos = window.rfind(" ")
    if pos > max_chars // 3:
        return window[:pos].rstrip()

    return window.rstrip()


def _turns_to_text(turns: list[dict], max_chars_per_turn: int = 800) -> str:
    """Convert turns to display text for LLM prompt.

    Args:
        turns: List of turn dicts
        max_chars_per_turn: Soft limit per turn — truncates at nearest
            sentence or word boundary.

    Returns:
        Formatted string with turn numbers and text
    """
    lines = []
    for i, t in enumerate(turns):
        text = _truncate_at_boundary(t["text"], max_chars_per_turn)
        lines.append(f"[{i}] {t['sender']}: {text}")
    return "\n".join(lines)


def _build_fallback_chunks(conversation: dict, prompt_preset: str) -> list[dict]:
    """Build fallback chunks when LLM fails.

    Args:
        conversation: Conversation dict
        prompt_preset: Prompt preset name

    Returns:
        List of fallback chunk dicts
    """
    chunks = []
    turns = conversation["turns"]
    for i in range(0, max(1, len(turns)), 2):
        pair = turns[i:i+2]
        text = " ".join(t["text"] for t in pair)
        chunks.append({
            "chunk_id": str(uuid_lib.uuid4()),
            "conversation_id": conversation["id"],
            "source_file": conversation["source_file"],
            "text": text,
            "turns": [i, i+1] if len(pair) > 1 else [i],
            "tags": {"geometry": "linear", "coherence": "loose",
                     "texture": "raw", "terrain": "conceptual", "motifs": []},
            "chunk_type": "exchange",
            "split_rationale": "fallback: LLM failure",
            "schema_version": 0,
            "llm_failure": True,
            "prompt_preset": prompt_preset,
            "schema_proposals": [],
        })
    return chunks


def _post_process_chunks(
    raw_chunks: list[dict],
    proposals: list[dict],
    conversation: dict,
    min_tok: int,
    prompt_preset: str,
    schema_version: int,
) -> list[dict]:
    """Build chunk dicts from LLM output, validate, and repair."""
    turns = conversation["turns"]
    chunks = []

    for i, rc in enumerate(raw_chunks):
        turn_indices = rc.get("turns", [])
        turn_texts = [turns[j]["text"] for j in turn_indices if j < len(turns)]
        text = " ".join(turn_texts)
        tok_count = estimate_tokens(text)

        if tok_count < min_tok and len(raw_chunks) > 1:
            continue
        chunk = {
            "chunk_id": str(uuid_lib.uuid4()),
            "conversation_id": conversation["id"],
            "source_file": conversation["source_file"],
            "text": text,
            "turns": turn_indices,
            "tags": rc.get("tags", {}),
            "chunk_type": rc.get("chunk_type", "exchange"),
            "split_rationale": rc.get("split_rationale", ""),
            "schema_version": schema_version,
            "llm_failure": False,
            "prompt_preset": prompt_preset,
            "schema_proposals": proposals if i == 0 else [],
        }
        chunks.append(chunk)

    if not chunks:
        return _build_fallback_chunks(conversation, prompt_preset)

    # Validate and repair
    validation = validate_chunks(chunks, len(turns))
    if not validation["is_valid"]:
        chunks = repair_chunks(chunks, len(turns), conversation)

    return chunks


def chunk_conversation(
    conversation: dict,
    schema: dict,
    llm,
    config: dict,
    prompt: str,
    prompt_preset: str = "conversational",
    schema_version: int = 1,
    max_retries: int = 2,
    concept_map_summary: str | None = None,
) -> list[dict]:
    """Chunk a conversation using LLM.

    Args:
        conversation: Parsed conversation dict
        schema: Current schema dict
        llm: LLMClient instance
        config: Pipeline config
        prompt: System prompt
        prompt_preset: Name of the prompt preset
        schema_version: Current schema version
        max_retries: Number of retry attempts
        concept_map_summary: Optional JSON summary from discovery phase.
            When provided, it is appended to the system prompt so the LLM
            chunks with archive-aware, pattern-informed boundaries.

    Returns:
        List of chunk dicts
    """
    system_prompt = prompt
    if concept_map_summary:
        system_prompt = (
            prompt
            + "\n\n## Archive Pattern Map\n"
            + "Use these discovered patterns to guide your chunking decisions:\n"
            + concept_map_summary
        )

    dims = schema["dimensions"]
    user_msg = CHUNK_USER_TEMPLATE.format(
        name=conversation.get("conversation_name", "(unnamed)"),
        num_turns=len(conversation["turns"]),
        max_turn_index=len(conversation["turns"]) - 1,
        turns_text=_turns_to_text(conversation["turns"]),
        geometry=", ".join(dims["geometry"]),
        coherence=", ".join(dims["coherence"]),
        texture=", ".join(dims["texture"]),
        terrain=", ".join(dims["terrain"]),
        motifs=", ".join(dims["motifs"]),
        chunk_types=", ".join(schema["chunk_types"]),
    )

    min_tok = config["pipeline"]["chunk_min_tokens"]
    max_tok = config["pipeline"]["chunk_max_tokens"]

    data = None
    for attempt in range(max_retries + 1):
        try:
            response_text = llm.complete(system=system_prompt, user=user_msg)
            text = response_text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text)
            break
        except Exception:
            if attempt == max_retries:
                return _build_fallback_chunks(conversation, prompt_preset)

    if data is None:
        return _build_fallback_chunks(conversation, prompt_preset)

    raw_chunks = data.get("chunks", [])
    proposals = data.get("schema_proposals", [])
    return _post_process_chunks(
        raw_chunks, proposals, conversation,
        min_tok, prompt_preset, schema_version,
    )


def chunk_conversations_batch(
    conversations: list[dict],
    schema: dict,
    llm,
    config: dict,
    prompt: str,
    prompt_preset: str = "conversational",
    schema_version: int = 1,
    concept_map_summary: str | None = None,
    on_complete=None,
) -> list[list[dict]]:
    """Chunk multiple conversations concurrently.

    Uses llm.complete_batch() to fire requests in parallel when
    llm.concurrency > 1, otherwise falls back to sequential processing.

    Args:
        conversations: List of parsed conversation dicts.
        schema: Current schema dict.
        llm: LLMClient instance (must have complete_batch and concurrency).
        config: Pipeline config.
        prompt: System prompt.
        prompt_preset: Name of the prompt preset.
        schema_version: Current schema version.
        concept_map_summary: Optional discovery map summary JSON.
        on_complete: Optional callback(index, chunks_or_none, error_or_none)
            called as each conversation finishes.

    Returns:
        List of chunk lists, one per conversation (same order as input).
        Failed conversations get fallback chunks.
    """
    system_prompt = prompt
    if concept_map_summary:
        system_prompt = (
            prompt
            + "\n\n## Archive Pattern Map\n"
            + "Use these discovered patterns to guide your chunking decisions:\n"
            + concept_map_summary
        )

    dims = schema["dimensions"]
    min_tok = config["pipeline"]["chunk_min_tokens"]
    max_tok = config["pipeline"]["chunk_max_tokens"]

    # Build all request payloads
    request_list: list[tuple[str, str]] = []
    for conv in conversations:
        user_msg = CHUNK_USER_TEMPLATE.format(
            name=conv.get("conversation_name", "(unnamed)"),
            num_turns=len(conv["turns"]),
            max_turn_index=len(conv["turns"]) - 1,
            turns_text=_turns_to_text(conv["turns"]),
            geometry=", ".join(dims["geometry"]),
            coherence=", ".join(dims["coherence"]),
            texture=", ".join(dims["texture"]),
            terrain=", ".join(dims["terrain"]),
            motifs=", ".join(dims["motifs"]),
            chunk_types=", ".join(schema["chunk_types"]),
        )
        request_list.append((system_prompt, user_msg))

    # Fire all requests (concurrent if concurrency > 1, sequential otherwise)
    all_results: list[list[dict]] = [[] for _ in conversations]

    def _on_result(idx, text, error):
        conv = conversations[idx]
        if error or text is None:
            all_results[idx] = _build_fallback_chunks(conv, prompt_preset)
            if on_complete:
                on_complete(idx, all_results[idx], error)
            return

        try:
            cleaned = text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            data = json.loads(cleaned)
        except (json.JSONDecodeError, IndexError):
            all_results[idx] = _build_fallback_chunks(conv, prompt_preset)
            if on_complete:
                on_complete(idx, all_results[idx], None)
            return

        raw_chunks = data.get("chunks", [])
        proposals = data.get("schema_proposals", [])
        all_results[idx] = _post_process_chunks(
            raw_chunks, proposals, conv,
            min_tok, prompt_preset, schema_version,
        )
        if on_complete:
            on_complete(idx, all_results[idx], None)

    llm.complete_batch(request_list, on_complete=_on_result)

    return all_results
