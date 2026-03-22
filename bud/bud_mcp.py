"""MCP server for Bud RAG Archive - provides semantic search via stdio."""

import json
import random
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field

# Resolve bud package - works whether installed or run from source
try:
    from bud.config import load_config, get_output_dir
    from bud.lib.embeddings import EmbeddingClient
    from bud.lib.store import VectorStore
    from bud.lib.model_registry import resolve_embedding_model
    from bud.lib.schema_manager import SchemaManager
    from bud.mcp_logger import log_tool_call, start_logging_session, MCPLogger
except ModuleNotFoundError:
    _here = Path(__file__).resolve().parent
    sys.path.insert(0, str(_here))
    from bud.config import load_config, get_output_dir
    from bud.lib.embeddings import EmbeddingClient
    from bud.lib.store import VectorStore
    from bud.lib.model_registry import resolve_embedding_model
    from bud.lib.schema_manager import SchemaManager
    from bud.mcp_logger import log_tool_call, start_logging_session, MCPLogger

# Constants
DEFAULT_K = 5
MAX_K = 20
ORIENT_SAMPLE_SIZE = 12
ORIENT_PREVIEW_CHARS = 200

# Import SESSIONS_DIR and CURRENT_SESSION_FILE from mcp_logger
from bud.mcp_logger import SESSIONS_DIR, CURRENT_SESSION_FILE


# Lifespan - loads index on first access, keeps warm for session
@asynccontextmanager
async def bud_lifespan():
    """Lazy load and hold the bud index for session lifetime."""
    config = load_config()
    output_dir = get_output_dir()

    model_cfg = resolve_embedding_model(
        config.get("embeddings", {}).get("model", "")
    )

    index_path = str(output_dir / "index" / "chunks")
    store = VectorStore(index_path, dim=model_cfg["dimension"])
    store.load()

    embedding_client = EmbeddingClient(config)

    schema_path = str(output_dir / "schema.json")
    schema_mgr = SchemaManager(str(schema_path))
    schema = schema_mgr.load()

    # Initialize MCP logger with session
    logger = MCPLogger()
    session_id = logger.start_session()  # Start/resume session
    session_file = CURRENT_SESSION_FILE.parent / f"{session_id}.jsonl"

    yield {
        "store": store,
        "embedding_client": embedding_client,
        "schema": schema,
        "output_dir": output_dir,
        "config": config,
        "logger": logger,
        "session_file": session_file,
    }


mcp = FastMCP("bud_mcp", lifespan=bud_lifespan)


def _get_state(ctx) -> dict:
    """Pull lifespan state from context."""
    return ctx.request_context.lifespan_state


def _format_chunk(chunk: dict, rank: Optional[int] = None) -> dict:
    """Produce a clean, structured chunk record for tool responses."""
    record: dict[str, Any] = {}
    if rank is not None:
        record["rank"] = rank
    record["chunk_id"] = chunk.get("chunk_id", "")
    record["score"] = round(float(chunk.get("score", 0.0)), 4) if "score" in chunk else None
    record["chunk_type"] = chunk.get("chunk_type", "")
    record["tags"] = chunk.get("tags", {})
    record["source_file"] = chunk.get("source_file", chunk.get("source", ""))
    record["conversation_id"] = chunk.get("conversation_id", "")
    record["turns"] = chunk.get("turns", [])
    record["text"] = chunk.get("text", "")
    return record


def _error(msg: str) -> str:
    return json.dumps({"error": msg})


class RecallInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")
    query: str = Field(..., description="Natural language query to search the conversation archive.", min_length=1, max_length=2000)
    k: int = Field(default=DEFAULT_K, description=f"Number of chunks to retrieve (1-{MAX_K}). Default {DEFAULT_K}.", ge=1, le=MAX_K)


@mcp.tool(name="bud_recall", annotations={"title": "Recall from Archive", "readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
async def bud_recall(params: RecallInput, ctx) -> str:
    """Search the personal conversation archive using semantic similarity."""
    state = _get_state(ctx)
    store: VectorStore = state["store"]
    embedding_client: EmbeddingClient = state["embedding_client"]

    if store.count() == 0:
        result = _error("Index is empty. Run 'bud process' first.")
        return result

    start_time = time.perf_counter()
    try:
        vector = embedding_client.embed(params.query)
    except Exception as e:
        duration = (time.perf_counter() - start_time) * 1000
        result = _error(f"Failed to embed query: {e}")
        log_tool_call("bud_recall", {"query": params.query, "k": params.k}, {"error": str(e)}, duration)
        return result

    try:
        raw_results = store.search(vector, k=params.k)
        duration = (time.perf_counter() - start_time) * 1000
        results = [_format_chunk(chunk, rank=i + 1) for i, chunk in enumerate(raw_results)]

        result_data = {
            "query": params.query,
            "k": params.k,
            "total_in_index": store.count(),
            "results": results,
        }
        result = json.dumps(result_data, indent=2)

        log_tool_call("bud_recall", {"query": params.query, "k": params.k}, result_data, duration)
        return result
    except Exception as e:
        duration = (time.perf_counter() - start_time) * 1000
        result = _error(f"Search failed: {e}")
        log_tool_call("bud_recall", {"query": params.query, "k": params.k}, {"error": str(e)}, duration)
        return result


@mcp.tool(name="bud_orient", annotations={"title": "Orient to Archive", "readOnlyHint": True, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False})
async def bud_orient(ctx) -> str:
    """Get a structural snapshot of the conversation archive."""
    state = _get_state(ctx)
    store: VectorStore = state["store"]
    schema: dict = state["schema"]

    if store.count() == 0:
        result = _error("Index is empty. Run 'bud process' first.")
        return result

    start_time = time.perf_counter()

    all_meta = store._metadata
    sample_size = min(ORIENT_SAMPLE_SIZE, len(all_meta))
    sampled = random.sample(all_meta, sample_size) if len(all_meta) > sample_size else all_meta

    sample_records = []
    for chunk in sampled:
        sample_records.append({
            "chunk_id": chunk.get("chunk_id", ""),
            "chunk_type": chunk.get("chunk_type", ""),
            "tags": chunk.get("tags", {}),
            "source_file": chunk.get("source_file", chunk.get("source", "")),
            "turns": chunk.get("turns", []),
            "preview": chunk.get("text", "")[:ORIENT_PREVIEW_CHARS],
        })

    duration = (time.perf_counter() - start_time) * 1000

    result_data = {
        "index": {
            "total_chunks": store.count(),
            "schema_version": schema.get("version", 1),
            "dimensions": schema.get("dimensions", {}),
            "chunk_types": schema.get("chunk_types", []),
        },
        "sample": sample_records,
    }
    result = json.dumps(result_data, indent=2)

    log_tool_call("bud_orient", {}, result_data, duration)
    return result


class ReflectInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")
    dimension: str = Field(..., description="Tag dimension to filter on (geometry, coherence, texture, terrain, motifs, chunk_type).", min_length=1, max_length=64)
    value: str = Field(..., description="Value within that dimension to match.", min_length=1, max_length=64)
    limit: int = Field(default=10, description="Maximum number of matching chunks to return (1-50). Default 10.", ge=1, le=50)


@mcp.tool(name="bud_reflect", annotations={"title": "Reflect on Archive by Tag", "readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
async def bud_reflect(params: ReflectInput, ctx) -> str:
    """Retrieve chunks from the archive filtered by structural tag dimension and value."""
    state = _get_state(ctx)
    store: VectorStore = state["store"]

    if store.count() == 0:
        result = _error("Index is empty. Run 'bud process' first.")
        return result

    # Get schema from state for dimension validation
    schema: dict = state.get("schema", {})
    valid_dimensions = set(schema.get("dimensions", {}).keys())
    # chunk_type is a special case - it's not in dimensions but is valid
    valid_dimensions.add("chunk_type")

    dim = params.dimension.lower().strip()

    # Validate dimension against schema
    if dim not in valid_dimensions:
        result = _error(f"Invalid dimension '{params.dimension}'. Valid dimensions are: {', '.join(sorted(valid_dimensions))}")
        return result

    start_time = time.perf_counter()

    val = params.value.lower().strip()

    matched = []
    for chunk in store._metadata:
        tags = chunk.get("tags", {})

        if dim == "chunk_type":
            if chunk.get("chunk_type", "").lower() == val:
                matched.append(chunk)
            continue

        tag_val = tags.get(dim)
        if tag_val is None:
            continue

        if isinstance(tag_val, list):
            if val in [m.lower() for m in tag_val]:
                matched.append(chunk)
        else:
            if tag_val.lower() == val:
                matched.append(chunk)

    random.shuffle(matched)
    returned = matched[: params.limit]
    results = [_format_chunk(chunk) for chunk in returned]

    duration = (time.perf_counter() - start_time) * 1000

    result_data = {
        "dimension": params.dimension,
        "value": params.value,
        "total_matched": len(matched),
        "returned": len(returned),
        "results": results,
    }
    result = json.dumps(result_data, indent=2)

    log_tool_call("bud_reflect", {"dimension": params.dimension, "value": params.value, "limit": params.limit}, result_data, duration)
    return result


# ---------------------------------------------------------------------------
# Tool: bud_context - Session tracking for Claude
# ---------------------------------------------------------------------------


class ContextInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")
    history_size: int = Field(default=10, description="Number of recent tool calls to include in history (1-50). Default 10.", ge=1, le=50)


@mcp.tool(name="bud_context", annotations={"title": "Get Session Context", "readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
async def bud_context(params: ContextInput, ctx) -> str:
    """Get the current MCP session context for Claude conversation tracking.

    Returns the session ID and recent tool call history so Claude can:
    - Detect if this is a new session or continuation of a previous one
    - Avoid duplicating tool calls with the same parameters
    - Track conversation state across MCP tool invocations

    Args:
        params (ContextInput): Input containing history_size to control
                               how many recent calls to return

    Returns:
        str: JSON object with schema:
        {
          "session_id": "mcp-abc123",
          "is_new_session": false,
          "history": [
            {
              "tool_name": "bud_orient",
              "timestamp": "2026-03-22T10:30:00.123456Z",
              "parameters": {}
            },
            {
              "tool_name": "bud_recall",
              "timestamp": "2026-03-22T10:30:01.234567Z",
              "parameters": {"query": "auth", "k": 5}
            }
          ],
          "total_calls_in_session": 5
        }
    """
    state = _get_state(ctx)
    logger: MCPLogger = state.get("logger")
    session_file = state.get("session_file")

    if logger is None or session_file is None:
        # Fallback: try to get logger from module
        from bud.mcp_logger import get_logger
        logger = get_logger()

    session_id = logger._current_session_id or "unknown"

    # Load history from log file
    history = []
    if session_file and session_file.exists():
        try:
            with open(session_file, "r") as f:
                lines = f.readlines()
                # Get last N entries, parse them
                for line in lines[-params.history_size:]:
                    try:
                        entry = json.loads(line.strip())
                        history.append({
                            "tool_name": entry.get("tool_name", "unknown"),
                            "timestamp": entry.get("timestamp", ""),
                            "parameters": entry.get("parameters", {}),
                        })
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass

    total_calls = len(history)

    # Use _was_new_session from logger if available, otherwise check if session file existed before
    is_new_session = getattr(logger, '_was_new_session', False)

    return json.dumps({
        "session_id": session_id,
        "is_new_session": is_new_session,
        "history": history,
        "total_calls_in_session": total_calls,
    }, indent=2)


# Entry point - starts logging session and MCP server
def main():
    """Start the MCP server with logging initialized."""
    session_id = start_logging_session()
    print(f"MCP session started: {session_id}", file=sys.stderr)
    mcp.run()


if __name__ == "__main__":
    main()
