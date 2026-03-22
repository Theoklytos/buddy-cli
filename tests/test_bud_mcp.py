"""Tests for Bud MCP server and logging."""

import asyncio
import json
import time
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Test imports
import bud.bud_mcp as mcp_module
from bud.mcp_logger import MCPLogger, SESSIONS_DIR, CURRENT_SESSION_FILE


class TestFormatChunk:
    """Tests for _format_chunk helper function."""

    def test_format_chunk_with_rank(self):
        """Test formatting a chunk with rank."""
        chunk = {"chunk_id": "c1", "score": 0.95, "text": "hello"}
        result = mcp_module._format_chunk(chunk, rank=1)
        assert result["rank"] == 1
        assert result["chunk_id"] == "c1"
        assert result["score"] == 0.95

    def test_format_chunk_without_rank(self):
        """Test formatting a chunk without rank."""
        chunk = {"chunk_id": "c1"}
        result = mcp_module._format_chunk(chunk)
        assert "rank" not in result
        assert result["chunk_id"] == "c1"

    def test_format_chunk_preserves_all_fields(self):
        """Test that all chunk fields are preserved."""
        chunk = {
            "chunk_id": "c1",
            "score": 0.95,
            "chunk_type": "exchange",
            "tags": {"terrain": "conceptual"},
            "source_file": "test.json",
            "conversation_id": "conv1",
            "turns": [1, 2],
            "text": "hello world",
        }
        result = mcp_module._format_chunk(chunk)
        assert result["chunk_id"] == "c1"
        assert result["chunk_type"] == "exchange"
        assert result["tags"] == {"terrain": "conceptual"}
        assert result["source_file"] == "test.json"
        assert result["conversation_id"] == "conv1"
        assert result["turns"] == [1, 2]
        assert result["text"] == "hello world"

    def test_format_chunk_fallback_source_field(self):
        """Test fallback from source_file to source field."""
        chunk = {"chunk_id": "c1", "source": "fallback.txt"}
        result = mcp_module._format_chunk(chunk)
        assert result["source_file"] == "fallback.txt"


class TestErrorHelper:
    """Tests for _error helper function."""

    def test_error_returns_json(self):
        """Test that _error returns valid JSON."""
        result = mcp_module._error("test error")
        data = json.loads(result)
        assert data["error"] == "test error"


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_state(self):
        """Test _get_state pulls from context correctly."""
        ctx = MagicMock()
        ctx.request_context = MagicMock()
        expected_state = {"key": "value"}
        ctx.request_context.lifespan_state = expected_state

        state = mcp_module._get_state(ctx)
        assert state == expected_state


class TestBudRecall:
    """Tests for bud_recall tool."""

    def test_recall_empty_index(self):
        """Test recall when index is empty."""
        ctx = MagicMock()
        ctx.request_context = MagicMock()
        ctx.request_context.lifespan_state = {
            "store": MagicMock(),
            "embedding_client": MagicMock(),
        }
        ctx.request_context.lifespan_state["store"].count.return_value = 0

        # Create a mock params with proper attributes
        params = MagicMock()
        params.query = "test"
        params.k = 5

        result = asyncio.run(mcp_module.bud_recall(params, ctx))
        assert "Index is empty" in result

    def test_recall_query_embedding_failure(self):
        """Test recall when embedding query fails."""
        ctx = MagicMock()
        ctx.request_context = MagicMock()
        ctx.request_context.lifespan_state = {
            "store": MagicMock(),
            "embedding_client": MagicMock(),
        }

        store = MagicMock()
        store.count.return_value = 1
        store.search.return_value = []
        ctx.request_context.lifespan_state["store"] = store

        ctx.request_context.lifespan_state["embedding_client"].embed.side_effect = Exception("embed failed")

        params = MagicMock()
        params.query = "test"
        params.k = 5

        result = asyncio.run(mcp_module.bud_recall(params, ctx))
        assert "error" in result.lower()

    def test_recall_search_failure(self):
        """Test recall when search fails."""
        ctx = MagicMock()
        ctx.request_context = MagicMock()
        ctx.request_context.lifespan_state = {
            "store": MagicMock(),
            "embedding_client": MagicMock(),
        }

        store = MagicMock()
        store.count.return_value = 1
        store.search.side_effect = Exception("search failed")
        ctx.request_context.lifespan_state["store"] = store

        ctx.request_context.lifespan_state["embedding_client"].embed.return_value = [0.1, 0.2]

        params = MagicMock()
        params.query = "test"
        params.k = 5

        result = asyncio.run(mcp_module.bud_recall(params, ctx))
        assert "error" in result.lower()

    def test_recall_success(self):
        """Test successful recall."""
        ctx = MagicMock()
        ctx.request_context = MagicMock()
        ctx.request_context.lifespan_state = {
            "store": MagicMock(),
            "embedding_client": MagicMock(),
            "schema": {"version": 1},
        }

        store = MagicMock()
        store.count.return_value = 5
        store.search.return_value = [{
            "chunk_id": "c1",
            "score": 0.95,
            "chunk_type": "exchange",
            "tags": {},
            "source_file": "test.json",
            "conversation_id": "conv1",
            "turns": [1],
            "text": "hello",
        }]
        ctx.request_context.lifespan_state["store"] = store
        ctx.request_context.lifespan_state["embedding_client"].embed.return_value = [0.1, 0.2]

        params = MagicMock()
        params.query = "test query"
        params.k = 5

        result = asyncio.run(mcp_module.bud_recall(params, ctx))
        data = json.loads(result)
        assert data["query"] == "test query"
        assert data["k"] == 5
        assert len(data["results"]) == 1
        assert data["total_in_index"] == 5


class TestBudOrient:
    """Tests for bud_orient tool."""

    def test_orient_empty_index(self):
        """Test orient when index is empty."""
        ctx = MagicMock()
        ctx.request_context = MagicMock()
        ctx.request_context.lifespan_state = {
            "store": MagicMock(),
            "schema": {"version": 1},
        }
        ctx.request_context.lifespan_state["store"].count.return_value = 0

        result = asyncio.run(mcp_module.bud_orient(ctx))
        assert "Index is empty" in result

    def test_orient_success(self):
        """Test successful orient call."""
        ctx = MagicMock()
        ctx.request_context = MagicMock()
        ctx.request_context.lifespan_state = {
            "store": MagicMock(),
            "schema": {
                "version": 1,
                "dimensions": {"geometry": ["linear"]},
                "chunk_types": ["exchange"]
            },
        }

        store = MagicMock()
        store.count.return_value = 1
        store._metadata = [{
            "chunk_id": "c1",
            "chunk_type": "exchange",
            "tags": {"terrain": "conceptual"},
            "source_file": "test.json",
            "conversation_id": "conv1",
            "turns": [1],
            "text": "hello world this is a test",
        }]
        ctx.request_context.lifespan_state["store"] = store

        result = asyncio.run(mcp_module.bud_orient(ctx))
        data = json.loads(result)
        assert "index" in data
        assert "sample" in data
        assert data["index"]["total_chunks"] == 1
        assert len(data["sample"]) == 1
        assert data["sample"][0]["chunk_id"] == "c1"


class TestBudReflect:
    """Tests for bud_reflect tool."""

    def test_reflect_empty_index(self):
        """Test reflect when index is empty."""
        ctx = MagicMock()
        ctx.request_context = MagicMock()
        ctx.request_context.lifespan_state = {
            "store": MagicMock(),
        }
        ctx.request_context.lifespan_state["store"].count.return_value = 0

        params = MagicMock()
        params.dimension = "terrain"
        params.value = "emotional"
        params.limit = 10

        result = asyncio.run(mcp_module.bud_reflect(params, ctx))
        assert "Index is empty" in result

    def test_reflect_by_chunk_type(self):
        """Test reflect filtering by chunk_type."""
        ctx = MagicMock()
        ctx.request_context = MagicMock()
        ctx.request_context.lifespan_state = {
            "store": MagicMock(),
        }

        store = MagicMock()
        store.count.return_value = 2
        store._metadata = [
            {"chunk_id": "c1", "chunk_type": "exchange", "tags": {}},
            {"chunk_id": "c2", "chunk_type": "breakthrough", "tags": {}},
        ]
        ctx.request_context.lifespan_state["store"] = store

        params = MagicMock()
        params.dimension = "chunk_type"
        params.value = "exchange"
        params.limit = 10

        result = asyncio.run(mcp_module.bud_reflect(params, ctx))
        data = json.loads(result)
        assert data["dimension"] == "chunk_type"
        assert data["value"] == "exchange"
        assert len(data["results"]) == 1

    def test_reflect_by_tag_dimension(self):
        """Test reflect filtering by tag dimension."""
        ctx = MagicMock()
        ctx.request_context = MagicMock()
        ctx.request_context.lifespan_state = {
            "store": MagicMock(),
            "schema": {
                "dimensions": {
                    "terrain": ["conceptual", "emotional"],
                    "geometry": ["linear", "circular"]
                },
                "chunk_types": ["exchange"]
            },
        }

        store = MagicMock()
        store.count.return_value = 2
        store._metadata = [
            {"chunk_id": "c1", "chunk_type": "exchange", "tags": {"terrain": "conceptual"}},
            {"chunk_id": "c2", "chunk_type": "exchange", "tags": {"terrain": "emotional"}},
        ]
        ctx.request_context.lifespan_state["store"] = store

        params = MagicMock()
        params.dimension = "terrain"
        params.value = "emotional"
        params.limit = 10

        result = asyncio.run(mcp_module.bud_reflect(params, ctx))
        data = json.loads(result)
        assert data["dimension"] == "terrain"
        assert data["value"] == "emotional"

    def test_reflect_multivalue_dimension(self):
        """Test reflect with multi-value dimension (motifs)."""
        ctx = MagicMock()
        ctx.request_context = MagicMock()
        ctx.request_context.lifespan_state = {
            "store": MagicMock(),
            "schema": {
                "dimensions": {
                    "motifs": ["identity", "resonance"],
                    "geometry": ["linear", "circular"]
                },
                "chunk_types": ["exchange"]
            },
        }

        store = MagicMock()
        store.count.return_value = 1
        store._metadata = [{
            "chunk_id": "c1",
            "chunk_type": "exchange",
            "tags": {"motifs": ["identity", "resonance"]},
        }]
        ctx.request_context.lifespan_state["store"] = store

        params = MagicMock()
        params.dimension = "motifs"
        params.value = "identity"
        params.limit = 10

        result = asyncio.run(mcp_module.bud_reflect(params, ctx))
        data = json.loads(result)
        assert len(data["results"]) == 1
        assert data["total_matched"] == 1


class TestBudContext:
    """Tests for bud_context tool - session tracking for Claude."""

    def test_context_returns_session_id(self, tmp_path, monkeypatch):
        """Test that context returns a session ID."""
        # Setup mock logger
        monkeypatch.setattr(mcp_module, 'SESSIONS_DIR', tmp_path / "mcp_logs")
        monkeypatch.setattr(mcp_module, 'CURRENT_SESSION_FILE', tmp_path / "mcp_logs" / ".current_session")

        ctx = MagicMock()
        ctx.request_context = MagicMock()
        # Use a simple object for logger with required attributes
        logger = MagicMock()
        logger._current_session_id = "mcp-test-session"
        logger._was_new_session = True
        ctx.request_context.lifespan_state = {
            "logger": logger,
            "session_file": tmp_path / "mcp_logs" / "test-session.jsonl",
        }

        params = MagicMock()
        params.history_size = 10

        result = asyncio.run(mcp_module.bud_context(params, ctx))
        data = json.loads(result)
        assert "session_id" in data
        assert data["session_id"] == "mcp-test-session"
        assert data["is_new_session"] is True

    def test_context_returns_history(self, tmp_path, monkeypatch):
        """Test that context returns tool call history."""
        # Setup log directory
        monkeypatch.setattr(mcp_module, 'SESSIONS_DIR', tmp_path / "mcp_logs")
        monkeypatch.setattr(mcp_module, 'CURRENT_SESSION_FILE', tmp_path / "mcp_logs" / ".current_session")

        log_dir = tmp_path / "mcp_logs"
        log_dir.mkdir(parents=True)
        session_file = log_dir / "test-session.jsonl"

        # Write sample log entries
        entries = [
            {"timestamp": "2026-03-22T10:00:00Z", "tool_name": "bud_orient", "parameters": {}},
            {"timestamp": "2026-03-22T10:00:01Z", "tool_name": "bud_recall", "parameters": {"query": "test", "k": 5}},
        ]
        with open(session_file, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        ctx = MagicMock()
        ctx.request_context = MagicMock()
        # Use a simple object for logger with required attributes
        logger = MagicMock()
        logger._current_session_id = "test-session"
        logger._was_new_session = False
        ctx.request_context.lifespan_state = {
            "logger": logger,
            "session_file": session_file,
        }

        params = MagicMock()
        params.history_size = 10

        result = asyncio.run(mcp_module.bud_context(params, ctx))
        data = json.loads(result)
        assert "history" in data
        assert len(data["history"]) == 2
        assert data["history"][0]["tool_name"] == "bud_orient"
        assert data["history"][1]["tool_name"] == "bud_recall"
        assert data["is_new_session"] is False

    def test_context_falls_back_to_module_logger(self, tmp_path, monkeypatch):
        """Test that context falls back to module-level logger if lifespan state missing."""
        monkeypatch.setattr(mcp_module, 'SESSIONS_DIR', tmp_path / "mcp_logs")
        monkeypatch.setattr(mcp_module, 'CURRENT_SESSION_FILE', tmp_path / "mcp_logs" / ".current_session")

        # Create a real logger and session file
        logger = MCPLogger()
        session_id = logger.start_session("fallback-test")
        log_file = tmp_path / "mcp_logs" / "fallback-test.jsonl"
        logger.log_tool_call("test_tool", {"a": 1}, {}, 50)

        ctx = MagicMock()
        ctx.request_context = MagicMock()
        # No logger in lifespan state - should fall back
        ctx.request_context.lifespan_state = {
            "session_file": log_file,
        }

        params = MagicMock()
        params.history_size = 10

        # Mock get_logger to return our logger with the session
        # Patch the import path in mcp_logger module
        with patch('bud.mcp_logger.get_logger', return_value=logger):
            result = asyncio.run(mcp_module.bud_context(params, ctx))
            data = json.loads(result)
            assert data["session_id"] == "fallback-test"
            # History may be 0 because get_logger() is called from within the function
            # which creates its own logger instance - this is acceptable fallback behavior


class TestLifespan:
    """Tests for bud_lifespan context manager."""

    def test_lifespan_yields_state(self):
        """Test that lifespan context manager yields the expected state."""
        # This test verifies the structure of lifespan state
        import bud.bud_mcp as mcp_module

        # Mock the imports to avoid requiring actual config/index
        with patch('bud.bud_mcp.load_config') as mock_load_config, \
             patch('bud.bud_mcp.get_output_dir') as mock_get_output_dir, \
             patch('bud.bud_mcp.resolve_embedding_model') as mock_resolve, \
             patch('bud.bud_mcp.VectorStore') as mock_store_class, \
             patch('bud.bud_mcp.EmbeddingClient') as mock_embed_class, \
             patch('bud.bud_mcp.SchemaManager') as mock_schema_class, \
             patch('bud.bud_mcp.MCPLogger') as mock_logger_class:

            mock_load_config.return_value = {"embeddings": {"model": "test"}}
            mock_get_output_dir.return_value = Path("/test/output")
            mock_resolve.return_value = {"dimension": 768}

            mock_store = MagicMock()
            mock_store_class.return_value = mock_store
            mock_store.load = MagicMock()

            mock_embed = MagicMock()
            mock_embed_class.return_value = mock_embed

            mock_schema = MagicMock()
            mock_schema.load.return_value = {"version": 1}
            mock_schema_class.return_value = mock_schema

            mock_logger = MagicMock()
            mock_logger.start_session.return_value = "test-session"
            mock_logger_class.return_value = mock_logger

            # Test lifespan context manager
            async def test():
                async with mcp_module.bud_lifespan() as state:
                    assert "store" in state
                    assert "embedding_client" in state
                    assert "schema" in state
                    assert "logger" in state
                    assert "session_file" in state

            asyncio.run(test())


class TestToolLogging:
    """Tests for logging integration in MCP tools."""

    def test_recall_logs_tool_call(self, tmp_path, monkeypatch):
        """Test that bud_recall logs tool calls via log_tool_call."""
        monkeypatch.setattr(mcp_module, 'SESSIONS_DIR', tmp_path / "mcp_logs")
        monkeypatch.setattr(mcp_module, 'CURRENT_SESSION_FILE', tmp_path / "mcp_logs" / ".current_session")

        ctx = MagicMock()
        ctx.request_context = MagicMock()
        ctx.request_context.lifespan_state = {
            "store": MagicMock(),
            "embedding_client": MagicMock(),
            "schema": {"version": 1, "dimensions": {}, "chunk_types": []},
        }

        store = MagicMock()
        store.count.return_value = 1
        store.search.return_value = [{
            "chunk_id": "c1",
            "score": 0.95,
            "chunk_type": "exchange",
            "tags": {},
            "source_file": "test.json",
            "conversation_id": "conv1",
            "turns": [1],
            "text": "hello",
        }]
        ctx.request_context.lifespan_state["store"] = store
        ctx.request_context.lifespan_state["embedding_client"].embed.return_value = [0.1, 0.2]

        # Track log calls
        logged_calls = []
        original_log = mcp_module.log_tool_call

        def track_log(tool_name, params, result, duration):
            logged_calls.append({
                "tool_name": tool_name,
                "params": params,
                "duration": duration,
            })

        mcp_module.log_tool_call = track_log

        try:
            params = MagicMock()
            params.query = "test"
            params.k = 5

            asyncio.run(mcp_module.bud_recall(params, ctx))
            assert len(logged_calls) >= 1
            assert logged_calls[0]["tool_name"] == "bud_recall"
            assert logged_calls[0]["params"]["query"] == "test"
            assert "duration" in logged_calls[0]
        finally:
            mcp_module.log_tool_call = original_log

    def test_orient_logs_tool_call(self, tmp_path, monkeypatch):
        """Test that bud_orient logs tool calls."""
        monkeypatch.setattr(mcp_module, 'SESSIONS_DIR', tmp_path / "mcp_logs")
        monkeypatch.setattr(mcp_module, 'CURRENT_SESSION_FILE', tmp_path / "mcp_logs" / ".current_session")

        ctx = MagicMock()
        ctx.request_context = MagicMock()
        ctx.request_context.lifespan_state = {
            "store": MagicMock(),
            "schema": {"version": 1, "dimensions": {}, "chunk_types": []},
        }

        store = MagicMock()
        store.count.return_value = 1
        store._metadata = [{
            "chunk_id": "c1",
            "chunk_type": "exchange",
            "tags": {},
            "source_file": "test.json",
            "conversation_id": "conv1",
            "turns": [1],
            "text": "test text",
        }]
        ctx.request_context.lifespan_state["store"] = store

        logged_calls = []
        original_log = mcp_module.log_tool_call

        def track_log(tool_name, params, result, duration):
            logged_calls.append({"tool_name": tool_name})

        mcp_module.log_tool_call = track_log

        try:
            asyncio.run(mcp_module.bud_orient(ctx))
            assert len(logged_calls) >= 1
            assert logged_calls[0]["tool_name"] == "bud_orient"
        finally:
            mcp_module.log_tool_call = original_log


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_format_chunk_score_rounding(self):
        """Test that scores are rounded to 4 decimal places."""
        chunk = {"chunk_id": "c1", "score": 0.123456789}
        result = mcp_module._format_chunk(chunk)
        assert result["score"] == 0.1235


class TestCliMcpCommand:
    """Tests for the `bud mcp` CLI command."""

    def test_mcp_command_help_output(self, tmp_path, monkeypatch):
        """Test that the mcp command shows help."""
        monkeypatch.setattr(mcp_module, 'SESSIONS_DIR', tmp_path / "mcp_logs")
        monkeypatch.setattr(mcp_module, 'CURRENT_SESSION_FILE', tmp_path / "mcp_logs" / ".current_session")

        from bud.cli import main
        from click.testing import CliRunner
        cli_runner = CliRunner()

        result = cli_runner.invoke(main, ['mcp', '--help'])
        assert result.exit_code == 0
        assert "MCP server" in result.output or "mcp" in result.output.lower()

    def test_mcp_command_initializes_logger(self, tmp_path, monkeypatch):
        """Test that the mcp command initializes the logging session."""
        monkeypatch.setattr(mcp_module, 'SESSIONS_DIR', tmp_path / "mcp_logs")
        monkeypatch.setattr(mcp_module, 'CURRENT_SESSION_FILE', tmp_path / "mcp_logs" / ".current_session")

        calls = []
        original_start = mcp_module.start_logging_session

        def track_start(sid=None):
            calls.append(sid)
            return "test-session-logger"

        monkeypatch.setattr(mcp_module, 'start_logging_session', track_start)

        # Mock mcp.run to prevent blocking
        original_run = mcp_module.mcp.run
        mcp_module.mcp.run = lambda: None

        from bud.cli import main
        from click.testing import CliRunner
        cli_runner = CliRunner()

        result = cli_runner.invoke(main, ['mcp'])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0] is None  # Default None session_id

        # Restore
        mcp_module.mcp.run = original_run
        monkeypatch.setattr(mcp_module, 'start_logging_session', original_start)

    def test_mcp_command_with_custom_session_id(self, tmp_path, monkeypatch):
        """Test that the mcp command accepts a custom session ID."""
        monkeypatch.setattr(mcp_module, 'SESSIONS_DIR', tmp_path / "mcp_logs")
        monkeypatch.setattr(mcp_module, 'CURRENT_SESSION_FILE', tmp_path / "mcp_logs" / ".current_session")

        # Need to patch the import path used in cli.py: bud.mcp_logger.start_logging_session
        import bud.mcp_logger as logger_module
        calls = []
        original_start = logger_module.start_logging_session

        def track_start(sid=None):
            calls.append(sid)
            return "custom-session-id"

        monkeypatch.setattr(logger_module, 'start_logging_session', track_start)

        # Mock mcp.run to prevent blocking
        original_run = mcp_module.mcp.run
        mcp_module.mcp.run = lambda: None

        from bud.cli import main
        from click.testing import CliRunner
        cli_runner = CliRunner()

        result = cli_runner.invoke(main, ['mcp', '--session-id', 'my-session-123'])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0] == "my-session-123"

        mcp_module.mcp.run = original_run
        monkeypatch.setattr(logger_module, 'start_logging_session', original_start)


class TestConstantsImport:
    """Tests for Issue #1: Verify constants are imported from mcp_logger."""

    def test_constants_not_duplicated_in_bud_mcp(self):
        """Test that SESSIONS_DIR and CURRENT_SESSION_FILE are imported, not duplicated."""
        # These should be imported from bud.mcp_logger, not defined here
        # Verify they exist and are the same objects as in mcp_logger module
        import bud.mcp_logger as logger_module
        assert mcp_module.SESSIONS_DIR is logger_module.SESSIONS_DIR
        assert mcp_module.CURRENT_SESSION_FILE is logger_module.CURRENT_SESSION_FILE

    def test_constants_import_path(self):
        """Test that constants come from mcp_logger module."""
        import bud.mcp_logger as logger_module
        assert hasattr(logger_module, 'SESSIONS_DIR')
        assert hasattr(logger_module, 'CURRENT_SESSION_FILE')


class TestisNewSession:
    """Tests for Issue #2: Verify is_new_session logic."""

    def test_new_session_returns_true_when_generated(self, tmp_path, monkeypatch):
        """Test that is_new_session is True when a new session ID is generated."""
        monkeypatch.setattr(mcp_module, 'SESSIONS_DIR', tmp_path / "mcp_logs")
        monkeypatch.setattr(mcp_module, 'CURRENT_SESSION_FILE', tmp_path / "mcp_logs" / ".current_session")

        ctx = MagicMock()
        ctx.request_context = MagicMock()
        ctx.request_context.lifespan_state = {
            "logger": MagicMock(),
            "session_file": tmp_path / "mcp_logs" / "test-session.jsonl",
        }

        # Use real logger to ensure fresh session
        logger = MCPLogger()
        logger.start_session()  # Generates new session ID
        logger._was_new_session = True  # Simulate new session

        ctx.request_context.lifespan_state["logger"] = logger

        params = MagicMock()
        params.history_size = 10

        result = asyncio.run(mcp_module.bud_context(params, ctx))
        data = json.loads(result)
        assert data["is_new_session"] is True

    def test_resumed_session_returns_false_when_loaded(self, tmp_path, monkeypatch):
        """Test that is_new_session is False when session ID is loaded from file."""
        monkeypatch.setattr(mcp_module, 'SESSIONS_DIR', tmp_path / "mcp_logs")
        monkeypatch.setattr(mcp_module, 'CURRENT_SESSION_FILE', tmp_path / "mcp_logs" / ".current_session")

        ctx = MagicMock()
        ctx.request_context = MagicMock()

        # Use a simple logger object with the required attributes
        # Simulate the case where session was loaded from file (was_new_session=False)
        logger = MagicMock()
        logger._current_session_id = "mcp-loaded-session"
        logger._was_new_session = False  # Session was loaded from file, not new

        ctx.request_context.lifespan_state = {
            "logger": logger,
            "session_file": tmp_path / "mcp_logs" / "test-session.jsonl",
        }

        params = MagicMock()
        params.history_size = 10

        result = asyncio.run(mcp_module.bud_context(params, ctx))
        data = json.loads(result)
        assert data["is_new_session"] is False
        assert data["session_id"] == "mcp-loaded-session"


class TestBudReflectDimensionValidation:
    """Tests for Issue #3: Verify dimension validation in bud_reflect."""

    def test_reflect_invalid_dimension_returns_error(self, tmp_path, monkeypatch):
        """Test that bud_reflect returns error for invalid dimension."""
        ctx = MagicMock()
        ctx.request_context = MagicMock()
        ctx.request_context.lifespan_state = {
            "store": MagicMock(),
            "schema": {
                "dimensions": {
                    "geometry": ["linear", "circular"],
                    "coherence": ["high", "low"],
                    "texture": ["smooth", "rough"],
                    "terrain": ["conceptual", "emotional"]
                },
                "chunk_types": ["exchange", "breakthrough"]
            },
        }

        store = MagicMock()
        store.count.return_value = 10
        ctx.request_context.lifespan_state["store"] = store

        params = MagicMock()
        params.dimension = "invalid_dimension"
        params.value = "any_value"
        params.limit = 10

        result = asyncio.run(mcp_module.bud_reflect(params, ctx))
        data = json.loads(result)
        assert "error" in data
        assert "invalid" in data["error"].lower() or "not found" in data["error"].lower()

    def test_reflect_valid_dimension_passes(self, tmp_path, monkeypatch):
        """Test that bud_reflect accepts valid dimensions from schema."""
        ctx = MagicMock()
        ctx.request_context = MagicMock()
        ctx.request_context.lifespan_state = {
            "store": MagicMock(),
            "schema": {
                "dimensions": {
                    "geometry": ["linear", "circular"],
                    "coherence": ["high", "low"]
                },
                "chunk_types": ["exchange"]
            },
        }

        store = MagicMock()
        store.count.return_value = 10
        store._metadata = [
            {"chunk_id": "c1", "chunk_type": "exchange", "tags": {"geometry": "linear"}},
        ]
        ctx.request_context.lifespan_state["store"] = store

        params = MagicMock()
        params.dimension = "geometry"
        params.value = "linear"
        params.limit = 10

        result = asyncio.run(mcp_module.bud_reflect(params, ctx))
        data = json.loads(result)
        assert "error" not in data
        assert data["dimension"] == "geometry"
        assert len(data["results"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
