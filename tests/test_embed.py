"""Tests for bud.stages.embed."""

import json
import os
import unittest
from unittest.mock import MagicMock, patch, call

import pytest

from bud.lib.errors import EmbeddingError
from bud.lib.embeddings import EmbeddingClient
from bud.stages.embed import (
    embed_chunks,
    load_embed_queue,
    write_embed_queue,
    clear_embed_queue,
)


def _make_chunk(chunk_id: str, text: str = "some text") -> dict:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "conversation_id": "conv-1",
        "source_file": "file.json",
    }


def _make_embedding_client(vector=None, dimension=4):
    client = MagicMock()
    client.embed.return_value = vector or [0.1, 0.2, 0.3, 0.4]
    client.dimension = dimension
    return client


def _make_store(existing_ids=None):
    store = MagicMock()
    existing = set(existing_ids or [])
    store.chunk_id_exists.side_effect = lambda cid: cid in existing
    store.dimension = 4
    store._index_path = "/fake/path.faiss"
    store._meta_path = "/fake/path_metadata.jsonl"
    return store


def test_embed_chunks_calls_embed_for_each_chunk(tmp_path):
    chunks = [_make_chunk("a"), _make_chunk("b")]
    client = _make_embedding_client()
    store = _make_store()
    queue = str(tmp_path / "queue.jsonl")

    embed_chunks(chunks, client, store, queue)

    assert client.embed.call_count == 2
    assert store.add.call_count == 2


def test_embed_chunks_stores_text_in_metadata(tmp_path):
    """Text must be included in stored metadata so queries can retrieve it."""
    chunks = [_make_chunk("a", text="hello world")]
    client = _make_embedding_client()
    store = _make_store()
    queue = str(tmp_path / "queue.jsonl")

    embed_chunks(chunks, client, store, queue)

    stored_metadata = store.add.call_args[0][1][0]
    assert stored_metadata.get("text") == "hello world"


def test_embed_chunks_skips_existing_chunk_ids(tmp_path):
    chunks = [_make_chunk("existing"), _make_chunk("new")]
    client = _make_embedding_client()
    store = _make_store(existing_ids=["existing"])
    queue = str(tmp_path / "queue.jsonl")

    embed_chunks(chunks, client, store, queue)

    assert client.embed.call_count == 1


def test_failed_chunks_written_to_queue(tmp_path):
    chunks = [_make_chunk("a")]
    client = _make_embedding_client()
    client.embed.side_effect = EmbeddingError("API failure")
    store = _make_store()
    queue = str(tmp_path / "queue.jsonl")

    failures = embed_chunks(chunks, client, store, queue)

    assert failures == 1
    queued = load_embed_queue(queue)
    assert len(queued) == 1
    assert queued[0]["chunk_id"] == "a"


def test_embed_chunks_returns_zero_on_success(tmp_path):
    chunks = [_make_chunk("a")]
    client = _make_embedding_client()
    store = _make_store()
    queue = str(tmp_path / "queue.jsonl")

    failures = embed_chunks(chunks, client, store, queue)
    assert failures == 0


def test_load_embed_queue_returns_empty_when_no_file(tmp_path):
    result = load_embed_queue(str(tmp_path / "nonexistent.jsonl"))
    assert result == []


def test_write_and_load_embed_queue(tmp_path):
    path = str(tmp_path / "queue.jsonl")
    chunks = [_make_chunk("x"), _make_chunk("y")]
    write_embed_queue(path, chunks)
    loaded = load_embed_queue(path)
    assert len(loaded) == 2
    assert loaded[0]["chunk_id"] == "x"
    assert loaded[1]["chunk_id"] == "y"


def test_write_embed_queue_append(tmp_path):
    path = str(tmp_path / "queue.jsonl")
    write_embed_queue(path, [_make_chunk("a")])
    write_embed_queue(path, [_make_chunk("b")], append=True)
    loaded = load_embed_queue(path)
    assert len(loaded) == 2


def test_clear_embed_queue_removes_file(tmp_path):
    path = str(tmp_path / "queue.jsonl")
    write_embed_queue(path, [_make_chunk("a")])
    clear_embed_queue(path)
    assert not (tmp_path / "queue.jsonl").exists()


def test_clear_embed_queue_noop_when_no_file(tmp_path):
    clear_embed_queue(str(tmp_path / "nofile.jsonl"))  # should not raise


# ---------------------------------------------------------------------------
# on_chunk callback tests
# ---------------------------------------------------------------------------


def test_on_chunk_called_for_each_chunk(tmp_path):
    client = _make_embedding_client()
    store = _make_store()
    chunks = [_make_chunk(f"c{i}") for i in range(3)]
    calls = []
    embed_chunks(chunks, client, store, str(tmp_path / "q.jsonl"), on_chunk=lambda d, t: calls.append((d, t)))
    assert len(calls) == 3
    assert calls[0] == (1, 3)
    assert calls[2] == (3, 3)


def test_on_chunk_called_for_skipped_chunks(tmp_path):
    """on_chunk fires even when a chunk already exists in the store."""
    client = _make_embedding_client()
    store = _make_store(existing_ids=["dup"])
    chunks = [_make_chunk("dup")]
    calls = []
    embed_chunks(chunks, client, store, str(tmp_path / "q.jsonl"), on_chunk=lambda d, t: calls.append((d, t)))
    assert len(calls) == 1
    assert calls[0] == (1, 1)


def test_on_chunk_called_on_failure(tmp_path):
    """on_chunk fires even when embedding raises EmbeddingError."""
    client = _make_embedding_client()
    client.embed.side_effect = EmbeddingError("boom")
    store = _make_store()
    chunks = [_make_chunk("fail")]
    calls = []
    embed_chunks(chunks, client, store, str(tmp_path / "q.jsonl"), on_chunk=lambda d, t: calls.append((d, t)))
    assert len(calls) == 1


def test_on_chunk_none_does_not_raise(tmp_path):
    client = _make_embedding_client()
    store = _make_store()
    chunks = [_make_chunk("ok")]
    # no on_chunk — should complete without error
    result = embed_chunks(chunks, client, store, str(tmp_path / "q.jsonl"))
    assert result == 0


# ---------------------------------------------------------------------------
# on_error callback tests
# ---------------------------------------------------------------------------


def test_on_error_called_on_embedding_failure(tmp_path):
    client = _make_embedding_client()
    client.embed.side_effect = EmbeddingError("connection refused")
    store = _make_store()
    chunks = [_make_chunk("bad")]
    errors = []
    embed_chunks(
        chunks, client, store, str(tmp_path / "q.jsonl"),
        on_error=lambda chunk, msg: errors.append(msg),
    )
    assert len(errors) == 1
    assert "connection refused" in errors[0]


def test_on_error_receives_failing_chunk(tmp_path):
    client = _make_embedding_client()
    client.embed.side_effect = EmbeddingError("boom")
    store = _make_store()
    chunks = [_make_chunk("fail-me")]
    received = []
    embed_chunks(
        chunks, client, store, str(tmp_path / "q.jsonl"),
        on_error=lambda chunk, msg: received.append(chunk["chunk_id"]),
    )
    assert received == ["fail-me"]


def test_on_error_not_called_on_success(tmp_path):
    client = _make_embedding_client()
    store = _make_store()
    chunks = [_make_chunk("ok")]
    errors = []
    embed_chunks(
        chunks, client, store, str(tmp_path / "q.jsonl"),
        on_error=lambda chunk, msg: errors.append(msg),
    )
    assert errors == []


def test_on_error_none_does_not_raise_on_failure(tmp_path):
    client = _make_embedding_client()
    client.embed.side_effect = EmbeddingError("boom")
    store = _make_store()
    chunks = [_make_chunk("fail")]
    # no on_error — should still queue the failure without crashing
    result = embed_chunks(chunks, client, store, str(tmp_path / "q.jsonl"))
    assert result == 1


def test_on_error_called_for_each_failing_chunk(tmp_path):
    client = _make_embedding_client()
    client.embed.side_effect = EmbeddingError("nope")
    store = _make_store()
    chunks = [_make_chunk(f"bad-{i}") for i in range(4)]
    errors = []
    embed_chunks(
        chunks, client, store, str(tmp_path / "q.jsonl"),
        on_error=lambda chunk, msg: errors.append(chunk["chunk_id"]),
    )
    assert len(errors) == 4


# ---------------------------------------------------------------------------
# EmbeddingClient tests
# ---------------------------------------------------------------------------


def test_embedding_client_tries_new_api_first(tmp_path):
    """Client should POST to /api/embed before /api/embeddings."""
    from unittest.mock import patch as _patch
    import requests

    config = {
        "embeddings": {"provider": "ollama", "base_url": "http://localhost:11434", "model": "test"},
        "llm": {"timeout_seconds": 10},
    }
    from bud.lib.embeddings import EmbeddingClient

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}

    with _patch("requests.post", return_value=mock_resp) as mock_post:
        client = EmbeddingClient(config)
        result = client.embed("hello")

    first_call_url = mock_post.call_args_list[0][0][0]
    assert "/api/embed" in first_call_url
    assert result == [0.1, 0.2, 0.3]


def test_embedding_client_falls_back_to_legacy_api(tmp_path):
    """Client falls back to /api/embeddings when /api/embed returns non-200."""
    from unittest.mock import patch as _patch, call as _call

    config = {
        "embeddings": {"provider": "ollama", "base_url": "http://localhost:11434", "model": "test"},
        "llm": {"timeout_seconds": 10},
    }
    from bud.lib.embeddings import EmbeddingClient

    new_resp = MagicMock()
    new_resp.status_code = 404

    legacy_resp = MagicMock()
    legacy_resp.status_code = 200
    legacy_resp.json.return_value = {"embedding": [0.4, 0.5, 0.6]}

    with _patch("requests.post", side_effect=[new_resp, legacy_resp]) as mock_post:
        client = EmbeddingClient(config)
        result = client.embed("hello")

    assert mock_post.call_count == 2
    assert "/api/embed" in mock_post.call_args_list[0][0][0]
    assert "/api/embeddings" in mock_post.call_args_list[1][0][0]
    assert result == [0.4, 0.5, 0.6]


def test_embedding_client_raises_on_both_api_failures():
    """EmbeddingError raised when both endpoints fail."""
    from unittest.mock import patch as _patch
    from bud.lib.embeddings import EmbeddingClient
    from bud.lib.errors import EmbeddingError

    config = {
        "embeddings": {"provider": "ollama", "base_url": "http://localhost:11434", "model": "test"},
        "llm": {"timeout_seconds": 10},
    }

    bad_resp = MagicMock()
    bad_resp.status_code = 500
    bad_resp.text = "internal error"

    with _patch("requests.post", return_value=bad_resp):
        client = EmbeddingClient(config)
        with pytest.raises(EmbeddingError):
            client.embed("hello")


def test_embedding_client_sets_dimension_after_first_embed():
    """dimension property is None before first call, set after."""
    from unittest.mock import patch as _patch
    from bud.lib.embeddings import EmbeddingClient

    config = {
        "embeddings": {"provider": "ollama", "base_url": "http://localhost:11434", "model": "test"},
        "llm": {"timeout_seconds": 10},
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"embeddings": [[0.1, 0.2, 0.3, 0.4]]}

    client = EmbeddingClient(config)
    assert client.dimension is None

    with _patch("requests.post", return_value=mock_resp):
        client.embed("test")

    assert client.dimension == 4


# ---------------------------------------------------------------------------
# _resolve_api_key tests
# ---------------------------------------------------------------------------


class TestResolveApiKey(unittest.TestCase):
    """Tests for _resolve_api_key() — config > env var > error."""

    def _make_config(self, **embed_overrides):
        base = {"provider": "voyage", "model": "voyage-3-large", "base_url": "https://api.voyageai.com"}
        base.update(embed_overrides)
        return {"embeddings": base, "llm": {"timeout_seconds": 10}}

    def test_returns_key_from_config(self):
        client = EmbeddingClient(self._make_config(api_key="cfg-key-123"))
        assert client._resolve_api_key("voyage") == "cfg-key-123"

    def test_falls_back_to_env_var_voyage(self):
        client = EmbeddingClient(self._make_config())
        with patch.dict(os.environ, {"VOYAGE_API_KEY": "env-key-456"}):
            assert client._resolve_api_key("voyage") == "env-key-456"

    def test_falls_back_to_env_var_openai(self):
        client = EmbeddingClient(self._make_config(provider="openai", model="text-embedding-3-small"))
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key-789"}):
            assert client._resolve_api_key("openai") == "env-key-789"

    def test_raises_on_missing_key(self):
        client = EmbeddingClient(self._make_config())
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("VOYAGE_API_KEY", None)
            with self.assertRaises(ValueError) as ctx:
                client._resolve_api_key("voyage")
            assert "VOYAGE_API_KEY" in str(ctx.exception)

    def test_ignores_none_string_api_key(self):
        """The old bug: api_key defaulting to 'NONE' should not be treated as valid."""
        client = EmbeddingClient(self._make_config(api_key="NONE", provider="openai", model="text-embedding-3-small"))
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            with self.assertRaises(ValueError):
                client._resolve_api_key("openai")


# ---------------------------------------------------------------------------
# _embed_openai_compatible tests
# ---------------------------------------------------------------------------


import requests as _requests_module


class TestOpenAICompatibleErrors(unittest.TestCase):
    """Tests for 401/429/timeout handling in _embed_openai_compatible."""

    def _make_config(self, **embed_overrides):
        base = {"provider": "openai", "api_key": "test-key", "model": "text-embedding-3-small", "base_url": "https://api.openai.com"}
        base.update(embed_overrides)
        return {"embeddings": base, "llm": {"timeout_seconds": 10}}

    @patch("requests.post")
    def test_401_raises_auth_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Invalid API key"
        mock_post.return_value = mock_resp
        client = EmbeddingClient(self._make_config())
        with self.assertRaises(ValueError) as ctx:
            client._embed_openai_compatible("test text", "https://api.openai.com/v1/embeddings", "bad-key", "text-embedding-3-small")
        assert "401" in str(ctx.exception) or "authentication" in str(ctx.exception).lower()

    @patch("requests.post")
    def test_429_raises_rate_limit_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.text = "Rate limit exceeded"
        mock_post.return_value = mock_resp
        client = EmbeddingClient(self._make_config())
        with self.assertRaises(RuntimeError) as ctx:
            client._embed_openai_compatible("test text", "https://api.openai.com/v1/embeddings", "key", "text-embedding-3-small")
        assert "429" in str(ctx.exception) or "rate limit" in str(ctx.exception).lower()

    @patch("requests.post")
    def test_timeout_raises_connection_error(self, mock_post):
        mock_post.side_effect = _requests_module.exceptions.Timeout("Connection timed out")
        client = EmbeddingClient(self._make_config())
        with self.assertRaises(ConnectionError):
            client._embed_openai_compatible("test text", "https://api.openai.com/v1/embeddings", "key", "text-embedding-3-small")

    @patch("requests.post")
    def test_successful_embedding(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp
        client = EmbeddingClient(self._make_config())
        result = client._embed_openai_compatible("test", "https://api.openai.com/v1/embeddings", "key", "model")
        assert result == [0.1, 0.2, 0.3]
