"""Tests for buddy.core.config."""

import copy
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

import buddy.core.config as cfg


# ---------------------------------------------------------------------------
# _deep_merge
# ---------------------------------------------------------------------------


def test_deep_merge_simple():
    base = {"a": 1, "b": 2}
    override = {"b": 99, "c": 3}
    result = cfg._deep_merge(base, override)
    assert result == {"a": 1, "b": 99, "c": 3}


def test_deep_merge_nested():
    base = {"x": {"y": 1, "z": 2}}
    override = {"x": {"z": 99}}
    result = cfg._deep_merge(base, override)
    assert result == {"x": {"y": 1, "z": 99}}


def test_deep_merge_does_not_mutate_base():
    base = {"a": {"b": 1}}
    override = {"a": {"b": 2}}
    cfg._deep_merge(base, override)
    assert base == {"a": {"b": 1}}


def test_deep_merge_override_wins_for_non_dict():
    base = {"a": [1, 2]}
    override = {"a": [3, 4]}
    result = cfg._deep_merge(base, override)
    assert result["a"] == [3, 4]


# ---------------------------------------------------------------------------
# load_buddy_config
# ---------------------------------------------------------------------------


def test_load_buddy_config_returns_defaults_when_missing(tmp_path, monkeypatch):
    nonexistent = tmp_path / "buddy.yaml"
    monkeypatch.setattr(cfg, "BUDDY_CONFIG_FILE", nonexistent)
    result = cfg.load_buddy_config()
    assert result == cfg.DEFAULT_BUDDY_CONFIG


def test_load_buddy_config_merges_with_defaults(tmp_path, monkeypatch):
    config_file = tmp_path / "buddy.yaml"
    config_file.write_text(yaml.dump({"model": "my-custom-model"}))
    monkeypatch.setattr(cfg, "BUDDY_CONFIG_FILE", config_file)
    result = cfg.load_buddy_config()
    assert result["model"] == "my-custom-model"
    # Defaults still present
    assert result["max_tokens"] == cfg.DEFAULT_BUDDY_CONFIG["max_tokens"]


def test_load_buddy_config_returns_defaults_on_yaml_error(tmp_path, monkeypatch):
    config_file = tmp_path / "buddy.yaml"
    config_file.write_text(": bad: yaml: {{")
    monkeypatch.setattr(cfg, "BUDDY_CONFIG_FILE", config_file)
    result = cfg.load_buddy_config()
    assert result == cfg.DEFAULT_BUDDY_CONFIG


# ---------------------------------------------------------------------------
# save_buddy_config
# ---------------------------------------------------------------------------


def test_save_buddy_config_creates_file(tmp_path, monkeypatch):
    config_file = tmp_path / "buddy.yaml"
    monkeypatch.setattr(cfg, "BUDDY_CONFIG_FILE", config_file)
    monkeypatch.setattr(cfg, "CONFIG_DIR", tmp_path)
    data = copy.deepcopy(cfg.DEFAULT_BUDDY_CONFIG)
    path = cfg.save_buddy_config(data)
    assert path == config_file
    assert config_file.exists()


def test_save_and_reload_buddy_config(tmp_path, monkeypatch):
    config_file = tmp_path / "buddy.yaml"
    monkeypatch.setattr(cfg, "BUDDY_CONFIG_FILE", config_file)
    monkeypatch.setattr(cfg, "CONFIG_DIR", tmp_path)
    data = copy.deepcopy(cfg.DEFAULT_BUDDY_CONFIG)
    data["model"] = "my-test-model"
    cfg.save_buddy_config(data)
    result = cfg.load_buddy_config()
    assert result["model"] == "my-test-model"


def test_save_buddy_config_creates_config_dir(tmp_path, monkeypatch):
    new_dir = tmp_path / "new_config_dir"
    config_file = new_dir / "buddy.yaml"
    monkeypatch.setattr(cfg, "CONFIG_DIR", new_dir)
    monkeypatch.setattr(cfg, "BUDDY_CONFIG_FILE", config_file)
    assert not new_dir.exists()
    cfg.save_buddy_config(cfg.DEFAULT_BUDDY_CONFIG)
    assert new_dir.exists()
    assert config_file.exists()


# ---------------------------------------------------------------------------
# resolve_api_key
# ---------------------------------------------------------------------------


def test_resolve_api_key_from_buddy_config(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "BUD_CONFIG_FILE", tmp_path / "config.yaml")
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("ANTHROPIC_API_KEY", None)
        result = cfg.resolve_api_key({"api_key": "buddy-key"})
    assert result == "buddy-key"


def test_resolve_api_key_from_env(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "BUD_CONFIG_FILE", tmp_path / "config.yaml")
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-key"}, clear=False):
        result = cfg.resolve_api_key({"api_key": None})
    assert result == "env-key"


def test_resolve_api_key_buddy_config_beats_env(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "BUD_CONFIG_FILE", tmp_path / "config.yaml")
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-key"}, clear=False):
        result = cfg.resolve_api_key({"api_key": "buddy-key"})
    assert result == "buddy-key"


def test_resolve_api_key_from_bud_config(tmp_path, monkeypatch):
    bud_config_file = tmp_path / "config.yaml"
    bud_config_file.write_text(yaml.dump({"llm": {"api_key": "bud-key"}}))
    monkeypatch.setattr(cfg, "BUD_CONFIG_FILE", bud_config_file)
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("ANTHROPIC_API_KEY", None)
        result = cfg.resolve_api_key({"api_key": None})
    assert result == "bud-key"


def test_resolve_api_key_returns_none_when_all_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "BUD_CONFIG_FILE", tmp_path / "missing.yaml")
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("ANTHROPIC_API_KEY", None)
        result = cfg.resolve_api_key({"api_key": None})
    assert result is None


# ---------------------------------------------------------------------------
# list_buddy_profiles / load_buddy_profile / save_buddy_profile
# ---------------------------------------------------------------------------


def test_list_buddy_profiles_empty_when_dir_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "BUDDY_PROFILES_DIR", tmp_path / "no_profiles")
    result = cfg.list_buddy_profiles()
    assert result == []


def test_save_and_list_buddy_profiles(tmp_path, monkeypatch):
    profiles_dir = tmp_path / "buddy_profiles"
    monkeypatch.setattr(cfg, "BUDDY_PROFILES_DIR", profiles_dir)
    cfg.save_buddy_profile("alpha", {"model": "m1"})
    cfg.save_buddy_profile("beta", {"model": "m2"})
    names = cfg.list_buddy_profiles()
    assert names == ["alpha", "beta"]


def test_load_buddy_profile_returns_data(tmp_path, monkeypatch):
    profiles_dir = tmp_path / "buddy_profiles"
    monkeypatch.setattr(cfg, "BUDDY_PROFILES_DIR", profiles_dir)
    cfg.save_buddy_profile("myprofile", {"model": "special-model", "context_depth": 10})
    result = cfg.load_buddy_profile("myprofile")
    assert result["model"] == "special-model"
    assert result["context_depth"] == 10


def test_load_buddy_profile_raises_on_missing(tmp_path, monkeypatch):
    profiles_dir = tmp_path / "buddy_profiles"
    monkeypatch.setattr(cfg, "BUDDY_PROFILES_DIR", profiles_dir)
    with pytest.raises(FileNotFoundError):
        cfg.load_buddy_profile("nonexistent")


def test_save_buddy_profile_creates_dir(tmp_path, monkeypatch):
    profiles_dir = tmp_path / "new_profiles"
    monkeypatch.setattr(cfg, "BUDDY_PROFILES_DIR", profiles_dir)
    assert not profiles_dir.exists()
    path = cfg.save_buddy_profile("test", {"key": "val"})
    assert profiles_dir.exists()
    assert path.exists()
    assert path.name == "test.yaml"


def test_save_buddy_profile_returns_correct_path(tmp_path, monkeypatch):
    profiles_dir = tmp_path / "buddy_profiles"
    monkeypatch.setattr(cfg, "BUDDY_PROFILES_DIR", profiles_dir)
    path = cfg.save_buddy_profile("myprof", {"a": 1})
    assert path == profiles_dir / "myprof.yaml"


# ---------------------------------------------------------------------------
# run_config_wizard (light smoke test with mocked prompts)
# ---------------------------------------------------------------------------


def test_run_config_wizard_saves_config(tmp_path, monkeypatch):
    config_file = tmp_path / "buddy.yaml"
    monkeypatch.setattr(cfg, "BUDDY_CONFIG_FILE", config_file)
    monkeypatch.setattr(cfg, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cfg, "BUD_CONFIG_FILE", tmp_path / "no_bud_config.yaml")

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    inputs = iter([
        "sk-test-key",                   # API key
        "claude-sonnet-4-20250514",   # model
        "You are a helpful assistant.",  # system prompt
        "",                              # temperature (blank = None)
        "5",                             # context depth
    ])

    with patch("buddy.core.config.Prompt.ask", side_effect=lambda *a, **kw: next(inputs)):
        result = cfg.run_config_wizard()

    assert result["model"] == "claude-sonnet-4-20250514"
    assert result["temperature"] is None
    assert result["context_depth"] == 5
    assert config_file.exists()


def test_run_config_wizard_imports_bud_api_key(tmp_path, monkeypatch):
    config_file = tmp_path / "buddy.yaml"
    bud_config_file = tmp_path / "config.yaml"
    bud_config_file.write_text(yaml.dump({"llm": {"api_key": "from-bud"}}))

    monkeypatch.setattr(cfg, "BUDDY_CONFIG_FILE", config_file)
    monkeypatch.setattr(cfg, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cfg, "BUD_CONFIG_FILE", bud_config_file)

    inputs = iter([
        "claude-sonnet-4-20250514",
        "You are a helpful assistant.",
        "",
        "5",
    ])

    with patch("buddy.core.config.Prompt.ask", side_effect=lambda *a, **kw: next(inputs)):
        result = cfg.run_config_wizard()

    assert result["api_key"] == "from-bud"
