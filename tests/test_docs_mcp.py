"""Tests for docs/mcp.md documentation completeness."""

import os
from pathlib import Path


def test_docs_mcp_exists():
    """Test that docs/mcp.md exists."""
    docs_mcp = Path("/root/buddy-cli/docs/mcp.md")
    assert docs_mcp.exists(), "docs/mcp.md should exist"
    assert docs_mcp.is_file(), "docs/mcp.md should be a file"


def test_docs_mcp_has_overview_section():
    """Test that docs/mcp.md contains an Overview section."""
    content = Path("/root/buddy-cli/docs/mcp.md").read_text()
    assert "# Overview" in content or "## Overview" in content, "docs/mcp.md should have an Overview section"


def test_docs_mcp_has_bud_context_tool():
    """Test that docs/mcp.md documents the bud_context tool."""
    content = Path("/root/buddy-cli/docs/mcp.md").read_text()
    assert "bud_context" in content, "docs/mcp.md should document bud_context tool"


def test_docs_mcp_has_bud_recall_tool():
    """Test that docs/mcp.md documents the bud_recall tool."""
    content = Path("/root/buddy-cli/docs/mcp.md").read_text()
    assert "bud_recall" in content, "docs/mcp.md should document bud_recall tool"


def test_docs_mcp_has_bud_orient_tool():
    """Test that docs/mcp.md documents the bud_orient tool."""
    content = Path("/root/buddy-cli/docs/mcp.md").read_text()
    assert "bud_orient" in content, "docs/mcp.md should document bud_orient tool"


def test_docs_mcp_has_bud_reflect_tool():
    """Test that docs/mcp.md documents the bud_reflect tool."""
    content = Path("/root/buddy-cli/docs/mcp.md").read_text()
    assert "bud_reflect" in content, "docs/mcp.md should document bud_reflect tool"


def test_docs_mcp_has_logging_section():
    """Test that docs/mcp.md has a Logging section."""
    content = Path("/root/buddy-cli/docs/mcp.md").read_text()
    assert ("# Logging" in content or "## Logging" in content or "### Logging" in content), \
        "docs/mcp.md should have a Logging section"


def test_docs_mcp_has_log_location():
    """Test that docs/mcp.md mentions the log location ~/mcp_logs/."""
    content = Path("/root/buddy-cli/docs/mcp.md").read_text()
    assert "mcp_logs" in content, "docs/mcp.md should mention ~/mcp_logs/ directory"


def test_docs_mcp_has_jsonl_format():
    """Test that docs/mcp.md mentions JSONL format."""
    content = Path("/root/buddy-cli/docs/mcp.md").read_text()
    assert "jsonl" in content.lower() or "JSONL" in content, "docs/mcp.md should mention JSONL format"


def test_docs_mcp_has_log_entry_structure():
    """Test that docs/mcp.md documents log entry structure."""
    content = Path("/root/buddy-cli/docs/mcp.md").read_text()
    # Should have JSON example with required fields
    assert "session_id" in content, "docs/mcp.md should document session_id field"
    assert "tool_name" in content, "docs/mcp.md should document tool_name field"


def test_docs_mcp_has_configuration_section():
    """Test that docs/mcp.md has a Configuration section."""
    content = Path("/root/buddy-cli/docs/mcp.md").read_text()
    assert ("# Configuration" in content or "## Configuration" in content or "### Configuration" in content), \
        "docs/mcp.md should have a Configuration section"


def test_docs_mcp_has_claude_desktop_config():
    """Test that docs/mcp.md documents Claude Desktop configuration."""
    content = Path("/root/buddy-cli/docs/mcp.md").read_text()
    assert "claude" in content.lower(), "docs/mcp.md should mention Claude Desktop configuration"


def test_docs_mcp_has_cli_usage():
    """Test that docs/mcp.md documents CLI usage."""
    content = Path("/root/buddy-cli/docs/mcp.md").read_text()
    assert ("bud mcp" in content or "`bud mcp`" in content or "bud mcp" in content), \
        "docs/mcp.md should document CLI usage"


def test_docs_mcp_has_manual_invocation():
    """Test that docs/mcp.md documents manual invocation."""
    content = Path("/root/buddy-cli/docs/mcp.md").read_text()
    assert ("manual" in content.lower() or "python" in content.lower()), \
        "docs/mcp.md should document manual invocation"


def test_docs_mcp_has_requirements_section():
    """Test that docs/mcp.md has a Requirements section."""
    content = Path("/root/buddy-cli/docs/mcp.md").read_text()
    assert ("# Requirements" in content or "## Requirements" in content or "### Requirements" in content), \
        "docs/mcp.md should have a Requirements section"


def test_docs_mcp_has_usage_notes_section():
    """Test that docs/mcp.md has a Usage Notes section."""
    content = Path("/root/buddy-cli/docs/mcp.md").read_text()
    assert ("# Usage" in content or "## Usage" in content or "### Usage" in content or "Notes" in content), \
        "docs/mcp.md should have a Usage/Notes section"


def test_docs_mcp_has_session_context_for_claude():
    """Test that docs/mcp.md explains session context for Claude."""
    content = Path("/root/buddy-cli/docs/mcp.md").read_text()
    assert "session" in content.lower(), "docs/mcp.md should explain session context"
    assert "claude" in content.lower(), "docs/mcp.md should explain Claude integration"


def test_docs_mcp_has_read_only_note():
    """Test that docs/mcp.md mentions tools are read-only."""
    content = Path("/root/buddy-cli/docs/mcp.md").read_text()
    assert ("read-only" in content.lower() or "readonly" in content.lower() or "read only" in content.lower()), \
        "docs/mcp.md should mention tools are read-only"


def test_docs_mcp_has_session_id_tracking():
    """Test that docs/mcp.md documents session ID tracking for Claude."""
    content = Path("/root/buddy-cli/docs/mcp.md").read_text()
    assert "session_id" in content, "docs/mcp.md should document session_id tracking"


def test_docs_mcp_has_history_availability():
    """Test that docs/mcp.md mentions history availability in bud_context."""
    content = Path("/root/buddy-cli/docs/mcp.md").read_text()
    assert ("history" in content.lower() or "tool call" in content.lower()), \
        "docs/mcp.md should mention history availability"


def test_docs_mcp_has_duplicate_prevention():
    """Test that docs/mcp.md explains duplicate prevention using history."""
    content = Path("/root/buddy-cli/docs/mcp.md").read_text()
    # Should mention avoiding duplicate tool calls
    assert ("duplicate" in content.lower() or "prevent" in content.lower() or "avoid" in content.lower()), \
        "docs/mcp.md should explain how to avoid duplicate tool calls"


def test_docs_mcp_has_session_persistence():
    """Test that docs/mcp.md mentions session persistence across restarts."""
    content = Path("/root/buddy-cli/docs/mcp.md").read_text()
    assert ("persist" in content.lower() or "restart" in content.lower() or "resume" in content.lower()), \
        "docs/mcp.md should mention session persistence"


def test_docs_mcp_has_json_code_blocks():
    """Test that docs/mcp.md contains JSON code blocks."""
    content = Path("/root/buddy-cli/docs/mcp.md").read_text()
    assert "```json" in content or "```" in content, "docs/mcp.md should contain JSON code blocks"


def test_docs_mcp_helpful_tips_for_claude():
    """Test that docs/mcp.md includes helpful tips for Claude integration."""
    content = Path("/root/buddy-cli/docs/mcp.md").read_text()
    # Should contain tips or best practices for Claude
    assert ("tip" in content.lower() or "use" in content.lower() or "best practice" in content.lower() or "example" in content.lower()), \
        "docs/mcp.md should include helpful tips for Claude"


def test_docs_mcp_all_sections_present():
    """Comprehensive test that all required sections are present."""
    content = Path("/root/buddy-cli/docs/mcp.md").read_text().lower()

    required_sections = [
        ("overview", "Overview section"),
        ("bud_context", "bud_context tool"),
        ("bud_recall", "bud_recall tool"),
        ("bud_orient", "bud_orient tool"),
        ("bud_reflect", "bud_reflect tool"),
        ("logging", "Logging section"),
        ("mcp_logs", "Log location"),
        ("jsonl", "JSONL format"),
        ("configuration", "Configuration section"),
        ("claude", "Claude Desktop"),
        ("cli", "CLI usage"),
        ("manual", "Manual invocation"),
        ("requirement", "Requirements section"),
        ("usage", "Usage notes"),
        ("session", "Session context for Claude"),
        ("read-only", "Read-only note"),
    ]

    missing = []
    for keyword, description in required_sections:
        if keyword not in content:
            missing.append(description)

    assert not missing, f"Missing sections: {', '.join(missing)}"
