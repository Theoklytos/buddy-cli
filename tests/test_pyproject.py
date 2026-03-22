"""Tests for pyproject.toml configuration."""

import re
from pathlib import Path

import pytest

PYPROJECT_PATH = Path(__file__).parent.parent / "pyproject.toml"


def test_mcp_dependency_in_pyproject():
    """Test that pyproject.toml has mcp>=1.1.0 in dependencies."""
    content = PYPROJECT_PATH.read_text()

    # Check that mcp>=1.1.0 is in the dependencies list
    assert 'mcp>=1.1.0' in content, (
        "pyproject.toml must include 'mcp>=1.1.0' in dependencies"
    )


def test_bud_mcp_script_entry_point():
    """Test that pyproject.toml has bud-mcp script entry point."""
    content = PYPROJECT_PATH.read_text()

    # Check that bud-mcp = "bud.bud_mcp:main" is in [project.scripts]
    assert 'bud-mcp = "bud.bud_mcp:main"' in content, (
        "pyproject.toml must include 'bud-mcp = \"bud.bud_mcp:main\"' in [project.scripts]"
    )


def test_mcp_dependency_in_bash():
    """Verify mcp>=1.1.0 is present using grep-style check."""
    import subprocess
    result = subprocess.run(
        ["grep", "-q", "mcp>=1.1.0", str(PYPROJECT_PATH)],
        capture_output=True
    )
    assert result.returncode == 0, "mcp>=1.1.0 should be in pyproject.toml"


def test_script_entry_point_format():
    """Test that script entry point uses correct format."""
    content = PYPROJECT_PATH.read_text()

    # Find the [project.scripts] section and verify bud-mcp entry
    scripts_section = re.search(r'\[project\.scripts\](.*?)(?=\[|\Z)', content, re.DOTALL)
    assert scripts_section, "project.scripts section must exist"

    scripts_content = scripts_section.group(1)
    assert 'bud-mcp' in scripts_content, "bud-mcp script must be defined"
    assert 'bud.bud_mcp:main' in scripts_content, "bud-mcp must point to bud.bud_mcp:main"
