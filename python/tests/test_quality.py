"""Tests for code quality (linting and formatting)."""

import subprocess
import sys


def test_ruff_check():
    """Test that ruff check passes."""
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "src/greybox/"],
        capture_output=True,
        text=True,
        cwd=".",
    )
    assert result.returncode == 0, (
        f"ruff check failed:\n{result.stdout}\n{result.stderr}"
    )


def test_ruff_format():
    """Test that ruff format check passes."""
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--check", "src/greybox/"],
        capture_output=True,
        text=True,
        cwd=".",
    )
    assert result.returncode == 0, (
        f"ruff format check failed:\n{result.stdout}\n{result.stderr}"
    )
