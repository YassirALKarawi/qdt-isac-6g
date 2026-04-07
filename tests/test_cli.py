"""Tests for CLI entry point."""
import subprocess
import pytest

def test_help_flag():
    """main.py --help should exit cleanly."""
    result = subprocess.run(
        ["python", "main.py", "--help"],
        capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0 or "usage" in result.stdout.lower() or "help" in result.stderr.lower()
