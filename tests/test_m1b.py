"""Tests for Module M1b: Real trace collection harness (data/collect_traces.py).
Implements PRD section 6 (M1b), Section 0 (Rule 3).

These tests verify:
1. --dry-run makes zero network calls (no sockets opened, no API clients instantiated).
2. Dry-run prints correct call counts: 39 calls/task, 27 paths/task.
3. Confirmation guard prevents live collection without explicit 'CONFIRM' input.
"""
import io
import subprocess
import sys
import os
import unittest.mock as mock
import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.abspath("."))


def test_m1b_dry_run_no_network_calls(tmp_path, monkeypatch):
    """PRD M1b / Rule 3: --dry-run must make zero network calls.

    Verifies that collect_traces with dry_run=True returns immediately after
    printing the summary without touching any socket or API client.
    Implements PRD Section 6 (M1b) and Section 0 (Rule 3).
    """
    import socket
    import urllib.request

    # Patch socket.create_connection to detect any network attempt
    original_create = socket.create_connection

    def fail_if_network(address, *args, **kwargs):
        raise AssertionError(
            f"--dry-run must not make network calls; attempted connection to {address}"
        )

    monkeypatch.setattr(socket, "create_connection", fail_if_network)

    # Also patch urllib in case requests library is used
    def fail_urlopen(url, *args, **kwargs):
        raise AssertionError(f"--dry-run must not call urlopen; attempted: {url}")

    monkeypatch.setattr(urllib.request, "urlopen", fail_urlopen)

    # Import and run dry_run_summary directly
    from data.collect_traces import dry_run_summary
    import yaml

    with open("configs/real_models.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    models = config["models"]

    captured = io.StringIO()
    import sys as _sys
    old_stdout = _sys.stdout
    _sys.stdout = captured
    try:
        dry_run_summary(n_tasks=5, models_config=models)
    finally:
        _sys.stdout = old_stdout

    output = captured.getvalue()
    # Network guard: if we got here without the assertion error, no calls were made
    assert "DRY RUN SUMMARY" in output
    assert "No live API calls were made" in output


def test_m1b_dry_run_call_counts():
    """PRD M1b: dry-run prints correct call counts per PRD Section 5.1.

    39 calls/task = 3 (hop 0) + 9 (hop 1) + 27 (hop 2).
    27 paths per task.
    Implements PRD Section 6 (M1b).
    """
    from data.collect_traces import dry_run_summary
    import yaml

    with open("configs/real_models.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    models = config["models"]

    captured = io.StringIO()
    import sys as _sys
    old_stdout = _sys.stdout
    _sys.stdout = captured
    try:
        dry_run_summary(n_tasks=5, models_config=models)
        dry_run_summary(n_tasks=60, models_config=models)
    finally:
        _sys.stdout = old_stdout

    output = captured.getvalue()
    # 5 tasks: 5 * 39 = 195 calls
    assert "195 calls" in output, f"Expected 195 calls for 5 tasks, got: {output}"
    # 60 tasks: 60 * 39 = 2340 calls
    assert "2340 calls" in output, f"Expected 2340 calls for 60 tasks, got: {output}"
    # Full paths per task always 27
    assert "27 paths" in output


def test_m1b_dry_run_via_subprocess():
    """PRD M1b: --dry-run flag via subprocess makes no API calls and exits 0."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.abspath(".")
    result = subprocess.run(
        [sys.executable, "data/collect_traces.py", "--dry-run", "--n-tasks", "5"],
        capture_output=True,
        text=True,
        cwd=os.path.abspath("."),
        env=env,
        timeout=30,
    )
    assert result.returncode == 0, f"--dry-run exited non-zero: {result.stderr}"
    assert "DRY RUN SUMMARY" in result.stdout
    assert "195 calls" in result.stdout
    assert "No live API calls were made" in result.stdout
    # Confirm no error output suggesting network activity
    assert "ConnectionError" not in result.stderr
    assert "APIError" not in result.stderr


def test_m1b_live_collection_requires_confirm(monkeypatch, tmp_path):
    """PRD Rule 3: Live collection must be aborted without explicit 'CONFIRM' input.

    Passing anything other than 'CONFIRM' should abort without calling any API.
    Implements PRD Section 6 (M1b) and Section 0 (Rule 3).
    """
    from data.collect_traces import collect_traces

    # Simulate user typing 'no' at the confirmation prompt
    monkeypatch.setattr("builtins.input", lambda _: "no")

    output_path = str(tmp_path / "real.json")
    captured = io.StringIO()
    import sys as _sys
    old_stdout = _sys.stdout
    _sys.stdout = captured
    try:
        collect_traces(
            tasks_source="agentbench_dbbench",
            output_path=output_path,
            models_config_path="configs/real_models.yaml",
            n_tasks=1,
            dry_run=False,
            resume=False,
        )
    finally:
        _sys.stdout = old_stdout

    output = captured.getvalue()
    assert "aborted" in output.lower(), (
        f"Expected abort message when user does not CONFIRM, got: {output}"
    )
    # No real output file should be created
    assert not os.path.exists(output_path), "real.json must not be created on abort"


# Implementation Note:
# Tests cover --dry-run network isolation (Rule 3), correct call counts (PRD 5.1),
# subprocess exit code, and confirmation guard for live collection.
# There is no dedicated test_m1b.py referenced in the original PRD build order;
# these tests supplement test_m9.py which only checks file existence.
