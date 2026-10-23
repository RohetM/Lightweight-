"""Tests for Module M9: Interactive demo application.
Implements PRD section 6 (M9) and Section 0 (Rule 5).
"""
import os
import pytest
import pandas as pd


def test_m9_demo_files_exist():
    """Verify demo files and artifacts are created, including >100KB demo video."""
    assert os.path.exists("demo/app.py")
    assert os.path.exists("demo/record_demo.py")
    assert os.path.exists("demo/working_model_demo.webm")
    video_size_kb = os.path.getsize("demo/working_model_demo.webm") / 1024.0
    assert video_size_kb > 100.0, f"Expected video > 100 KB, got {video_size_kb:.2f} KB"


def test_m9_synthetic_banner_in_app():
    """PRD Rule 5 & M9: SYNTHETIC DATA banner must be present in demo/app.py."""
    with open("demo/app.py", "r", encoding="utf-8") as f:
        content = f.read()

    assert "SYNTHETIC DATA" in content
    assert "Run Live Multi-Agent Routing Simulation" in content
