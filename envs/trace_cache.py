"""TraceCache module for loading and accessing cached LLM traces.
Implements PRD section 6 (M1) and Section 5.1.
"""
import json
import os
from typing import Any


class TraceCache:
    """Read-only cache of pipeline calls and outcomes loaded from JSON.

    Implements PRD Section 6 (M1) and Section 5.1.
    """

    def __init__(self, json_path: str):
        """Load the trace cache from a JSON file.

        Implements PRD Section 6 (M1).
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Trace cache file not found at: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.metadata: dict[str, Any] = data.get("metadata", {})
        self._tasks: dict[str, Any] = data.get("tasks", {})

    def call(self, task_id: str, hop: int, prefix: tuple[str, ...]) -> dict[str, Any]:
        """Look up the call record for a given hop and prefix sequence.

        Implements PRD Section 6 (M1).
        Prefix should be a tuple of model choices up to this hop, e.g. ('m0',) at hop 0,
        ('m0', 'm2') at hop 1, ('m0', 'm2', 'm1') at hop 2.
        Key format: '{hop}_{prefix}'.
        """
        if task_id not in self._tasks:
            raise KeyError(f"Task '{task_id}' not found in trace cache")

        task = self._tasks[task_id]
        calls = task.get("calls", {})

        call_key = f"{hop}_{'_'.join(prefix)}"
        if call_key not in calls:
            raise KeyError(f"Call record '{call_key}' not found for task '{task_id}'")

        return calls[call_key]

    def outcome(self, task_id: str, path: tuple[str, str, str]) -> float:
        """Look up the final outcome (success score in [0, 1]) for a full 3-hop path.

        Implements PRD Section 6 (M1).
        Path key format: 'm0_m1_m2'.
        """
        if task_id not in self._tasks:
            raise KeyError(f"Task '{task_id}' not found in trace cache")

        task = self._tasks[task_id]
        outcomes = task.get("outcomes", {})

        path_key = "_".join(path)
        if path_key not in outcomes:
            raise KeyError(f"Outcome for path '{path_key}' not found for task '{task_id}'")

        return float(outcomes[path_key])

    def task_ids(self) -> list[str]:
        """Return the list of task IDs in the cache.

        Implements PRD Section 6 (M1).
        """
        return list(self._tasks.keys())


# Implementation Note:
# Implemented exactly as specified in PRD Section 6 (M1) and Section 5.1.
# Read-only interface with explicit error checking for missing tasks and path keys.
