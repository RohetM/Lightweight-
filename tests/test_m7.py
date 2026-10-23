"""Tests for Module M7: Async credit worker.
Implements PRD section 6 (M7) and Section 2 (S4).
"""
import os
import sys
import time
import json
import subprocess
import tempfile
import yaml
import pytest
import torch

from envs.trace_cache import TraceCache
from envs.cloud_env import CloudEnv
from envs.pipeline_env import PipelineEnv
from router.whittle_router import WhittleRouter


@pytest.fixture
def config():
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def trace_cache():
    return TraceCache("data/traces/synthetic.json")


def test_m7_s4_async_worker_subprocess_and_serving(config, trace_cache):
    """PRD S4 & M7: Credit worker runs as a separate process while router keeps serving."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runs_file = os.path.join(tmpdir, "runs.jsonl")
        credits_file = os.path.join(tmpdir, "credits.jsonl")
        weights_file = os.path.join(tmpdir, "weights.pt")
        stop_file = os.path.join(tmpdir, "stop.flag")

        env = dict(os.environ)
        env["PYTHONPATH"] = os.path.abspath(".")

        # 1. Start async worker subprocess
        cmd = [
            sys.executable,
            "-m",
            "credit.async_worker",
            "--runs-file", runs_file,
            "--credits-file", credits_file,
            "--weights-file", weights_file,
            "--trace-cache", os.path.abspath("data/traces/synthetic.json"),
            "--credit-method", "shapley",
            "--retrain-every-n", "10",
            "--poll-interval-s", "0.05",
            "--stop-flag", stop_file,
        ]
        proc = subprocess.Popen(
            cmd, cwd=os.path.abspath("."), env=env, stderr=subprocess.PIPE, text=True
        )

        try:
            # 2. Serving router keeps serving runs while worker is active
            router = WhittleRouter(seed=42)
            cloud = CloudEnv(config["models"], config["cloud"], sigma=0.0, seed=42)
            pipeline = PipelineEnv(
                trace_cache, cloud, config["reward"], config["models"], log_file=runs_file
            )

            # Serve 15 tasks
            tasks = trace_cache.task_ids()[:15]
            for tid in tasks:
                rec = pipeline.run_task(tid, router, method_name="proposed_shapley")
                assert "utility" in rec

            # Poll for credits_file and retraining weights_file (import torch in worker takes ~2.5s)
            deadline = time.time() + 15.0
            credit_lines = []
            while time.time() < deadline:
                if os.path.exists(credits_file):
                    with open(credits_file, "r", encoding="utf-8") as f:
                        credit_lines = [json.loads(line) for line in f if line.strip()]
                    if len(credit_lines) >= 10 and os.path.exists(weights_file):
                        break
                time.sleep(0.1)

            # 3. Check credits.jsonl was populated
            assert os.path.exists(credits_file), "credits.jsonl should be created by worker"
            assert len(credit_lines) >= 10, f"Expected at least 10 credit records, got {len(credit_lines)}"
            for c in credit_lines:
                assert c["method"] == "shapley"
                assert len(c["role_credits"]) == 3

            # 4. Check retraining happened (retrain_every_n=10) and weights file exists
            assert os.path.exists(weights_file), "weights.pt should be created after retraining"

        finally:
            # Signal stop to worker and wait
            with open(stop_file, "w") as f:
                f.write("stop")
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def test_m7_none_credit_ablation_runs(config, trace_cache):
    """PRD M7: With credit disabled (method='none') the worker runs and assigns equal credit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runs_file = os.path.join(tmpdir, "runs.jsonl")
        credits_file = os.path.join(tmpdir, "credits.jsonl")
        weights_file = os.path.join(tmpdir, "weights.pt")
        stop_file = os.path.join(tmpdir, "stop.flag")

        env = dict(os.environ)
        env["PYTHONPATH"] = os.path.abspath(".")

        cmd = [
            sys.executable,
            "-m",
            "credit.async_worker",
            "--runs-file", runs_file,
            "--credits-file", credits_file,
            "--weights-file", weights_file,
            "--trace-cache", os.path.abspath("data/traces/synthetic.json"),
            "--credit-method", "none",
            "--retrain-every-n", "5",
            "--poll-interval-s", "0.05",
            "--stop-flag", stop_file,
        ]
        proc = subprocess.Popen(
            cmd, cwd=os.path.abspath("."), env=env, stderr=subprocess.PIPE, text=True
        )

        try:
            router = WhittleRouter(seed=42)
            cloud = CloudEnv(config["models"], config["cloud"], sigma=0.0, seed=42)
            pipeline = PipelineEnv(
                trace_cache, cloud, config["reward"], config["models"], log_file=runs_file
            )

            tasks = trace_cache.task_ids()[:6]
            for tid in tasks:
                pipeline.run_task(tid, router, method_name="proposed_none")

            # Poll for credits_file to be populated
            deadline = time.time() + 15.0
            lines = []
            while time.time() < deadline:
                if os.path.exists(credits_file):
                    with open(credits_file, "r", encoding="utf-8") as f:
                        lines = [json.loads(line) for line in f if line.strip()]
                    if len(lines) >= 5:
                        break
                time.sleep(0.1)

            assert os.path.exists(credits_file)
            assert len(lines) >= 5
            for c in lines:
                assert c["method"] == "none"
                # All 3 roles get equal credit (success / 3)
                assert abs(c["role_credits"][0] - c["role_credits"][1]) < 1e-6
                assert abs(c["role_credits"][1] - c["role_credits"][2]) < 1e-6

        finally:
            with open(stop_file, "w") as f:
                f.write("stop")
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
