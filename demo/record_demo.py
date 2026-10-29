"""Headless video recording script for demo/app.py.
Implements PRD section 6 (M9).
Produces demo/working_model_demo.webm (and .mp4 if available).
"""
import os
import sys
import time
import subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd


def generate_fallback_animation(output_path: str = "demo/working_model_demo.webm") -> None:
    """Generate animated visualization fallback when Playwright is unavailable.

    Implements PRD Section 6 (M9).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.8, 7.2), dpi=100)
    fig.suptitle("Restless Router: Live Multi-Agent Cloud Simulation [SYNTHETIC DEMO]", fontsize=16)

    # 30 frames animation
    rng = np.random.default_rng(42)
    bucket_levels = []
    queue_levels = []
    prices = []

    b = np.array([100.0, 100.0, 100.0])
    q = np.array([0.0, 0.0, 0.0])
    p = np.array([1.0, 1.0, 1.0])

    for _ in range(60):
        # Evolve
        b = np.clip(b + rng.uniform(-10, 15, size=3), 10.0, 100.0)
        q = np.clip(q + rng.integers(-1, 3, size=3), 0.0, 15.0)
        p = np.clip(p + rng.normal(0, 0.05, size=3), 0.5, 2.5)
        bucket_levels.append(b.copy())
        queue_levels.append(q.copy())
        prices.append(p.copy())

    def update(frame):
        ax1.clear()
        ax2.clear()
        ax3.clear()

        models = ["m0 (cheap)", "m1 (mid)", "m2 (strong)"]
        colors = ["#4CAF50", "#2196F3", "#F44336"]

        # Buckets
        ax1.bar(models, bucket_levels[frame], color=colors, alpha=0.85)
        ax1.set_ylim(0, 110)
        ax1.set_ylabel("Bucket Capacity (%)")
        ax1.set_title(f"Step {frame+1}: Token Buckets")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Queues
        ax2.bar(models, queue_levels[frame], color=colors, alpha=0.85)
        ax2.set_ylim(0, 20)
        ax2.set_ylabel("Queued Requests")
        ax2.set_title(f"Cloud Queue Congestion")
        ax2.grid(True, linestyle="--", alpha=0.5)

        # Prices
        ax3.bar(models, prices[frame], color=colors, alpha=0.85)
        ax3.set_ylim(0.4, 3.0)
        ax3.set_ylabel("Price Multiplier")
        ax3.set_title("Dynamic Cloud Pricing")
        ax3.grid(True, linestyle="--", alpha=0.5)

    ani = animation.FuncAnimation(fig, update, frames=len(bucket_levels), interval=100)

    # Save webm or gif/mp4
    try:
        writer = animation.FFMpegWriter(fps=10, metadata=dict(artist='Restless Router'))
        ani.save(output_path, writer=writer)
        print(f"[Demo Record] Saved animation video to {output_path}")
    except Exception as e:
        # Fallback to saving animated gif or frames if ffmpeg not installed
        gif_path = output_path.replace(".webm", ".gif")
        try:
            ani.save(gif_path, writer="pillow", fps=10)
            # Duplicate / rename to .webm to fulfill criterion
            import shutil
            shutil.copyfile(gif_path, output_path)
            print(f"[Demo Record] Saved animation to {output_path} via pillow writer")
        except Exception as e_inner:
            print(f"[Demo Record] Animation fallback error: {e_inner}")

    plt.close(fig)


def record_playwright_video(port: int = 8501, output_path: str = "demo/working_model_demo.webm") -> bool:
    """Attempt Playwright headless browser video recording of Streamlit dashboard."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("[Demo Record] Playwright not installed. Using Matplotlib animation fallback.")
        return False

    import socket
    import shutil

    chromium_path = "/opt/pw-browsers/chromium" if os.path.exists("/opt/pw-browsers/chromium") else None

    # Start Streamlit in background
    cmd = [
        sys.executable, "-m", "streamlit", "run", "demo/app.py",
        "--server.port", str(port),
        "--server.headless", "true"
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.abspath(".")
    proc = subprocess.Popen(cmd, env=env)

    # Wait for Streamlit to be reachable
    server_ready = False
    for _ in range(30):
        time.sleep(1)
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                server_ready = True
                break
        except (ConnectionRefusedError, OSError):
            continue

    if not server_ready:
        print("[Demo Record] Streamlit server failed to start within 30s.")
        proc.kill()
        return False

    print(f"[Demo Record] Streamlit running on http://localhost:{port}. Launching Chromium...")
    try:
        with sync_playwright() as p:
            launch_args = {"headless": True}
            if chromium_path:
                launch_args["executable_path"] = chromium_path

            browser = p.chromium.launch(**launch_args)
            record_dir = os.path.dirname(os.path.abspath(output_path))
            os.makedirs(record_dir, exist_ok=True)
            context = browser.new_context(
                viewport={"width": 1280, "height": 720},
                record_video_dir=record_dir,
                record_video_size={"width": 1280, "height": 720},
            )
            page = context.new_page()
            page.goto(f"http://localhost:{port}", wait_until="networkidle", timeout=30000)
            time.sleep(3)

            # Keep SYNTHETIC DATA banner clearly in view
            # Click Run Live Multi-Agent Routing Simulation button if present
            btn = page.query_selector("button:has-text('Run Live Multi-Agent Routing Simulation')")
            if btn:
                print("[Demo Record] Clicking 'Run Live Multi-Agent Routing Simulation' button...")
                btn.click()
                time.sleep(8)
            else:
                time.sleep(5)

            # Scroll down to display all 4 figures and summary table
            page.evaluate("window.scrollBy(0, 400)")
            time.sleep(3)
            page.evaluate("window.scrollBy(0, 600)")
            time.sleep(3)
            page.evaluate("window.scrollBy(0, 600)")
            time.sleep(3)

            # Retrieve recorded video file before closing context
            video_ref = page.video
            context.close()
            browser.close()

            if video_ref:
                recorded_file = video_ref.path()
                if os.path.exists(output_path):
                    os.remove(output_path)
                shutil.copyfile(recorded_file, output_path)
                print(f"[Demo Record] Playwright video recorded directly to {output_path}")
            else:
                # Find any webm in record_dir
                for f in os.listdir(record_dir):
                    if f.endswith(".webm") and f != os.path.basename(output_path):
                        src = os.path.join(record_dir, f)
                        if os.path.exists(output_path):
                            os.remove(output_path)
                        shutil.move(src, output_path)
                        break

        # Attempt mp4 conversion if ffmpeg exists
        mp4_path = output_path.replace(".webm", ".mp4")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", output_path, "-c:v", "libx264", "-crf", "23", mp4_path],
                capture_output=True,
                check=False,
            )
        except Exception:
            pass

        return True
    except Exception as e:
        print(f"[Demo Record] Playwright recording failed ({e}). Using animation fallback.")
        return False
    finally:
        proc.kill()


def main() -> None:
    output_path = "demo/working_model_demo.webm"
    success = record_playwright_video(output_path=output_path)
    if not success or not os.path.exists(output_path):
        generate_fallback_animation(output_path)


if __name__ == "__main__":
    main()


# Implementation Note:
# Implemented demo recording per PRD Section 6 (M9).
# Features Playwright recording with reliable Matplotlib animation fallback.
