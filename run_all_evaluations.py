"""
Sequential Evaluation Suite Runner
Strictly runs all individual and base-model evaluation benchmarks one after another.

Skipped (as requested):
- evaluations/group/evaluate_mars_local.py (running separately)
- evaluations/group/evaluate_market1501_local.py

Execution:
    python run_all_evaluations.py
"""

import os
import sys
import time
import subprocess
from datetime import timedelta

# Project Root Directory
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

# Strictly ordered benchmark list (excluding evaluate_mars_local and evaluate_market1501_local)
BENCHMARKS = [
    # -------------------------------------------------------------
    # 1. MARS - Individual Feature Benchmarks
    # -------------------------------------------------------------
    {
        "id": "mars_app",
        "name": "MARS - Standalone Appearance (OSNet Re-ID)",
        "script": os.path.join(ROOT_DIR, "evaluations", "individual", "appearance_mars.py"),
        "description": "Evaluates tracklet appearance matching with stride-based sampling on GPU."
    },
    {
        "id": "mars_face",
        "name": "MARS - Standalone Face (InsightFace ArcFace)",
        "script": os.path.join(ROOT_DIR, "evaluations", "individual", "face_mars.py"),
        "description": "Evaluates ArcFace recognition and face detection coverage across MARS tracklets."
    },
    {
        "id": "mars_gait",
        "name": "MARS - Standalone Gait (Base Paper GEI vs SkeletonGait)",
        "script": os.path.join(ROOT_DIR, "evaluations", "individual", "gait_mars.py"),
        "description": "Head-to-head comparison between base paper GEI+PCA and YOLOv8-pose 3D CNN."
    },

    # -------------------------------------------------------------
    # 2. Market-1501 - Individual Feature Benchmarks
    # -------------------------------------------------------------
    {
        "id": "market_app",
        "name": "Market-1501 - Standalone Appearance (OSNet Re-ID)",
        "script": os.path.join(ROOT_DIR, "evaluations", "individual", "appearance_market1501.py"),
        "description": "Batched OSNet evaluation following official Market-1501 camera filtering protocol."
    },
    {
        "id": "market_face",
        "name": "Market-1501 - Standalone Face (InsightFace ArcFace)",
        "script": os.path.join(ROOT_DIR, "evaluations", "individual", "face_market1501.py"),
        "description": "Evaluates ArcFace retrieval and face detection rates on static Market-1501 crops."
    },

    # -------------------------------------------------------------
    # 3. MARS - Base Paper Multi-Modal Benchmark
    # -------------------------------------------------------------
    {
        "id": "mars_group_paper",
        "name": "MARS - Multi-Modal Pipeline (Base Paper Implementation)",
        "script": os.path.join(ROOT_DIR, "evaluations", "group", "evaluate_project_mars_local.py"),
        "description": "Base paper reference benchmark using Project/ feature extractors."
    },
]


def print_banner(text: str, char="="):
    line = char * 75
    print(f"\n{line}")
    print(f" {text}")
    print(f"{line}\n")


def format_duration(seconds: float) -> str:
    """Formats elapsed seconds into H:M:S string."""
    return str(timedelta(seconds=int(seconds)))


def run_benchmark(item: dict) -> dict:
    """Runs a single benchmark in a separate Python process to cleanly flush GPU VRAM."""
    script_path = item["script"]
    name = item["name"]

    print_banner(f"🚀 RUNNING: {name}", char="=")
    print(f" Script Path : {script_path}")
    print(f" Description : {item['description']}")
    print(f" Python Exec : {sys.executable}\n")

    if not os.path.exists(script_path):
        print(f"❌ Error: Script file not found at '{script_path}'! Skipping...")
        return {
            "name": name,
            "status": "MISSING",
            "elapsed_sec": 0,
            "exit_code": -1
        }

    start_time = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    try:
        # Run process interactively so output streams in real-time
        process = subprocess.run(
            [sys.executable, script_path],
            cwd=ROOT_DIR,
            env=env
        )
        elapsed = time.perf_counter() - start_time
        status = "PASSED" if process.returncode == 0 else "FAILED"
        return {
            "name": name,
            "status": status,
            "elapsed_sec": elapsed,
            "exit_code": process.returncode
        }
    except KeyboardInterrupt:
        print("\n\n⚠️ Suite interrupted by user (Ctrl+C).")
        raise
    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        print(f"\n❌ Execution exception: {exc}")
        return {
            "name": name,
            "status": "ERROR",
            "elapsed_sec": elapsed,
            "exit_code": -1
        }


def print_final_summary(results: list):
    """Prints a consolidated summary report table of all executed benchmarks."""
    print_banner("🏁 ALL BENCHMARKS COMPLETED - FINAL SUMMARY REPORT 🏁", char="=")

    print(f"{'#':<3} | {'Benchmark Name':<52} | {'Status':<10} | {'Duration':<10}")
    print("-" * 82)

    total_time = 0
    passed_count = 0

    for idx, res in enumerate(results, start=1):
        status_str = res["status"]
        if status_str == "PASSED":
            status_icon = "✅ PASSED"
            passed_count += 1
        elif status_str == "FAILED":
            status_icon = "❌ FAILED"
        elif status_str == "MISSING":
            status_icon = "⚠️ MISSING"
        else:
            status_icon = "⚠️ " + status_str

        duration_str = format_duration(res["elapsed_sec"])
        total_time += res["elapsed_sec"]

        print(f"{idx:<3} | {res['name'][:52]:<52} | {status_icon:<10} | {duration_str:<10}")

    print("-" * 82)
    print(f"Total Benchmarks Run : {len(results)}")
    print(f"Successful (Passed)  : {passed_count} / {len(results)}")
    print(f"Total Elapsed Time   : {format_duration(total_time)}")
    print("=" * 82 + "\n")


def main():
    print_banner(f"STARTING SEQUENTIAL EVALUATION SUITE ({len(BENCHMARKS)} TESTS)", char="*")
    print(f"Working Directory: {ROOT_DIR}")
    print(f"Hardware Target  : NVIDIA GPU (CUDA)")
    print("Excluded (Skipped):")
    print("  - evaluations/group/evaluate_mars_local.py")
    print("  - evaluations/group/evaluate_market1501_local.py\n")

    results = []
    try:
        for idx, item in enumerate(BENCHMARKS, start=1):
            print(f"\n>>> Running Test [{idx}/{len(BENCHMARKS)}]: {item['name']}")
            res = run_benchmark(item)
            results.append(res)
    except KeyboardInterrupt:
        print("\n\nExecution stopped by user.")
    finally:
        if results:
            print_final_summary(results)


if __name__ == "__main__":
    main()
