"""
Latency Analytics & Benchmarking Module.
Computes P50 / P70 / P95 / P99 / P100 percentile distributions across logged queries.
Saves detailed JSON reports to results/latency_report.json.
"""
import time
import csv
import json
import os
import sys
from contextlib import contextmanager
import numpy as np

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

LOG_FILE = os.path.join(os.path.dirname(__file__), "latency_log.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
REPORT_FILE = os.path.join(RESULTS_DIR, "latency_report.json")


@contextmanager
def timed_stage():
    """Context manager for high-precision stage timing in milliseconds."""
    result = {}
    start = time.perf_counter()
    yield result
    result["ms"] = (time.perf_counter() - start) * 1000.0


def log_run(stage_timings: dict):
    """Appends stage timings dictionary to latency_log.csv."""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    file_exists = os.path.exists(LOG_FILE)
    
    clean_timings = {k: v for k, v in stage_timings.items() if isinstance(v, (int, float))}
    if not clean_timings:
        return

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(clean_timings.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(clean_timings)


def compute_percentiles(values: list) -> dict:
    if not values:
        return {"p50": 0.0, "p70": 0.0, "p95": 0.0, "p99": 0.0, "p100": 0.0, "mean": 0.0}
    arr = np.array(values, dtype=float)
    return {
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p70": round(float(np.percentile(arr, 70)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "p99": round(float(np.percentile(arr, 99)), 2),
        "p100": round(float(np.percentile(arr, 100)), 2),
        "mean": round(float(np.mean(arr)), 2),
    }


def summarize():
    if not os.path.exists(LOG_FILE):
        print("No latency_log.csv found — run benchmark queries first.")
        return

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("Latency log is empty.")
        return

    field_data = {}
    for r in rows:
        for k, v in r.items():
            if v and v.strip():
                try:
                    field_data.setdefault(k, []).append(float(v))
                except ValueError:
                    pass

    report = {
        "total_queries": len(rows),
        "stages": {}
    }

    print("\n" + "=" * 65)
    print(" 📊 STAGE-BY-STAGE LATENCY BREAKDOWN (POST-STT)")
    print("=" * 65)
    print(f"{'Stage':<25} {'P50 (ms)':>10} {'P70 (ms)':>10} {'P95 (ms)':>10} {'P99 (ms)':>10}")
    print("-" * 65)

    for field, values in field_data.items():
        stats = compute_percentiles(values)
        report["stages"][field] = stats
        print(f"{field:<25} {stats['p50']:>10.2f} {stats['p70']:>10.2f} {stats['p95']:>10.2f} {stats['p99']:>10.2f}")

    total_stats = report["stages"].get("total_ms", compute_percentiles(field_data.get("total_ms", [])))
    print("=" * 65)
    print(f"TOTAL / WALL (P50: {total_stats['p50']}ms | P95: {total_stats['p95']}ms | P99: {total_stats['p99']}ms)")
    if total_stats['p95'] <= 200.0:
        print(f"🎯 Latency budget target: 200ms | Status: PASS ({total_stats['p95']}ms <= 200ms)")
    print("=" * 65)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"📄 Full report saved to: {REPORT_FILE}\n")


if __name__ == "__main__":
    summarize()
