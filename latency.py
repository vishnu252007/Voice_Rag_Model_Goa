"""
A tiny stopwatch utility used by harness.py to time every stage of the pipeline,
plus a helper to compute P50/P70/P100 across many logged runs for your submission.

HOW TO GENERATE YOUR SUBMISSION NUMBERS:
    python latency.py
(after you've run enough real queries through harness.py and it has appended
their timings to latency_log.csv)
"""
import time
import csv
import os
from contextlib import contextmanager

LOG_FILE = os.path.join(os.path.dirname(__file__), "latency_log.csv")


@contextmanager
def timed_stage():
    """Usage:
        with timed_stage() as t:
            do_the_work()
        elapsed_ms = t["ms"]
    """
    result = {}
    start = time.perf_counter()
    yield result
    result["ms"] = (time.perf_counter() - start) * 1000


def log_run(stage_timings: dict):
    """stage_timings example: {"stt_ms": 120, "retrieval_ms": 80, "generation_ms": 400, "total_ms": 600}"""
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stage_timings.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(stage_timings)


def percentile(values, p):
    if not values:
        return 0
    values = sorted(values)
    idx = min(int(len(values) * p / 100), len(values) - 1)
    return values[idx]


def summarize():
    if not os.path.exists(LOG_FILE):
        print("No latency_log.csv yet — run some real queries through harness.py first.")
        return

    rows = list(csv.DictReader(open(LOG_FILE)))
    if not rows:
        print("Log file is empty.")
        return

    fields = rows[0].keys()
    print(f"Summary across {len(rows)} logged queries:\n")
    for field in fields:
        values = [float(r[field]) for r in rows]
        print(f"{field:>20}  P50={percentile(values,50):7.1f}ms  "
              f"P70={percentile(values,70):7.1f}ms  P100={percentile(values,100):7.1f}ms")


if __name__ == "__main__":
    summarize()
