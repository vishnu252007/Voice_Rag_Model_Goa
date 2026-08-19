"""
Automated Latency Benchmark Suite.
Runs batch queries across languages, measures stage-by-stage timings,
and produces P50 / P70 / P95 / P99 / P100 latency reports.

Usage:
    python run_benchmark.py --language en --n 30
    python run_benchmark.py --language hi --n 30
    python run_benchmark.py --language te --n 30
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import torch
import pyarrow.parquet as pq
import argparse
from huggingface_hub import hf_hub_download

from harness import answer_query
from latency import summarize, LOG_FILE

LANGUAGE_TO_FILE_PREFIX = {
    "en": "hin", "as": "asm", "bn": "ben", "gu": "guj", "hi": "hin", "kn": "kan",
    "ml": "mal", "mr": "mar", "ne": "nep", "or": "ori", "pa": "pan",
    "sa": "san", "ta": "tam", "te": "tel", "ur": "urd",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="en", choices=list(LANGUAGE_TO_FILE_PREFIX.keys()))
    parser.add_argument("--n", type=int, default=30, help="Number of queries to benchmark.")
    parser.add_argument("--reset", action="store_true", help="Reset latency_log.csv before starting.")
    args = parser.parse_args()

    print("=================================================================")
    print(" 🚀 Warming up (model load + first inference)...")
    print("=================================================================")
    # Pre-warm embedding, FAISS index, and BM25 index
    answer_query("warmup initialization text", language_filter=None)
    if args.reset and os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    print("Warm-up complete.\n")

    file_prefix = LANGUAGE_TO_FILE_PREFIX.get(args.language, "hin")
    local_path = hf_hub_download(
        repo_id="ai4bharat/MSMARCO-XI", repo_type="dataset",
        filename=f"train/{file_prefix}train.parquet",
    )

    parquet_file = pq.ParquetFile(local_path)
    field = "Eng_Query" if args.language == "en" else "query"

    questions = []
    for batch in parquet_file.iter_batches(batch_size=50):
        for row in batch.to_pylist():
            q = row.get(field, "")
            if q and len(q.strip()) > 5:
                questions.append(q.strip())
            if len(questions) >= args.n:
                break
        if len(questions) >= args.n:
            break

    print(f"Running benchmark on {len(questions)} queries for language '{args.language}'...\n")

    grounded_count = 0
    refused_count = 0

    for i, q in enumerate(questions):
        res = answer_query(q, language_filter=args.language)
        status = "✓ GROUNDED" if res.get("grounded") else "✗ REFUSED"
        if res.get("grounded"):
            grounded_count += 1
        else:
            refused_count += 1

        t = res.get("timings", {})
        total_ms = t.get("total_ms", 0.0)
        retrieval_ms = t.get("retrieval_ms", 0.0)
        gen_ms = t.get("generation_ms", 0.0)
        print(f"  [{i+1:>2}/{len(questions)}] {status} | Total: {total_ms:>5.1f}ms (Retr: {retrieval_ms:>5.1f}ms, Gen: {gen_ms:>4.1f}ms) — {q[:55]}")

    print(f"\nCompleted {len(questions)} queries ({grounded_count} grounded, {refused_count} refused).")
    summarize()


if __name__ == "__main__":
    main()
