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
    parser.add_argument("--languages", type=str, default="en,hi,te",
                        help="Comma-separated language codes to benchmark (e.g. en,hi,te).")
    parser.add_argument("--language", type=str, default=None,
                        help="Single language code (legacy option).")
    parser.add_argument("--n", type=int, default=17,
                        help="Number of queries per language.")
    parser.add_argument("--reset", action="store_true", default=True,
                        help="Reset latency_log.csv before starting.")
    args = parser.parse_args()

    if args.reset and os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    print("=================================================================")
    print(" 🚀 Warming up (model load + first inference)...")
    print("=================================================================")
    # Pre-warm embedding, FAISS index, and BM25 index
    answer_query("warmup initialization query", language_filter=None)
    # Remove warmup from log so only benchmark queries count
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    print("Warm-up complete.\n")

    target_langs = [args.language] if args.language else [l.strip() for l in args.languages.split(",") if l.strip()]

    total_grounded = 0
    total_refused = 0
    total_tested = 0

    for lang in target_langs:
        file_prefix = LANGUAGE_TO_FILE_PREFIX.get(lang, "hin")
        split = "validation" if lang == "te" else "train"
        suffix = "val" if split == "validation" else "train"
        local_path = hf_hub_download(
            repo_id="ai4bharat/MSMARCO-XI", repo_type="dataset",
            filename=f"{split}/{file_prefix}{suffix}.parquet",
        )

        parquet_file = pq.ParquetFile(local_path)
        field = "Eng_Query" if lang == "en" else "query"

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

        print(f"Running benchmark on {len(questions)} queries for language '{lang}'...\n")

        for i, q in enumerate(questions):
            res = answer_query(q, language_filter=lang)
            status = "✓ GROUNDED" if res.get("grounded") else "✗ REFUSED"
            if res.get("grounded"):
                total_grounded += 1
            else:
                total_refused += 1
            total_tested += 1

            t = res.get("timings", {})
            total_ms = t.get("total_ms", 0.0)
            retrieval_ms = t.get("retrieval_ms", 0.0)
            gen_ms = t.get("generation_ms", 0.0)
            print(f"  [{total_tested:>2}] ({lang.upper()}) {status} | Total: {total_ms:>5.1f}ms (Retr: {retrieval_ms:>5.1f}ms, Gen: {gen_ms:>4.1f}ms) — {q[:50]}")

    print(f"\nCompleted {total_tested} total queries ({total_grounded} grounded, {total_refused} refused).")
    summarize()


if __name__ == "__main__":
    main()
