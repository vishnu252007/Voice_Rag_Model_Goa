"""
Runs a batch of REAL questions (pulled directly from the same MSMARCO-XI file you
indexed) through the full harness, so every question is guaranteed to have a real
matching answer in your data — giving you honest, non-refusal latency numbers for
your submission's P50/P70/P100 requirement.

Run: python run_benchmark.py --language hi --n 30

This appends to latency_log.csv (same file harness.py writes to normally), so you
can run this multiple times to build up a larger sample before generating your
final report with: python latency.py
"""
import argparse
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from harness import answer_query

LANGUAGE_TO_FILE_PREFIX = {
    "as": "asm", "bn": "ben", "gu": "guj", "hi": "hin", "kn": "kan",
    "ml": "mal", "mr": "mar", "ne": "nep", "or": "ori", "pa": "pan",
    "sa": "san", "ta": "tam", "te": "tel", "ur": "urd",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="hi")
    parser.add_argument("--n", type=int, default=30, help="Number of real questions to test.")
    parser.add_argument("--query_field", type=str, default="query",
                         choices=["query", "Eng_Query"],
                         help="'query' = question in the target language (matches --content "
                              "translated indexing). 'Eng_Query' = English question (matches "
                              "--content english indexing).")
    args = parser.parse_args()

    file_prefix = LANGUAGE_TO_FILE_PREFIX[args.language]
    local_path = hf_hub_download(
        repo_id="ai4bharat/MSMARCO-XI", repo_type="dataset",
        filename=f"train/{file_prefix}train.parquet",
    )

    parquet_file = pq.ParquetFile(local_path)
    questions = []
    for batch in parquet_file.iter_batches(batch_size=50):
        for row in batch.to_pylist():
            q = row.get(args.query_field, "")
            if q and q.strip():
                questions.append(q.strip())
            if len(questions) >= args.n:
                break
        if len(questions) >= args.n:
            break

    print(f"Loaded {len(questions)} real questions from the dataset. Running through the harness...\n")

    grounded_count = 0
    refused_count = 0
    for i, q in enumerate(questions):
        result = answer_query(q, language_filter=args.language if args.query_field == "query" else "en")
        status = "GROUNDED" if result.get("grounded") else "REFUSED"
        if result.get("grounded"):
            grounded_count += 1
        else:
            refused_count += 1
        total_ms = result.get("timings", {}).get("total_ms", 0)
        print(f"  [{i+1}/{len(questions)}] {status} ({total_ms}ms) — {q[:60]}")

    print(f"\nDone. {grounded_count} grounded, {refused_count} refused out of {len(questions)}.")
    print("Run 'python latency.py' now to see your P50/P70/P100 report.")


if __name__ == "__main__":
    main()
