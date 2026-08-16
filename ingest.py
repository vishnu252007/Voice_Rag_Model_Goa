"""
Run this ONCE (and again whenever you change chunking strategy) to build your
searchable index from the ai4bharat/MSMARCO-XI dataset.

    python ingest.py --strategy metadata --limit 500 --language hi --content english

This downloads ONLY the file for the language you pick (not the full 55GB
dataset), using Hugging Face's resumable downloader — if your connection
drops mid-download, re-running the same command picks up where it left off
instead of starting over or failing outright.

This does three things in order:
  1. Downloads the single language-specific parquet file
  2. Chunks every passage using the strategy you pick
  3. Uploads all chunks to Qdrant AND builds the local BM25 index
"""
import argparse
import pickle
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from chunking import fixed_size_chunk, semantic_chunk, metadata_aware_chunk
from vectorstore import upload_chunks
from retrieval import build_bm25_index

STRATEGY_FUNCS = {
    "fixed": fixed_size_chunk,
    "semantic": semantic_chunk,
    "metadata": metadata_aware_chunk,
}

# Where the BM25 index gets saved so harness.py / retrieval.py can load it at
# server startup without re-processing the whole dataset every time.
BM25_CACHE_PATH = "bm25_chunks.pkl"

SUPPORTED_LANGUAGES = ["as", "bn", "gu", "hi", "kn", "ml", "mr", "ne", "or", "pa", "sa", "ta", "te", "ur"]

# The actual files on the repo use 3-letter prefixes, not the 2-letter codes above
# (confirmed via list_repo_files — e.g. 'hintrain.parquet' for Hindi, not 'hitrain.parquet').
LANGUAGE_TO_FILE_PREFIX = {
    "as": "asm", "bn": "ben", "gu": "guj", "hi": "hin", "kn": "kan",
    "ml": "mal", "mr": "mar", "ne": "nep", "or": "ori", "pa": "pan",
    "sa": "san", "ta": "tam", "te": "tel", "ur": "urd",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=list(STRATEGY_FUNCS.keys()), default="metadata")
    parser.add_argument("--limit", type=int, default=1000,
                         help="Number of queries (rows) to process from the downloaded file.")
    parser.add_argument("--language", type=str, required=True, choices=SUPPORTED_LANGUAGES,
                         help="Which language file to download, e.g. 'hi' for Hindi.")
    parser.add_argument("--content", type=str, choices=["english", "translated"], default="translated",
                         help="'english' uses the original English passages (easier to test/debug yourself). "
                              "'translated' uses the Indic-language passages for --language.")
    parser.add_argument("--split", type=str, choices=["train", "validation"], default="train")
    args = parser.parse_args()

    # File naming pattern confirmed via list_repo_files: train/{prefix}train.parquet,
    # validation/{prefix}val.parquet (e.g. train/hintrain.parquet, validation/telval.parquet).
    file_prefix = LANGUAGE_TO_FILE_PREFIX[args.language]
    suffix = "train" if args.split == "train" else "val"
    remote_filename = f"{args.split}/{file_prefix}{suffix}.parquet"

    print(f"Downloading {remote_filename} (resumable — safe to re-run this exact "
          f"command if your connection drops, it won't restart from zero)...")
    local_path = hf_hub_download(
        repo_id="ai4bharat/MSMARCO-XI",
        repo_type="dataset",
        filename=remote_filename,
    )
    print(f"Downloaded to: {local_path}")

    print("Reading only the rows we need directly from the parquet file "
          "(skips converting the whole 3.7GB file, which is what was hanging before)...")
    parquet_file = pq.ParquetFile(local_path)

    chunk_fn = STRATEGY_FUNCS[args.strategy]
    all_chunks = []
    passage_field = "English_passages" if args.content == "english" else "Translated_passages"
    tag_language = "en" if args.content == "english" else args.language

    n_processed = 0
    for batch in parquet_file.iter_batches(batch_size=25):
        rows = batch.to_pylist()
        for row in rows:
            if n_processed >= args.limit:
                break
            query_id = row.get("query_id", n_processed)
            passages_list = row.get("passages", {}).get(passage_field, [])

            for p_idx, passage_text in enumerate(passages_list):
                if not passage_text or not passage_text.strip():
                    continue
                doc_id = f"q{query_id}_p{p_idx}"
                chunks = chunk_fn(passage_text, doc_id, language=tag_language)
                all_chunks.extend(chunks)

            n_processed += 1
            if n_processed % 25 == 0:
                print(f"  processed {n_processed}/{args.limit} rows -> {len(all_chunks)} chunks so far")

        if n_processed >= args.limit:
            break

    print(f"\nTotal chunks: {len(all_chunks)}")

    print("Uploading to Qdrant...")
    n = upload_chunks(all_chunks)
    print(f"Uploaded {n} chunks to vector DB.")

    print("Building BM25 index...")
    build_bm25_index(all_chunks)
    with open(BM25_CACHE_PATH, "wb") as f:
        pickle.dump(all_chunks, f)
    print(f"Saved BM25 chunk cache to {BM25_CACHE_PATH}")

    print("\nDone. You can now run: uvicorn main:app --reload")


if __name__ == "__main__":
    main()