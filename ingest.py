"""
Dataset Ingestion Script for MSMARCO-XI:
Downloads language-specific parquet splits from Hugging Face, applies chunking strategy,
and populates the in-process FAISS vector store + BM25 cache.

Example usage:
    python ingest.py --language en --limit 300 --strategy metadata
    python ingest.py --language hi --limit 300 --strategy metadata
    python ingest.py --language te --limit 300 --strategy metadata
    python ingest.py --language ta --limit 300 --strategy metadata
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Must import torch before pyarrow on Windows to prevent DLL collision
import torch
import pyarrow.parquet as pq

import argparse
import pickle
from huggingface_hub import hf_hub_download

from chunking import fixed_size_chunk, semantic_chunk, metadata_aware_chunk
from config import VECTOR_BACKEND
from retrieval import build_bm25_index, save_bm25_index

if VECTOR_BACKEND == "faiss":
    from faiss_store import build_index as upload_chunks
else:
    from vectorstore import upload_chunks

STRATEGY_FUNCS = {
    "fixed": fixed_size_chunk,
    "semantic": semantic_chunk,
    "metadata": metadata_aware_chunk,
}

BM25_CACHE_PATH = "bm25_chunks.pkl"

SUPPORTED_LANGUAGES = ["en", "as", "bn", "gu", "hi", "kn", "ml", "mr", "ne", "or", "pa", "sa", "ta", "te", "ur"]

LANGUAGE_TO_FILE_PREFIX = {
    "en": "hin", "as": "asm", "bn": "ben", "gu": "guj", "hi": "hin", "kn": "kan",
    "ml": "mal", "mr": "mar", "ne": "nep", "or": "ori", "pa": "pan",
    "sa": "san", "ta": "tam", "te": "tel", "ur": "urd",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=list(STRATEGY_FUNCS.keys()), default="metadata")
    parser.add_argument("--limit", type=int, default=300,
                        help="Number of query rows to ingest.")
    parser.add_argument("--language", type=str, required=True, choices=SUPPORTED_LANGUAGES,
                        help="Language code to ingest (e.g. en, hi, te, ta).")
    parser.add_argument("--content", type=str, choices=["english", "translated"], default="translated",
                        help="'english' uses English passages; 'translated' uses Indic language passages.")
    parser.add_argument("--split", type=str, choices=["train", "validation"], default="train")
    args = parser.parse_args()

    file_prefix = LANGUAGE_TO_FILE_PREFIX.get(args.language, "hin")
    split = args.split

    # AI4Bharat uploaded Telugu under validation split (97,941 rows available)
    if args.language == "te" and split == "train":
        split = "validation"

    suffix = "train" if split == "train" else "val"
    remote_filename = f"{split}/{file_prefix}{suffix}.parquet"

    print(f"Fetching dataset split: {remote_filename} from ai4bharat/MSMARCO-XI...")
    local_path = hf_hub_download(
        repo_id="ai4bharat/MSMARCO-XI",
        repo_type="dataset",
        filename=remote_filename,
    )
    print(f"Local file: {local_path}")

    parquet_file = pq.ParquetFile(local_path)
    chunk_fn = STRATEGY_FUNCS[args.strategy]
    all_chunks = []
    
    if args.language == "en" or args.content == "english":
        passage_field = "English_passages"
        tag_language = "en"
    else:
        passage_field = "Translated_passages"
        tag_language = args.language

    n_processed = 0
    print(f"Reading rows and creating '{args.strategy}' chunks for language '{tag_language}'...")
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
                doc_id = f"{args.split}_{tag_language}_q{query_id}_p{p_idx}"
                chunks = chunk_fn(passage_text, doc_id, language=tag_language)
                all_chunks.extend(chunks)

            n_processed += 1

        if n_processed >= args.limit:
            break

    print(f"Processed {n_processed} rows -> generated {len(all_chunks)} chunks.")

    print(f"Indexing into FAISS ({VECTOR_BACKEND})...")
    n_added = upload_chunks(all_chunks)
    print(f"Added {n_added} chunks to the vector store.")

    existing_chunks = []
    if os.path.exists(BM25_CACHE_PATH):
        try:
            with open(BM25_CACHE_PATH, "rb") as f:
                existing_chunks = pickle.load(f)
        except Exception:
            existing_chunks = []

    existing_ids = {c["chunk_id"] for c in existing_chunks}
    merged_chunks = existing_chunks + [c for c in all_chunks if c["chunk_id"] not in existing_ids]

    print(f"Rebuilding BM25 keyword index ({len(merged_chunks)} total chunks)...")
    build_bm25_index(merged_chunks)
    with open(BM25_CACHE_PATH, "wb") as f:
        pickle.dump(merged_chunks, f)
    print(f"Saved raw BM25 chunk cache to {BM25_CACHE_PATH}.")
    save_bm25_index("bm25_index.pkl")
    print("Ingestion completed successfully!")


if __name__ == "__main__":
    main()