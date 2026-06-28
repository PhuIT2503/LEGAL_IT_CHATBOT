import argparse
import os
from collections import defaultdict
from typing import List, Dict, Any

from supabase import create_client
from sentence_transformers import SentenceTransformer

from chunking import VBPLChunker


def _get_env_or_prompt(name: str) -> str:
    value = os.environ.get(name)
    if value:
        return value
    return input(f"Enter {name}: ").strip()


def _normalize_type(chunk_type: str) -> str:
    if chunk_type == "dieu_preamble":
        return "dieu"
    return chunk_type


def _batch(iterable: List[Dict[str, Any]], size: int) -> List[List[Dict[str, Any]]]:
    return [iterable[i : i + size] for i in range(0, len(iterable), size)]


def _make_record_ids_unique(records: List[Dict[str, Any]]) -> int:
    seen = defaultdict(int)
    duplicate_count = 0
    for record in records:
        original_id = record["id"]
        seen[original_id] += 1
        if seen[original_id] == 1:
            continue

        duplicate_count += 1
        record.setdefault("metadata", {})
        record["metadata"]["original_id"] = original_id
        record["metadata"]["duplicate_index"] = seen[original_id]
        record["id"] = f"{original_id}__dup{seen[original_id]}"

    return duplicate_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest VBPL parent/child chunks into Supabase")
    parser.add_argument("--data-dir", default="data/keep")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--embed-batch-size", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--model", default="AITeamVN/Vietnamese_Embedding_v2")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    supabase_url = _get_env_or_prompt("SUPABASE_URL")
    supabase_key = _get_env_or_prompt("SUPABASE_SERVICE_KEY")

    supabase = create_client(supabase_url, supabase_key)

    model = SentenceTransformer(args.model, device=args.device)
    model.max_seq_length = args.max_seq_length

    chunker = VBPLChunker()
    result = chunker.process_directory_parent_child(args.data_dir)
    parents = result["parents"]
    children = result["children"]

    # Prepare parent records
    parent_records = []
    for p in parents:
        meta = p.get("metadata", {}) or {}
        parent_records.append({
            "id": p["id"],
            "dieu_id": p["dieu_id"],
            "van_ban_id": meta.get("doc_id", ""),
            "content": p["content"],
            "metadata": meta,
        })

    # Prepare child records
    child_records = []
    texts = []
    for c in children:
        meta = c.get("metadata", {}) or {}
        chunk_type = _normalize_type(c.get("type", ""))
        child_records.append({
            "id": c["id"],
            "dieu_id": c["dieu_id"],
            "parent_id": c.get("parent_id", ""),
            "van_ban_id": meta.get("doc_id", ""),
            "chunk_type": chunk_type,
            "content": c["content"],
            "metadata": meta,
        })
        texts.append(c["content"])

    duplicate_child_ids = _make_record_ids_unique(child_records)
    if duplicate_child_ids:
        print(f"Renamed duplicate child ids: {duplicate_child_ids}")

    # Embed child chunks
    embeddings = model.encode(
        texts,
        batch_size=args.embed_batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    if len(embeddings) != len(child_records):
        raise RuntimeError("Embedding count mismatch")
    if len(embeddings) > 0 and len(embeddings[0]) != 1024:
        raise RuntimeError("Embedding dimension is not 1024")

    for i, emb in enumerate(embeddings):
        child_records[i]["embedding"] = emb.tolist()

    # Upsert parents
    for batch in _batch(parent_records, args.batch_size):
        supabase.table("legal_parent_chunks").upsert(batch).execute()

    # Upsert children
    for batch in _batch(child_records, args.batch_size):
        supabase.table("legal_child_chunks").upsert(batch).execute()

    print(f"Inserted parents: {len(parent_records)}")
    print(f"Inserted children: {len(child_records)}")


if __name__ == "__main__":
    main()
