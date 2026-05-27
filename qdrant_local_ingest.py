import argparse
import os
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

from chunking import VBPLChunker


def _batch(items: List[Any], size: int) -> List[List[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _ensure_collection(client: QdrantClient, name: str, size: int) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if name in existing:
        return
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=size, distance=Distance.COSINE),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest VBPL parent/child chunks into local Qdrant")
    parser.add_argument("--data-dir", default="/Users/nguyengiahuy/data/data")
    parser.add_argument("--db-path", default="/Users/nguyengiahuy/data/data/.qdrant")
    parser.add_argument("--child-collection", default="legal_child_chunks")
    parser.add_argument("--parent-collection", default="legal_parent_chunks")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--model", default="AITeamVN/Vietnamese_Embedding_v2")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    os.makedirs(args.db_path, exist_ok=True)

    client = QdrantClient(path=args.db_path)
    _ensure_collection(client, args.child_collection, size=1024)
    _ensure_collection(client, args.parent_collection, size=1)

    model = SentenceTransformer(args.model, device=args.device)

    chunker = VBPLChunker()
    result = chunker.process_directory_parent_child(args.data_dir)
    parents = result["parents"]
    children = result["children"]

    # Prepare parent points (dummy vector size=1, retrieval by filter)
    parent_points = []
    for p in parents:
        meta = p.get("metadata", {}) or {}
        payload = {
            "dieu_id": p["dieu_id"],
            "van_ban_id": meta.get("doc_id", ""),
            "content": p["content"],
            "metadata": meta,
        }
        parent_points.append(
            PointStruct(id=p["id"], vector=[0.0], payload=payload)
        )

    # Prepare child points + embeddings
    child_payloads = []
    texts = []
    for c in children:
        meta = c.get("metadata", {}) or {}
        chunk_type = c.get("type", "")
        if chunk_type == "dieu_preamble":
            chunk_type = "dieu"
        payload = {
            "dieu_id": c["dieu_id"],
            "parent_id": c.get("parent_id", ""),
            "van_ban_id": meta.get("doc_id", ""),
            "chunk_type": chunk_type,
            "content": c["content"],
            "metadata": meta,
        }
        child_payloads.append((c["id"], payload))
        texts.append(c["content"])

    embeddings = model.encode(texts, batch_size=64, normalize_embeddings=True)
    if len(embeddings) != len(child_payloads):
        raise RuntimeError("Embedding count mismatch")
    if len(embeddings) > 0 and len(embeddings[0]) != 1024:
        raise RuntimeError("Embedding dimension is not 1024")

    child_points = []
    for i, (point_id, payload) in enumerate(child_payloads):
        child_points.append(
            PointStruct(id=point_id, vector=embeddings[i].tolist(), payload=payload)
        )

    # Upsert parents
    for batch in _batch(parent_points, args.batch_size):
        client.upsert(collection_name=args.parent_collection, points=batch)

    # Upsert children
    for batch in _batch(child_points, args.batch_size):
        client.upsert(collection_name=args.child_collection, points=batch)

    print(f"Inserted parents: {len(parent_points)}")
    print(f"Inserted children: {len(child_points)}")
    print(f"Qdrant path: {args.db_path}")


if __name__ == "__main__":
    main()
