import argparse
import os
import uuid
from typing import Any, List

from qdrant_client import QdrantClient, models

from bm25_sparse import BM25SparseVectorizer, bm25_index_path
from chunking import VBPLChunker
from embedding_model import DEFAULT_FINETUNED_DIR, load_embedding_model


def _batch(items: List[Any], size: int) -> List[List[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _point_id(raw_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw_id))


def _ensure_parent_collection(client: QdrantClient, name: str, recreate: bool) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if name in existing and recreate:
        client.delete_collection(collection_name=name)
        existing.remove(name)
    if name in existing:
        return
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE),
    )


def _ensure_child_collection(client: QdrantClient, name: str, size: int, recreate: bool) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if name in existing and recreate:
        client.delete_collection(collection_name=name)
        existing.remove(name)
    if name in existing:
        return
    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": models.VectorParams(size=size, distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest VBPL parent/child chunks into local Qdrant")
    parser.add_argument("--data-dir", default="data/keep")
    parser.add_argument("--db-path", default="data/.qdrant")
    parser.add_argument("--child-collection", default="legal_child_chunks")
    parser.add_argument("--parent-collection", default="legal_parent_chunks")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--embed-batch-size", type=int, default=16)
    parser.add_argument("--model", default=DEFAULT_FINETUNED_DIR)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--recreate", action="store_true", help="Delete and rebuild Qdrant collections")
    args = parser.parse_args()

    os.makedirs(args.db_path, exist_ok=True)

    client = QdrantClient(path=args.db_path)
    _ensure_child_collection(client, args.child_collection, size=1024, recreate=args.recreate)
    _ensure_parent_collection(client, args.parent_collection, recreate=args.recreate)

    model = load_embedding_model(args.model, device=args.device, max_seq_length=args.max_seq_length)

    chunker = VBPLChunker()
    result = chunker.process_directory_parent_child(args.data_dir)
    parents = result["parents"]
    children = result["children"]

    # Prepare parent points (dummy vector size=1, retrieval by filter)
    parent_points = []
    for p in parents:
        meta = p.get("metadata", {}) or {}
        payload = {
            "id": p["id"],
            "dieu_id": p["dieu_id"],
            "van_ban_id": meta.get("doc_id", ""),
            "content": p["content"],
            "metadata": meta,
        }
        parent_points.append(
            models.PointStruct(id=_point_id(p["id"]), vector=[0.0], payload=payload)
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
            "id": c["id"],
            "dieu_id": c["dieu_id"],
            "parent_id": c.get("parent_id", ""),
            "van_ban_id": meta.get("doc_id", ""),
            "chunk_type": chunk_type,
            "order": c.get("order", 0),
            "content": c["content"],
            "metadata": meta,
        }
        child_payloads.append((c["id"], payload))
        texts.append(c["content"])

    bm25 = BM25SparseVectorizer().fit(texts)
    bm25.save(bm25_index_path(args.db_path, args.child_collection))

    embeddings = model.encode(
        texts,
        batch_size=args.embed_batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    if len(embeddings) != len(child_payloads):
        raise RuntimeError("Embedding count mismatch")
    if len(embeddings) > 0 and len(embeddings[0]) != 1024:
        raise RuntimeError("Embedding dimension is not 1024")

    child_points = []
    for i, (point_id, payload) in enumerate(child_payloads):
        sparse_indices, sparse_values = bm25.encode_document(payload["content"])
        child_points.append(
            models.PointStruct(
                id=_point_id(point_id),
                vector={
                    "dense": embeddings[i].tolist(),
                    "bm25": models.SparseVector(indices=sparse_indices, values=sparse_values),
                },
                payload=payload,
            )
        )

    # Upsert parents
    for batch in _batch(parent_points, args.batch_size):
        client.upsert(collection_name=args.parent_collection, points=batch)

    # Upsert children
    for batch in _batch(child_points, args.batch_size):
        client.upsert(collection_name=args.child_collection, points=batch)

    print(f"Inserted parents: {len(parent_points)}")
    print(f"Inserted children: {len(child_points)}")
    print(f"BM25 index: {bm25_index_path(args.db_path, args.child_collection)}")
    print(f"Qdrant path: {args.db_path}")


if __name__ == "__main__":
    main()
