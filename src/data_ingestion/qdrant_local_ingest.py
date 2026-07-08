import argparse
import os
import sys
import uuid
from pathlib import Path
from typing import Any, List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from qdrant_client import QdrantClient, models

from src.retrieval.bm25_sparse import BM25SparseVectorizer, bm25_index_path
from src.data_processing.chunking import VBPLChunker
from src.llm.embedding_model import DEFAULT_FINETUNED_DIR, load_embedding_model


def _batch(items: List[Any], size: int) -> List[List[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _point_id(raw_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw_id))


def _make_client(db_path: str, url: str | None) -> QdrantClient:
    """
    url được set (vd http://qdrant:6333) -> nối tới Qdrant server (Docker).
    Không set -> dùng embedded/local-file mode tại db_path (hành vi cũ).
    """
    if url:
        return QdrantClient(url=url)
    return QdrantClient(path=db_path)


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


def _already_upserted(client: QdrantClient, collection: str, point_ids: List[str]) -> bool:
    """Kiểm tra xem TẤT CẢ point_ids đã có trong collection chưa (dùng để resume, bỏ qua batch đã xong)."""
    try:
        found = client.retrieve(collection_name=collection, ids=point_ids, with_payload=False, with_vectors=False)
        return len(found) == len(point_ids)
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest VBPL parent/child chunks into local Qdrant")
    parser.add_argument("--data-dir", default="data/keep")
    parser.add_argument("--db-path", default="data/.qdrant", help="Nơi lưu BM25 index (luôn là file cục bộ, kể cả khi dùng --url)")
    parser.add_argument("--url", default=os.getenv("QDRANT_URL"), help="URL Qdrant server (vd http://qdrant:6333). Bỏ trống để dùng embedded/local mode tại --db-path")
    parser.add_argument("--child-collection", default="legal_child_chunks")
    parser.add_argument("--parent-collection", default="legal_parent_chunks")
    parser.add_argument("--batch-size", type=int, default=128, help="Số child chunk encode + upsert mỗi đợt (giữ nhỏ để thấy tiến độ và tránh giữ toàn bộ embedding trong RAM)")
    parser.add_argument("--embed-batch-size", type=int, default=16, help="Batch size nội bộ của model.encode() trong mỗi đợt")
    parser.add_argument("--model", default=DEFAULT_FINETUNED_DIR)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--recreate", action="store_true", help="Delete and rebuild Qdrant collections")
    parser.add_argument("--resume", action="store_true", help="Bỏ qua các batch đã upsert đầy đủ (kiểm tra bằng ID) — dùng khi chạy lại sau khi bị dừng giữa chừng")
    args = parser.parse_args()

    os.makedirs(args.db_path, exist_ok=True)

    client = _make_client(args.db_path, args.url)
    _ensure_child_collection(client, args.child_collection, size=1024, recreate=args.recreate)
    _ensure_parent_collection(client, args.parent_collection, recreate=args.recreate)

    chunker = VBPLChunker()
    result = chunker.process_directory_parent_child(args.data_dir)
    parents = result["parents"]
    children = result["children"]
    print(f"Parsed: {len(parents)} parent chunks (Điều), {len(children)} child chunks (Khoản/Điểm).")

    # ── Parents: vector giả (size=1), rẻ, upsert 1 lần theo batch lớn ──
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
    for batch in _batch(parent_points, 500):
        client.upsert(collection_name=args.parent_collection, points=batch)
    print(f"Upserted {len(parent_points)} parent points.")

    # ── Children: fit BM25 trên TOÀN BỘ corpus trước (rẻ, chỉ đếm từ) ──
    texts = [c["content"] for c in children]
    bm25 = BM25SparseVectorizer().fit(texts)
    bm25.save(bm25_index_path(args.db_path, args.child_collection))
    print(f"BM25 index fit xong trên {len(texts)} chunk, lưu tại {bm25_index_path(args.db_path, args.child_collection)}")

    # ── Encode + upsert THEO TỪNG BATCH NHỎ (không giữ toàn bộ embedding trong RAM,
    #    thấy tiến độ rõ ràng, và --resume bỏ qua batch đã upsert nếu bị dừng giữa chừng) ──
    model = load_embedding_model(args.model, device=args.device, max_seq_length=args.max_seq_length)

    total = len(children)
    done = 0
    skipped = 0
    child_batches = _batch(children, args.batch_size)

    for batch_idx, child_batch in enumerate(child_batches, 1):
        batch_ids = [_point_id(c["id"]) for c in child_batch]

        if args.resume and _already_upserted(client, args.child_collection, batch_ids):
            skipped += len(child_batch)
            done += len(child_batch)
            print(f"  [{batch_idx}/{len(child_batches)}] Bỏ qua (đã upsert trước đó) — {done}/{total}")
            continue

        batch_texts = [c["content"] for c in child_batch]
        embeddings = model.encode(
            batch_texts,
            batch_size=args.embed_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if len(embeddings) > 0 and len(embeddings[0]) != 1024:
            raise RuntimeError(f"Embedding dimension is not 1024 (got {len(embeddings[0])})")

        points = []
        for c, point_id, embedding in zip(child_batch, batch_ids, embeddings):
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
            sparse_indices, sparse_values = bm25.encode_document(payload["content"])
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        "dense": embedding.tolist(),
                        "bm25": models.SparseVector(indices=sparse_indices, values=sparse_values),
                    },
                    payload=payload,
                )
            )

        client.upsert(collection_name=args.child_collection, points=points)
        done += len(child_batch)
        print(f"  [{batch_idx}/{len(child_batches)}] Đã encode + upsert {done}/{total} child chunks")

    print(f"\nHoàn tất: {len(parent_points)} parent, {done} child ({skipped} bỏ qua vì đã có sẵn).")
    print(f"BM25 index: {bm25_index_path(args.db_path, args.child_collection)}")
    print(f"Qdrant: {args.url or args.db_path}")


if __name__ == "__main__":
    main()
