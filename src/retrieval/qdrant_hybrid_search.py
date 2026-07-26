import argparse
import os
import sys
import textwrap
import uuid
from pathlib import Path
from typing import Any, Dict, List

# Bootstrap sys.path để chạy được cả trực tiếp (python src/retrieval/qdrant_hybrid_search.py)
# lẫn khi import như package (from src.retrieval.qdrant_hybrid_search import ...)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from qdrant_client import QdrantClient, models

from src.retrieval.bm25_sparse import BM25SparseVectorizer, bm25_index_path
from src.llm.embedding_model import DEFAULT_FINETUNED_DIR, load_embedding_model


def _point_id(raw_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw_id))


def _make_client(db_path: str, url: str | None) -> QdrantClient:
    if url:
        return QdrantClient(url=url)
    return QdrantClient(path=db_path)


def _shorten(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _points(response: Any) -> List[Any]:
    return getattr(response, "points", response)


def _fusion_query(name: str) -> Any:
    normalized = name.lower()
    if normalized == "dbsf":
        return models.FusionQuery(fusion=models.Fusion.DBSF)
    return models.FusionQuery(fusion=models.Fusion.RRF)


def _hybrid_search(
    query: str,
    child_collection: str,
    parent_collection: str,
    limit: int,
    prefetch_limit: int,
    fusion: str,
    include_parent: bool,
    db_path: str | None = None,
    url: str | None = None,
    model_name: str | None = None,
    device: str | None = None,
    max_seq_length: int | None = None,
    client: "QdrantClient | None" = None,
    model: "Any | None" = None,
    bm25: "BM25SparseVectorizer | None" = None,
    legal_domains: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Cho phép truyền sẵn client/model/bm25 (dùng khi gọi lặp lại nhiều query trong
    1 phiên, vd chatbot pipeline) để tránh load lại embedding model mỗi lần gọi.
    Không truyền -> tự tạo mới từ db_path/url/model_name (hành vi CLI cũ).
    """
    if client is None:
        client = _make_client(db_path, url)
    if model is None:
        model = load_embedding_model(model_name or DEFAULT_FINETUNED_DIR, device=device, max_seq_length=max_seq_length)
    if bm25 is None:
        bm25 = BM25SparseVectorizer.load(bm25_index_path(db_path, child_collection))

    dense_vector = model.encode([query], normalize_embeddings=True)[0].tolist()
    sparse_indices, sparse_values = bm25.encode_query(query)

    domain_filter = None
    if legal_domains:
        domain_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.legal_domains",
                    match=models.MatchAny(any=list(dict.fromkeys(legal_domains))),
                )
            ]
        )

    prefetch = [
        models.Prefetch(
            query=dense_vector,
            using="dense",
            limit=prefetch_limit,
            filter=domain_filter,
        ),
    ]
    if sparse_indices:
        prefetch.insert(
            0,
            models.Prefetch(
                query=models.SparseVector(indices=sparse_indices, values=sparse_values),
                using="bm25",
                limit=prefetch_limit,
                filter=domain_filter,
            ),
        )

    response = client.query_points(
        collection_name=child_collection,
        prefetch=prefetch,
        query=_fusion_query(fusion),
        limit=limit,
        with_payload=True,
    )
    child_hits = _points(response)

    parents: Dict[str, Dict[str, Any]] = {}
    if include_parent:
        parent_ids = []
        seen = set()
        for point in child_hits:
            payload = point.payload or {}
            parent_id = payload.get("parent_id")
            if parent_id and parent_id not in seen:
                parent_ids.append(parent_id)
                seen.add(parent_id)

        if parent_ids:
            parent_points = client.retrieve(
                collection_name=parent_collection,
                ids=[_point_id(parent_id) for parent_id in parent_ids],
                with_payload=True,
            )
            parents = {
                (point.payload or {}).get("id", ""): point.payload or {}
                for point in parent_points
                if point.payload
            }

    return {
        "query": query,
        "children": child_hits,
        "parents": parents,
        "sparse_terms": len(sparse_indices),
    }


def hybrid_search(
    query: str,
    child_collection: str,
    parent_collection: str,
    limit: int,
    prefetch_limit: int,
    fusion: str,
    include_parent: bool,
    db_path: str | None = None,
    url: str | None = None,
    model_name: str | None = None,
    device: str | None = None,
    max_seq_length: int | None = None,
    client: "QdrantClient | None" = None,
    model: "Any | None" = None,
    bm25: "BM25SparseVectorizer | None" = None,
) -> Dict[str, Any]:
    """Public API cũ: hybrid search không áp domain filter."""

    return _hybrid_search(
        query=query,
        child_collection=child_collection,
        parent_collection=parent_collection,
        limit=limit,
        prefetch_limit=prefetch_limit,
        fusion=fusion,
        include_parent=include_parent,
        db_path=db_path,
        url=url,
        model_name=model_name,
        device=device,
        max_seq_length=max_seq_length,
        client=client,
        model=model,
        bm25=bm25,
        legal_domains=None,
    )


def hybrid_search_in_domains(
    query: str,
    child_collection: str,
    parent_collection: str,
    limit: int,
    prefetch_limit: int,
    fusion: str,
    include_parent: bool,
    legal_domains: List[str],
    db_path: str | None = None,
    url: str | None = None,
    model_name: str | None = None,
    device: str | None = None,
    max_seq_length: int | None = None,
    client: "QdrantClient | None" = None,
    model: "Any | None" = None,
    bm25: "BM25SparseVectorizer | None" = None,
) -> Dict[str, Any]:
    """Internal Phase 1 entrypoint: cùng search cũ nhưng giới hạn domain."""

    return _hybrid_search(
        query=query,
        child_collection=child_collection,
        parent_collection=parent_collection,
        limit=limit,
        prefetch_limit=prefetch_limit,
        fusion=fusion,
        include_parent=include_parent,
        db_path=db_path,
        url=url,
        model_name=model_name,
        device=device,
        max_seq_length=max_seq_length,
        client=client,
        model=model,
        bm25=bm25,
        legal_domains=legal_domains,
    )


def print_results(result: Dict[str, Any], max_chars: int, include_parent: bool) -> None:
    print(f"Query: {result['query']}")
    print(f"BM25 sparse terms matched: {result['sparse_terms']}")
    print()

    for index, point in enumerate(result["children"], start=1):
        payload = point.payload or {}
        print(f"#{index} score={point.score:.6f}")
        print(f"id: {payload.get('id')}")
        print(f"source: {(payload.get('metadata') or {}).get('source', '')}")
        print(f"type: {payload.get('chunk_type')} | dieu_id: {payload.get('dieu_id')}")
        print(textwrap.fill(_shorten(payload.get("content", ""), max_chars), width=100))

        if include_parent:
            parent = result["parents"].get(payload.get("parent_id"))
            if parent:
                print("Parent:")
                print(textwrap.fill(_shorten(parent.get("content", ""), max_chars), width=100))
        print("-" * 100)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid search over local Qdrant VBPL chunks")
    parser.add_argument("--query", default=None)
    parser.add_argument("--db-path", default="data/.qdrant", help="Nơi đọc BM25 index (luôn là file cục bộ, kể cả khi dùng --url)")
    parser.add_argument("--url", default=os.getenv("QDRANT_URL"), help="URL Qdrant server (vd http://qdrant:6333). Bỏ trống để dùng embedded/local mode tại --db-path")
    parser.add_argument("--child-collection", default="legal_child_chunks")
    parser.add_argument("--parent-collection", default="legal_parent_chunks")
    parser.add_argument("--model", default=DEFAULT_FINETUNED_DIR)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--prefetch-limit", type=int, default=30)
    parser.add_argument("--fusion", choices=["rrf", "dbsf"], default="rrf")
    parser.add_argument("--include-parent", action="store_true")
    parser.add_argument("--max-chars", type=int, default=900)
    args = parser.parse_args()

    query = args.query or input("Nhập câu hỏi: ").strip()
    if not query:
        raise SystemExit("Query không được để trống")

    result = hybrid_search(
        query=query,
        db_path=args.db_path,
        url=args.url,
        child_collection=args.child_collection,
        parent_collection=args.parent_collection,
        model_name=args.model,
        device=args.device,
        max_seq_length=args.max_seq_length,
        limit=args.limit,
        prefetch_limit=args.prefetch_limit,
        fusion=args.fusion,
        include_parent=args.include_parent,
    )
    print_results(result, max_chars=args.max_chars, include_parent=args.include_parent)


if __name__ == "__main__":
    main()
