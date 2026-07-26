"""Backfill ``metadata.legal_domains`` cho Qdrant đã ingest từ trước.

Script chỉ cập nhật payload, không encode lại văn bản và không thay vector/BM25.
Có thể chạy lại an toàn khi registry domain được mở rộng.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from qdrant_client import QdrantClient

from src.retrieval.legal_domains import document_legal_domains


def _make_client(db_path: str, url: str | None) -> QdrantClient:
    return QdrantClient(url=url) if url else QdrantClient(path=db_path)


def backfill_collection(
    client: QdrantClient,
    collection_name: str,
    *,
    page_size: int = 256,
) -> dict[str, Any]:
    """Cập nhật payload theo batch domain và trả thống kê migration."""

    groups: dict[tuple[str, ...], list[Any]] = defaultdict(list)
    sources: dict[tuple[str, ...], set[str]] = defaultdict(set)
    offset = None
    scanned = 0

    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            limit=page_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            metadata = payload.get("metadata") or {}
            source = str(metadata.get("source") or "")
            domains = tuple(document_legal_domains(source))
            groups[domains].append(point.id)
            sources[domains].add(source)
            scanned += 1
        if next_offset is None:
            break
        offset = next_offset

    for domains, point_ids in groups.items():
        client.set_payload(
            collection_name=collection_name,
            payload={"legal_domains": list(domains)},
            points=point_ids,
            key="metadata",
            wait=True,
        )

    return {
        "collection": collection_name,
        "scanned": scanned,
        "updated": sum(len(ids) for ids in groups.values()),
        "domain_groups": {
            ",".join(domains): {
                "points": len(groups[domains]),
                "sources": sorted(source for source in sources[domains] if source),
            }
            for domains in sorted(groups)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill legal_domains cho payload Qdrant hiện có"
    )
    parser.add_argument("--db-path", default="data/.qdrant_base")
    parser.add_argument("--url", default=os.getenv("QDRANT_URL"))
    parser.add_argument(
        "--collections",
        nargs="+",
        default=["legal_child_chunks", "legal_parent_chunks"],
    )
    parser.add_argument("--page-size", type=int, default=256)
    args = parser.parse_args()

    client = _make_client(args.db_path, args.url)
    existing = {item.name for item in client.get_collections().collections}
    try:
        for collection in args.collections:
            if collection not in existing:
                print(f"Skip {collection}: collection không tồn tại")
                continue
            stats = backfill_collection(
                client, collection, page_size=max(1, args.page_size)
            )
            print(stats)
    finally:
        client.close()


if __name__ == "__main__":
    main()
