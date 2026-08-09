"""
scripts/migrate_qdrant_to_server.py
===================================
Chuyển 2 index Qdrant embedded (đang nằm dưới dạng THƯ MỤC trong repo:
data/.qdrant_base và data/.qdrant_gte_base) sang MỘT Qdrant server chạy bằng
Docker (service `qdrant` trong docker-compose.yml, cổng 6333 ở máy host).

Vì cả 2 thư mục đều chứa CÙNG tên collection (legal_child_chunks +
legal_parent_chunks) nên khi gộp vào 1 server phải đặt tên khác nhau — thêm
hậu tố theo model embedding:

    data/.qdrant_base/      legal_child_chunks   ->  legal_child_chunks_base
    (Vietnamese_Embedding_v2, 1024d)
                            legal_parent_chunks  ->  legal_parent_chunks_base

    data/.qdrant_gte_base/  legal_child_chunks   ->  legal_child_chunks_gte
    (gte-multilingual-base, 768d)
                            legal_parent_chunks  ->  legal_parent_chunks_gte

Script copy nguyên trạng: cấu hình collection (dense + sparse "bm25"), toàn bộ
point ID, vector (dense + sparse) và payload — KHÔNG encode lại bằng embedding
model, nên chạy nhanh (vài phút) và kết quả retrieval y hệt trước khi chuyển.

BM25 index (file .json cạnh thư mục Qdrant, hybrid search bắt buộc phải có)
được copy sang data/bm25/<tên collection mới>.bm25.json để sau khi xóa 2 thư
mục .qdrant_* vẫn còn.

Cách chạy (khuyến nghị — không cần cài Python trên máy host):
    docker compose up -d qdrant
    docker compose --profile app run --rm --no-deps qdrant-ingest python scripts/migrate_qdrant_to_server.py

Chạy trực tiếp trên máy có sẵn Python + qdrant-client:
    python scripts/migrate_qdrant_to_server.py --url http://localhost:6333

Chạy lại lần 2 (ghi đè dữ liệu đã có trên server): thêm --recreate.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qdrant_client import QdrantClient, models

# Mỗi nguồn = 1 thư mục Qdrant embedded ứng với 1 model embedding. suffix chính
# là hậu tố gắn vào tên collection trên server để 2 không gian embedding KHÔNG
# lẫn vào nhau (khác dimension: 1024 vs 768).
SOURCES = [
    {
        "path": "data/.qdrant_base",
        "suffix": "base",
        "label": "AITeamVN/Vietnamese_Embedding_v2 (1024d)",
    },
    {
        "path": "data/.qdrant_gte_base",
        "suffix": "gte",
        "label": "Alibaba-NLP/gte-multilingual-base (768d)",
    },
]

BASE_COLLECTIONS = ("legal_child_chunks", "legal_parent_chunks")


def target_name(base_collection: str, suffix: str) -> str:
    return f"{base_collection}_{suffix}"


def _existing(client: QdrantClient) -> List[str]:
    return [c.name for c in client.get_collections().collections]


def _create_target(dst: QdrantClient, name: str, src_info: Any, recreate: bool) -> bool:
    """Tạo collection đích với ĐÚNG cấu hình của collection nguồn.

    Trả về False nếu collection đã tồn tại và không yêu cầu --recreate (bỏ qua,
    không ghi đè dữ liệu sẵn có trên server).
    """
    params = src_info.config.params
    if name in _existing(dst):
        if not recreate:
            return False
        dst.delete_collection(collection_name=name)

    dst.create_collection(
        collection_name=name,
        vectors_config=params.vectors,
        sparse_vectors_config=params.sparse_vectors,
    )
    return True


def _copy_points(src: QdrantClient, dst: QdrantClient, src_col: str, dst_col: str, batch_size: int) -> int:
    """Scroll toàn bộ point (kèm vector + payload) từ local rồi upsert sang server."""
    offset: Optional[Any] = None
    total = 0
    while True:
        points, offset = src.scroll(
            collection_name=src_col,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            break

        dst.upsert(
            collection_name=dst_col,
            points=[
                models.PointStruct(id=point.id, vector=point.vector, payload=point.payload)
                for point in points
            ],
            wait=True,
        )
        total += len(points)
        print(f"      đã copy {total} point...", flush=True)

        if offset is None:
            break
    return total


def _copy_bm25(src_dir: Path, base_collection: str, dst_dir: Path, dst_collection: str) -> Optional[Path]:
    src_file = src_dir / f"{base_collection}.bm25.json"
    if not src_file.exists():
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_file = dst_dir / f"{dst_collection}.bm25.json"
    shutil.copyfile(src_file, dst_file)
    return dst_file


def migrate_source(
    source: Dict[str, str],
    dst: QdrantClient,
    bm25_dir: Path,
    batch_size: int,
    recreate: bool,
) -> None:
    src_dir = PROJECT_ROOT / source["path"]
    if not src_dir.exists():
        print(f"[BỎ QUA] Không thấy thư mục {src_dir}")
        return

    print(f"\n=== {source['path']}  ->  hậu tố _{source['suffix']}  ({source['label']}) ===")
    src = QdrantClient(path=str(src_dir))
    try:
        for base_collection in BASE_COLLECTIONS:
            dst_col = target_name(base_collection, source["suffix"])
            src_info = src.get_collection(base_collection)
            src_count = src.count(collection_name=base_collection, exact=True).count
            print(f"  {base_collection} ({src_count} point) -> {dst_col}")

            if not _create_target(dst, dst_col, src_info, recreate):
                dst_count = dst.count(collection_name=dst_col, exact=True).count
                print(f"    [BỎ QUA] Collection {dst_col} đã có sẵn trên server ({dst_count} point). "
                      f"Dùng --recreate nếu muốn nạp lại từ đầu.")
                continue

            copied = _copy_points(src, dst, base_collection, dst_col, batch_size)
            dst_count = dst.count(collection_name=dst_col, exact=True).count
            status = "OK" if dst_count == src_count else "LỆCH SỐ LƯỢNG!"
            print(f"    {status} — nguồn {src_count}, đích {dst_count} (copy {copied})")

            if base_collection == "legal_child_chunks":
                bm25_file = _copy_bm25(src_dir, base_collection, bm25_dir, dst_col)
                if bm25_file:
                    print(f"    BM25 index -> {bm25_file.relative_to(PROJECT_ROOT)}")
                else:
                    print(f"    [CẢNH BÁO] Không thấy {base_collection}.bm25.json trong {src_dir} — "
                          f"hybrid search sẽ chỉ còn dense vector.")
    finally:
        src.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chuyển Qdrant embedded (thư mục cục bộ) sang Qdrant server chạy bằng Docker"
    )
    parser.add_argument(
        "--url",
        default=os.getenv("QDRANT_URL", "http://localhost:6333"),
        help="URL Qdrant server đích (mặc định http://localhost:6333, trong container dùng http://qdrant:6333)",
    )
    parser.add_argument("--bm25-dir", default="data/bm25", help="Thư mục lưu BM25 index sau khi chuyển")
    parser.add_argument("--batch-size", type=int, default=256, help="Số point mỗi lần scroll + upsert")
    parser.add_argument("--recreate", action="store_true", help="Xóa và nạp lại collection nếu đã tồn tại trên server")
    parser.add_argument(
        "--only",
        choices=[s["suffix"] for s in SOURCES],
        default=None,
        help="Chỉ chuyển 1 nguồn (base hoặc gte) thay vì cả 2",
    )
    args = parser.parse_args()

    bm25_dir = PROJECT_ROOT / args.bm25_dir
    # timeout cao vì mỗi batch upsert kèm vector 1024 chiều có thể mất vài giây.
    dst = QdrantClient(url=args.url, timeout=300)
    print(f"Qdrant server đích: {args.url}")

    sources = [s for s in SOURCES if args.only is None or s["suffix"] == args.only]

    # Không thấy thư mục nguồn nào -> DỪNG với mã lỗi, đừng "thành công" trong
    # im lặng. Máy vừa clone repo về chắc chắn rơi vào nhánh này (2 thư mục
    # .qdrant_* nằm trong .gitignore, không đi theo repo) — trước đây script
    # chỉ in "[BỎ QUA]" rồi thoát 0, khiến người chạy tưởng đã nạp xong, tới
    # lúc chat mới thấy Qdrant rỗng và bot trả "Tôi không biết".
    missing = [s["path"] for s in sources if not (PROJECT_ROOT / s["path"]).exists()]
    if len(missing) == len(sources):
        print("LỖI: không thấy thư mục Qdrant embedded nào để chuyển:")
        for path in missing:
            print(f"  - {path}")
        print(
            "\nĐây là lối tắt CHỈ dùng được trên máy đã có sẵn 2 thư mục đó.\n"
            "Máy mới: bỏ qua script này, dùng service qdrant-ingest để chunk +\n"
            "embed + upsert thẳng từ data/keep:\n"
            "    docker compose --profile app up qdrant-ingest"
        )
        raise SystemExit(1)

    for source in sources:
        migrate_source(source, dst, bm25_dir, args.batch_size, args.recreate)

    print("\n=== Collection hiện có trên server ===")
    for name in sorted(_existing(dst)):
        print(f"  {name}: {dst.count(collection_name=name, exact=True).count} point")


if __name__ == "__main__":
    main()
