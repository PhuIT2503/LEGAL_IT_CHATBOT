"""
scripts/run_neo4j_ingest.py
===========================
Script chạy import toàn bộ dữ liệu từ data/extracted_json/ vào Neo4j Docker.

Cách dùng:
    # 1. Khởi động Neo4j container
    docker compose up -d

    # 2. Đợi Neo4j sẵn sàng (~30 giây), rồi chạy:
    python scripts/run_neo4j_ingest.py

    # Xóa sạch graph và import lại từ đầu:
    python scripts/run_neo4j_ingest.py --clear

Kết nối:
    Neo4j Browser: http://localhost:7474
    Bolt URI     : bolt://localhost:7687
    Auth         : neo4j / legal_kg_2024
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# ── Thêm thư mục gốc project vào sys.path ─────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.knowledge_graph.neo4j_ingest import Neo4jGraphIngestor

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "legal_kg_2024")
DATA_DIR       = PROJECT_ROOT / "data" / "extracted_json"


# ── Helpers ────────────────────────────────────────────────────────────────

def wait_for_neo4j(ingestor: Neo4jGraphIngestor, max_retries: int = 10, delay: int = 5):
    """Đợi Neo4j container sẵn sàng nhận kết nối."""
    logger.info("Đang chờ Neo4j sẵn sàng...")
    for attempt in range(1, max_retries + 1):
        if ingestor.verify_connection():
            logger.info("✓ Neo4j đã sẵn sàng!")
            return True
        logger.info(f"  Thử lần {attempt}/{max_retries}, chờ {delay}s...")
        time.sleep(delay)
    logger.error("✗ Không kết nối được Neo4j sau nhiều lần thử.")
    return False


def load_jsonl_file(file_path: Path) -> tuple[list[dict], list[dict]]:
    """
    Đọc một file .jsonl và trả về (entities, relations).
    Mỗi dòng có dạng: {"entities": [...], "relations": [...], "chunk_id": "...", ...}
    """
    entities: list[dict] = []
    relations: list[dict] = []
    error_count = 0

    with open(file_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                chunk_id  = data.get("chunk_id", "")
                van_ban_id = data.get("van_ban_id", "")

                # Gắn thêm chunk_id và van_ban_id vào mỗi entity (nếu chưa có)
                for ent in data.get("entities", []):
                    if chunk_id and "chunk_id" not in ent:
                        ent["chunk_id"] = chunk_id
                    if van_ban_id and "van_ban_id" not in ent:
                        ent["van_ban_id"] = van_ban_id
                    entities.append(ent)

                # Gắn thêm van_ban_id vào mỗi relation
                for rel in data.get("relations", []):
                    if van_ban_id and "van_ban_id" not in rel:
                        rel["van_ban_id"] = van_ban_id
                    relations.append(rel)

            except json.JSONDecodeError as e:
                error_count += 1
                logger.warning(f"  ⚠ JSON error dòng {line_no} trong {file_path.name}: {e}")

    if error_count:
        logger.warning(f"  {error_count} dòng lỗi bị bỏ qua trong {file_path.name}")

    return entities, relations


def process_all_files(ingestor: Neo4jGraphIngestor):
    """Tìm và xử lý toàn bộ file .jsonl trong DATA_DIR."""
    jsonl_files = sorted(DATA_DIR.rglob("*.jsonl"))

    if not jsonl_files:
        logger.error(f"Không tìm thấy file .jsonl nào trong: {DATA_DIR}")
        return

    logger.info(f"Tìm thấy {len(jsonl_files)} file JSONL cần import:")
    for f in jsonl_files:
        logger.info(f"  • {f.relative_to(PROJECT_ROOT)}")

    total_entities  = 0
    total_relations = 0

    for idx, file_path in enumerate(jsonl_files, 1):
        logger.info(f"\n[{idx}/{len(jsonl_files)}] Xử lý: {file_path.name}")
        entities, relations = load_jsonl_file(file_path)

        logger.info(f"  → {len(entities)} entities, {len(relations)} relations")

        if entities:
            ingestor.upsert_nodes(entities)
        if relations:
            ingestor.upsert_relationships(relations)

        total_entities  += len(entities)
        total_relations += len(relations)

    logger.info(
        f"\n{'='*50}\n"
        f"✓ Hoàn thành! Tổng cộng:\n"
        f"  - {total_entities:,} entities đã upsert\n"
        f"  - {total_relations:,} relations đã upsert\n"
        f"{'='*50}"
    )


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Import Legal Knowledge Graph vào Neo4j Docker"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Xóa sạch toàn bộ graph trước khi import (CẢNH BÁO: không hoàn tác được!)",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Chỉ in thống kê graph, không import gì cả",
    )
    args = parser.parse_args()

    if not DATA_DIR.exists():
        logger.error(f"Thư mục data không tồn tại: {DATA_DIR}")
        sys.exit(1)

    logger.info(f"Neo4j URI : {NEO4J_URI}")
    logger.info(f"Data Dir  : {DATA_DIR}")

    with Neo4jGraphIngestor(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD,
    ) as ingestor:

        # Đợi Neo4j sẵn sàng
        if not wait_for_neo4j(ingestor):
            sys.exit(1)

        if args.stats_only:
            stats = ingestor.get_stats()
            logger.info("=== Graph Stats ===")
            logger.info(f"Total nodes    : {stats['total_nodes']:,}")
            logger.info(f"Total relations: {stats['total_relations']:,}")
            logger.info("Nodes by label :")
            for label, count in stats["nodes"].items():
                logger.info(f"  {label:<15}: {count:,}")
            logger.info("Relations by type:")
            for rtype, count in stats["relations"].items():
                logger.info(f"  {rtype:<25}: {count:,}")
            return

        if args.clear:
            confirm = input(
                "⚠ Bạn có chắc muốn XÓA TOÀN BỘ graph? Nhập 'yes' để xác nhận: "
            ).strip()
            if confirm.lower() != "yes":
                logger.info("Hủy thao tác xóa.")
                sys.exit(0)
            ingestor.clear_all()

        # Setup schema (indexes, constraints)
        ingestor.setup_schema()

        # Import tất cả files
        process_all_files(ingestor)

        # In thống kê cuối
        stats = ingestor.get_stats()
        logger.info("=== Graph Stats sau import ===")
        logger.info(f"Total nodes    : {stats['total_nodes']:,}")
        logger.info(f"Total relations: {stats['total_relations']:,}")


if __name__ == "__main__":
    main()
