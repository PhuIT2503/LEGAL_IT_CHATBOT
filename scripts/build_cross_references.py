"""
scripts/build_cross_references.py
==================================
Tự động tạo quan hệ THAM_CHIEU giữa Dieu/Khoan/Diem trong Neo4j.

Vì sao cần script này:
  - Pass 2 của entity_extractor (LLM) chỉ thấy entity IDs trong CÙNG 1 chunk
    (1 Điều), nên tham chiếu chéo sang Điều khác thường bị bỏ sót.
  - Node Khoan/Diem trong graph KHÔNG lưu text gốc (bị lược bỏ ở Pass 1 để
    tránh JSON bị truncate) nên không thể quét "mo_ta" của node để tìm
    tham chiếu — cách làm cũ gần như không có dữ liệu để quét.

Cách làm ở đây: đọc lại text GỐC của từng Khoản/Điểm/Điều trực tiếp từ file
.docx (qua VBPLChunker — cùng chunker dùng khi tạo dữ liệu cho LLM), regex
tìm các câu dạng "khoản X (, Y, và Z) Điều N", "Điều N", "Điều này", rồi suy
ra node id đích theo đúng quy ước ID dùng trong neo4j_ingest.py:
    Dieu:  {van_ban_id}_D<so>
    Khoan: {van_ban_id}_D<so>_K<khoan>
    Diem:  {van_ban_id}_D<so>_K<khoan>_P<ky_hieu>  (hoặc không có K nếu Điểm trực thuộc Điều)

Giới hạn đã biết: đây là regex/heuristic, không phải NLU — có thể bỏ sót
cách hành văn không chuẩn hoặc tham chiếu chéo văn bản không nằm trong
VAN_BAN_REGISTRY. Coi đây là bước vá nhanh, không phải giải pháp cuối cùng.
"""

import os
import re
import sys
import logging
import argparse
from pathlib import Path
from collections import Counter

from neo4j import GraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_processing.chunking import VBPLChunker
from knowledge_graph.graph_builder import get_van_ban_id, VAN_BAN_REGISTRY

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

DATA_KEEP_DIR = PROJECT_ROOT / "data" / "keep"

# Tách phần "[Điều X, tên văn bản]" / "[Khoản Y, Điều X, ...]" ở đầu content
# do VBPLChunker chèn vào, để không tự match chính tiêu đề của chunk.
HEADER_PATTERN = re.compile(
    r'^\[(?:Điểm\s+(?P<diem>[a-zđ]),\s*)?(?:Khoản\s+(?P<khoan>\d+),\s*)?'
    r'Điều\s+(?P<dieu>\d+[a-zA-Z]*),[^\]]*\]\s*',
    re.IGNORECASE
)

# "khoản 1, khoản 2 và khoản 3 Điều 19" / "khoản 1, 2 Điều 19" / "khoản này"
KHOAN_DIEU_PATTERN = re.compile(
    r'khoản\s+(?P<khoans>[^\.;]{1,40}?)\s*Điều\s+(?P<dieu>\d+[a-zA-Z]*|này)\b',
    re.IGNORECASE
)

# "Điều 19" độc lập (không đi kèm khoản phía trước)
DIEU_ONLY_PATTERN = re.compile(r'Điều\s+(?P<dieu>\d+[a-zA-Z]*|này)\b', re.IGNORECASE)

# Các từ đứng ngay sau "Điều N" cho biết đây là tham chiếu văn bản KHÁC
# (không phải cùng văn bản đang đọc) — nếu gặp và không phải "...này" thì
# không tạo cạnh cùng-văn-bản để tránh trỏ nhầm.
EXTERNAL_DOC_HINT = re.compile(
    r'^\s*(?:của|thuộc)?\s*(Luật|Nghị định|Bộ luật|Pháp lệnh|Thông tư|Quyết định)\b'
    r'(?!\s+này)',
    re.IGNORECASE
)

# so_hieu -> van_ban_id, dùng để bắt tham chiếu SANG văn bản khác trong corpus
# (vd "Nghị định 15/2020/NĐ-CP" xuất hiện trong nội dung của Nghị định 147/2024)
_SO_HIEU_INDEX = []
for _meta in VAN_BAN_REGISTRY:
    so_hieu = (_meta.get("so_hieu") or "").strip()
    # Chỉ số hiệu dạng "15/2020/NĐ-CP" / "20/2023/QH15" mới đủ đặc trưng để regex match
    if re.match(r'^\d+/\d{4}/', so_hieu):
        van_ban_id = get_van_ban_id(_meta["filename"])
        pattern = re.compile(re.escape(so_hieu), re.IGNORECASE)
        _SO_HIEU_INDEX.append((pattern, van_ban_id, so_hieu))


def _strip_header(content: str) -> str:
    return HEADER_PATTERN.sub('', content, count=1)


def _parse_location(content: str):
    """Đọc [Điểm x, Khoản y, Điều z, ...] ở đầu content để xác định vị trí chunk."""
    m = HEADER_PATTERN.match(content)
    if not m:
        return None
    return {
        "dieu": m.group("dieu"),
        "khoan": m.group("khoan"),
        "diem": m.group("diem"),
    }


def _node_id(van_ban_id: str, dieu: str, khoan: str = None, diem: str = None) -> str:
    node_id = f"{van_ban_id}_D{dieu}"
    if khoan:
        node_id += f"_K{khoan}"
    if diem:
        node_id += f"_P{diem}"
    return node_id


def _split_khoan_numbers(fragment: str) -> list:
    return re.findall(r'\d+', fragment)


class CrossReferenceBuilder:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    # ─────────────────────────────────────────────────────────────────────
    # EXTRACTION: đọc lại text gốc từ .docx và regex tìm tham chiếu
    # ─────────────────────────────────────────────────────────────────────

    def extract_edges_from_docx(self, data_dir: Path = DATA_KEEP_DIR) -> list:
        """
        Quét toàn bộ .docx trong data_dir, trả về list edge dict:
            {"source": node_id, "target": node_id, "kind": "same_doc"|"cross_doc", "note": str}
        """
        if not data_dir.exists():
            logger.error(f"Không tìm thấy thư mục: {data_dir}")
            return []

        chunker = VBPLChunker()
        docx_files = sorted(p for p in data_dir.iterdir() if p.suffix.lower() == ".docx")
        logger.info(f"📄 Quét {len(docx_files)} file .docx trong {data_dir}")

        edges = []
        stats = Counter()

        for file_path in docx_files:
            van_ban_id = get_van_ban_id(file_path.name)
            children = chunker.chunk_document(str(file_path))

            for child in children:
                loc = _parse_location(child["content"])
                if not loc or not loc["dieu"]:
                    continue

                source_id = _node_id(van_ban_id, loc["dieu"], loc["khoan"], loc["diem"])
                body = _strip_header(child["content"])

                consumed_spans = []

                # ── Pass 1: "khoản X[, Y và Z] Điều N|này" ──────────────
                for m in KHOAN_DIEU_PATTERN.finditer(body):
                    consumed_spans.append(m.span())
                    dieu_ref = m.group("dieu")
                    target_dieu = loc["dieu"] if dieu_ref.lower() == "này" else dieu_ref

                    # Tham chiếu tới văn bản khác ngay sau cụm này -> bỏ qua (không đủ tin cậy để suy id)
                    tail = body[m.end():m.end() + 25]
                    if EXTERNAL_DOC_HINT.match(tail):
                        continue

                    khoan_nums = _split_khoan_numbers(m.group("khoans"))
                    if not khoan_nums:
                        continue

                    for khoan_num in khoan_nums:
                        target_id = _node_id(van_ban_id, target_dieu, khoan_num)
                        if target_id == source_id:
                            continue
                        edges.append({
                            "source": source_id, "target": target_id,
                            "kind": "same_doc", "note": "khoan_dieu_regex",
                        })
                        stats["khoan_dieu"] += 1

                # ── Pass 2: "Điều N" hoặc "Điều này" độc lập ────────────
                for m in DIEU_ONLY_PATTERN.finditer(body):
                    span = m.span()
                    if any(span[0] >= s and span[1] <= e for s, e in consumed_spans):
                        continue  # đã xử lý ở pass 1

                    dieu_ref = m.group("dieu")
                    if dieu_ref.lower() == "này":
                        continue  # "Điều này" không có khoản đi kèm -> không thêm thông tin

                    tail = body[m.end():m.end() + 25]
                    if EXTERNAL_DOC_HINT.match(tail):
                        continue  # để pass 3 (so_hieu) xử lý nếu resolve được

                    if dieu_ref == loc["dieu"]:
                        continue  # tự trỏ về chính Điều đang xét -> không có giá trị

                    target_id = _node_id(van_ban_id, dieu_ref)
                    edges.append({
                        "source": source_id, "target": target_id,
                        "kind": "same_doc", "note": "dieu_only_regex",
                    })
                    stats["dieu_only"] += 1

                # ── Pass 3: tham chiếu sang văn bản khác qua số hiệu ────
                for pattern, other_van_ban_id, so_hieu in _SO_HIEU_INDEX:
                    if other_van_ban_id == van_ban_id:
                        continue
                    for m in pattern.finditer(body):
                        window = body[max(0, m.start() - 60):m.end() + 60]
                        dm = DIEU_ONLY_PATTERN.search(window)
                        if not dm or dm.group("dieu").lower() == "này":
                            continue
                        target_id = _node_id(other_van_ban_id, dm.group("dieu"))
                        edges.append({
                            "source": source_id, "target": target_id,
                            "kind": "cross_doc", "note": f"so_hieu:{so_hieu}",
                        })
                        stats["cross_doc"] += 1

        logger.info(
            f"🔎 Regex tìm được {len(edges)} candidate edges "
            f"(khoan_dieu={stats['khoan_dieu']}, dieu_only={stats['dieu_only']}, "
            f"cross_doc={stats['cross_doc']})"
        )
        return edges

    # ─────────────────────────────────────────────────────────────────────
    # INGEST
    # ─────────────────────────────────────────────────────────────────────

    def apply_edges(self, edges: list, batch_size: int = 500) -> int:
        """
        Upsert edges vào Neo4j. Dùng MATCH-MATCH-MERGE nên chỉ tạo cạnh khi
        CẢ HAI node đích đã tồn tại thật trong graph (không tạo placeholder).
        Đánh dấu auto_linked=true để phân biệt với THAM_CHIEU do LLM trích xuất.
        """
        # Dedupe theo (source, target)
        seen = set()
        unique_edges = []
        for e in edges:
            key = (e["source"], e["target"])
            if key in seen:
                continue
            seen.add(key)
            unique_edges.append(e)

        logger.info(f"🔗 Upsert {len(unique_edges)} cạnh THAM_CHIEU duy nhất (sau dedupe)...")

        created = 0
        with self.driver.session() as session:
            for i in range(0, len(unique_edges), batch_size):
                batch = unique_edges[i:i + batch_size]
                query = """
                UNWIND $batch AS edge
                MATCH (s {id: edge.source})
                MATCH (t {id: edge.target})
                MERGE (s)-[r:THAM_CHIEU]->(t)
                SET r.auto_linked = true, r.method = edge.note
                RETURN count(*) AS n
                """
                result = session.run(query, batch=batch)
                record = result.single()
                if record:
                    created += record["n"]
                logger.info(f"  Đã xử lý {min(i + batch_size, len(unique_edges))}/{len(unique_edges)} candidate edges...")

        logger.info(f"✅ {created} cạnh THAM_CHIEU đã được merge (node đích tồn tại thật trong graph).")
        return created

    def build_references(self):
        edges = self.extract_edges_from_docx()
        if not edges:
            logger.info("Không tìm thấy tham chiếu nào từ nội dung .docx.")
            return
        self.apply_edges(edges)

    # ─────────────────────────────────────────────────────────────────────
    # CÁC BƯỚC LÀM SẠCH / SUY DIỄN THÊM (giữ nguyên từ bản trước)
    # ─────────────────────────────────────────────────────────────────────

    def propagate_penalties(self):
        logger.info("⚡ Đang propagate CHE_TAI_BO_SUNG xuống HanhVi...")
        query_propagate = """
        MATCH (c:CheTai)<-[:CHE_TAI_BO_SUNG]-(struc)-[:QUY_DINH_HANH_VI]->(h:HanhVi)
        MERGE (h)-[:CHE_TAI_BO_SUNG]->(c)
        RETURN count(h) AS propagated
        """
        with self.driver.session() as session:
            result = session.run(query_propagate)
            record = result.single()
            if record:
                logger.info(f"✅ Đã propagate thành công CHE_TAI_BO_SUNG cho {record['propagated']} cặp HanhVi-CheTai.")

    def tag_orphans(self):
        logger.info("🧹 Đang tìm và gán nhãn cho Orphan nodes...")
        with self.driver.session() as session:
            # Gỡ nhãn cũ trước — nếu không, node vừa được nối thêm THAM_CHIEU
            # ở bước build_references() vẫn còn giữ nhãn :Orphan từ lần chạy trước.
            session.run("MATCH (n:Orphan) REMOVE n:Orphan")
            result = session.run("""
                MATCH (n) WHERE count{(n)--()} = 0
                SET n:Orphan
                RETURN count(n) AS orphans
            """)
            record = result.single()
            if record:
                logger.info(f"✅ Đã gán nhãn :Orphan cho {record['orphans']} nodes mồ côi.")


def main():
    parser = argparse.ArgumentParser(description="Auto-link THAM_CHIEU từ nội dung .docx gốc")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ in candidate edges, không ghi vào Neo4j")
    parser.add_argument("--data-dir", default=str(DATA_KEEP_DIR), help="Thư mục chứa .docx")
    parser.add_argument("--van-ban", default=None, help="Lọc candidate edges theo van_ban_id (dry-run)")
    parser.add_argument("--limit", type=int, default=60, help="Số dòng tối đa in ra (dry-run)")
    args = parser.parse_args()

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "legal_kg_2024")

    logger.info("=== BẮT ĐẦU TỰ ĐỘNG NỐI THAM CHIẾU VÀ LÀM SẠCH GRAPH ===")

    if args.dry_run:
        builder = CrossReferenceBuilder.__new__(CrossReferenceBuilder)  # không cần kết nối Neo4j
        edges = builder.extract_edges_from_docx(Path(args.data_dir))
        if args.van_ban:
            edges = [e for e in edges if e["source"].startswith(args.van_ban)]
        for e in edges[:args.limit]:
            print(f"  {e['source']:<55} --[{e['note']}]--> {e['target']}")
        print(f"\nTổng cộng: {len(edges)} candidate edges (in tối đa {args.limit} dòng ở trên).")
        return

    builder = CrossReferenceBuilder(uri, user, password)
    try:
        builder.build_references()
        builder.propagate_penalties()
        builder.tag_orphans()
    except Exception as e:
        logger.error(f"Lỗi: {e}")
        raise
    finally:
        builder.close()


if __name__ == "__main__":
    main()
