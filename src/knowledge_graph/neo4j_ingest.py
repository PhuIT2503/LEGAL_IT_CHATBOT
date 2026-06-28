"""
neo4j_ingest.py
===============
Upsert Knowledge Graph vào Neo4j (Docker container).
Sử dụng MERGE để idempotent — chạy lại không bị duplicate.

Schema nodes:
  VanBan, Chuong, Dieu, Khoan, Diem,
  HanhVi, CheTai, ChuThe, NghiaVu, QuyenHan, KhaiNiem, DieuKien

Schema relations:
  CO_CHUONG, CO_DIEU, CO_KHOAN, CO_DIEM,
  THAM_CHIEU, QUY_DINH_HANH_VI,
  CHE_TAI_CHINH, CHE_TAI_BO_SUNG,
  AP_DUNG_VOI, CO_NGHIA_VU, CO_QUYEN,
  DINH_NGHIA, CO_DIEU_KIEN, SUA_DOI, HUONG_DAN
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA_QUERIES = [
    # Unique constraints
    "CREATE CONSTRAINT van_ban_id   IF NOT EXISTS FOR (n:VanBan)   REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT chuong_id    IF NOT EXISTS FOR (n:Chuong)   REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT dieu_id      IF NOT EXISTS FOR (n:Dieu)     REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT khoan_id     IF NOT EXISTS FOR (n:Khoan)    REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT diem_id      IF NOT EXISTS FOR (n:Diem)     REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT hanh_vi_id   IF NOT EXISTS FOR (n:HanhVi)   REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT che_tai_id   IF NOT EXISTS FOR (n:CheTai)   REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT chu_the_id   IF NOT EXISTS FOR (n:ChuThe)   REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT nghia_vu_id  IF NOT EXISTS FOR (n:NghiaVu)  REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT quyen_han_id IF NOT EXISTS FOR (n:QuyenHan) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT khai_niem_id IF NOT EXISTS FOR (n:KhaiNiem) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT dieu_kien_id IF NOT EXISTS FOR (n:DieuKien) REQUIRE n.id IS UNIQUE",

    # Full-text indexes để critic agent / tìm kiếm nhanh
    "CREATE FULLTEXT INDEX hanh_vi_search  IF NOT EXISTS FOR (n:HanhVi)   ON EACH [n.mo_ta]",
    "CREATE FULLTEXT INDEX che_tai_search  IF NOT EXISTS FOR (n:CheTai)   ON EACH [n.mo_ta]",
    "CREATE FULLTEXT INDEX khai_niem_search IF NOT EXISTS FOR (n:KhaiNiem) ON EACH [n.ten, n.dinh_nghia]",
    "CREATE FULLTEXT INDEX dieu_search     IF NOT EXISTS FOR (n:Dieu)     ON EACH [n.ten]",

    # Index thường để tra cứu theo chunk_id
    "CREATE INDEX idx_chunk_id IF NOT EXISTS FOR (n:Dieu)   ON (n.chunk_id)",
    "CREATE INDEX idx_van_ban  IF NOT EXISTS FOR (n:Dieu)   ON (n.van_ban_id)",
]

# Map type string → Neo4j Label
# Bao gồm các variant spelling từ QWen extractor
TYPE_TO_LABEL: dict[str, str] = {
    "VanBan":     "VanBan",
    "Chuong":     "Chuong",
    "Dieu":       "Dieu",
    "Khoan":      "Khoan",
    "Diem":       "Diem",
    "HanhVi":     "HanhVi",
    "CheTai":     "CheTai",
    "ChuThe":     "ChuThe",
    "NghiaVu":    "NghiaVu",
    "NghiemVu":   "NghiaVu",   # QWen variant spelling
    "NghiaVu2":   "NghiaVu",   # QWen variant
    "QuyenHan":   "QuyenHan",
    "KhaiNiem":   "KhaiNiem",
    "DieuKien":   "DieuKien",
}

# Relation types hợp lệ (whitelist để tránh Cypher injection)
VALID_RELATION_TYPES = {
    "CO_CHUONG", "CO_DIEU", "CO_KHOAN", "CO_DIEM",
    "THAM_CHIEU", "QUY_DINH_HANH_VI",
    "CHE_TAI_CHINH", "CHE_TAI_BO_SUNG",
    "AP_DUNG_VOI", "CO_NGHIA_VU", "CO_QUYEN",
    "DINH_NGHIA", "CO_DIEU_KIEN", "SUA_DOI", "HUONG_DAN",
}

# Map các relation type sai/dị thường từ QWen → type chuẩn
# Nếu không có trong map này và không có trong VALID_RELATION_TYPES → bị skip
RELATION_TYPE_NORMALIZE: dict[str, str] = {
    "KHAI_NIEM":                   "DINH_NGHIA",
    "CHE_TAI_MINH_BAC_THONG_TIN": "CHE_TAI_BO_SUNG",
    "CT":                          "CHE_TAI_CHINH",
    "CO_CHE_TAI":                  "CHE_TAI_CHINH",
    "CO_HANH_VI":                  "QUY_DINH_HANH_VI",
    "LIEN_QUAN":                   "THAM_CHIEU",
    "THAM_KHAO":                   "THAM_CHIEU",
}


# ─────────────────────────────────────────────────────────────────────────────
# INGESTOR CLASS
# ─────────────────────────────────────────────────────────────────────────────

class Neo4jGraphIngestor:
    """
    Ingest Legal Knowledge Graph vào Neo4j Docker container.

    Dữ liệu đầu vào (từ extracted_results.jsonl):
      - entities: list[dict] với keys: id, type, + các props tuỳ loại
      - relations: list[dict] với keys: source_id, relation_type, target_id
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "legal_kg_2024",
        database: str = "neo4j",
    ):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        logger.info(f"Connected to Neo4j: {uri}")

    def close(self):
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def run_query(self, query: str, params: Optional[dict] = None) -> list:
        """Chạy Cypher query và trả về danh sách kết quả."""
        with self.driver.session(database=self.database) as session:
            result = session.run(query, params or {})
            return list(result)

    # ─────────────────────────────────────────────────────────────────────────
    # SCHEMA
    # ─────────────────────────────────────────────────────────────────────────

    def setup_schema(self):
        """Tạo constraints và indexes. Idempotent — chạy nhiều lần không sao."""
        logger.info("Setting up Neo4j schema...")
        for query in SCHEMA_QUERIES:
            try:
                self.run_query(query)
                logger.debug(f"  OK: {query[:70]}...")
            except Exception as e:
                logger.warning(f"  Schema warning (có thể đã tồn tại): {e}")
        logger.info("Schema setup complete.")

    def clear_all(self):
        """Xóa toàn bộ graph. CẢNH BÁO: không thể hoàn tác!"""
        logger.warning("⚠ Xóa toàn bộ graph...")
        self.run_query("MATCH (n) DETACH DELETE n")
        logger.info("Graph đã được xóa.")

    # ─────────────────────────────────────────────────────────────────────────
    # NODE UPSERT
    # ─────────────────────────────────────────────────────────────────────────

    def upsert_node(self, node_data: dict):
        """
        Upsert một node. Dùng MERGE nên idempotent.
        node_data phải có 'id' và 'type'.
        """
        node_type = node_data.get("type", "")
        label = TYPE_TO_LABEL.get(node_type)
        node_id = node_data.get("id", "")

        if not label or not node_id:
            logger.warning(f"Skip node (thiếu id hoặc type không hợp lệ): {node_data}")
            return

        # Properties: loại bỏ 'type', chỉ giữ scalar values
        props = {
            k: v for k, v in node_data.items()
            if k != "type" and v is not None
            and isinstance(v, (str, int, float, bool))
        }

        van_ban_id = str(node_data.get("van_ban_id", ""))
        if van_ban_id and node_type != "VanBan" and not str(node_id).startswith(van_ban_id):
            node_id = f"{van_ban_id}_{node_id}"
            props["id"] = node_id

        query = f"""
        MERGE (n:{label} {{id: $id}})
        SET n += $props
        """
        try:
            self.run_query(query, {"id": node_id, "props": props})
        except Exception as e:
            logger.error(f"Failed upsert node [{node_id}]: {e}")

    def upsert_nodes(self, nodes: list[dict], batch_size: int = 200):
        """
        Batch upsert danh sách nodes.
        Alias: upsert_nodes_batch.
        """
        total = len(nodes)
        logger.info(f"Upserting {total} nodes...")
        for i in range(0, total, batch_size):
            batch = nodes[i : i + batch_size]
            for node in batch:
                self.upsert_node(node)
            logger.info(f"  Nodes: {min(i + batch_size, total)}/{total}")

    # Alias để tương thích với code cũ
    upsert_nodes_batch = upsert_nodes

    # ─────────────────────────────────────────────────────────────────────────
    # RELATIONSHIP UPSERT
    # ─────────────────────────────────────────────────────────────────────────

    def upsert_relationship(self, rel_data: dict):
        """
        Upsert một relationship. Dùng MERGE nên idempotent.

        rel_data keys (từ JSONL):
          - source_id    : id của node nguồn
          - target_id    : id của node đích
          - relation_type: tên relationship (ví dụ: CO_KHOAN, CHE_TAI_BO_SUNG)

        Hoặc format cũ (graph_builder output):
          - source       : id node nguồn
          - target       : id node đích
          - relation      : tên relationship
        """
        # Hỗ trợ cả 2 format key
        source_id     = str(rel_data.get("source_id") or rel_data.get("source", ""))
        target_id     = str(rel_data.get("target_id") or rel_data.get("target", ""))
        relation_type = str(rel_data.get("relation_type") or rel_data.get("relation", ""))
        van_ban_id    = str(rel_data.get("van_ban_id", ""))

        if van_ban_id:
            if source_id != van_ban_id and not source_id.startswith(van_ban_id):
                source_id = f"{van_ban_id}_{source_id}"
            if target_id != van_ban_id and not target_id.startswith(van_ban_id):
                target_id = f"{van_ban_id}_{target_id}"

        if not source_id or not target_id or not relation_type:
            logger.warning(f"Skip rel (thiếu field): {rel_data}")
            return

        # Normalize trước (map QWen variant → type chuẩn)
        relation_type = RELATION_TYPE_NORMALIZE.get(relation_type, relation_type)

        # Whitelist để tránh Cypher injection
        if relation_type not in VALID_RELATION_TYPES:
            logger.warning(f"Skip rel (relation_type không hợp lệ): '{relation_type}'")
            return

        query = f"""
        MATCH (src {{id: $source_id}})
        MATCH (tgt {{id: $target_id}})
        MERGE (src)-[r:{relation_type}]->(tgt)
        """
        try:
            self.run_query(query, {"source_id": source_id, "target_id": target_id})
        except Exception as e:
            logger.error(
                f"Failed upsert rel [{source_id}]-[{relation_type}]->[{target_id}]: {e}"
            )

    def upsert_relationships(self, relations: list[dict], batch_size: int = 500):
        """
        Batch upsert danh sách relationships.
        Alias: upsert_edges_batch.
        """
        total = len(relations)
        logger.info(f"Upserting {total} relationships...")
        for i in range(0, total, batch_size):
            batch = relations[i : i + batch_size]
            for rel in batch:
                self.upsert_relationship(rel)
            logger.info(f"  Relations: {min(i + batch_size, total)}/{total}")

    # Alias để tương thích với code cũ
    upsert_edges_batch = upsert_relationships

    # ─────────────────────────────────────────────────────────────────────────
    # FULL INGEST (từ graph_builder output format)
    # ─────────────────────────────────────────────────────────────────────────

    def ingest_graph(self, graph_data: dict, clear_first: bool = False):
        """
        Ingest từ graph_builder output format: {"nodes": [...], "edges": [...]}.
        """
        if clear_first:
            self.clear_all()

        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])
        logger.info(f"Ingesting graph: {len(nodes)} nodes, {len(edges)} edges")

        self.upsert_nodes(nodes)
        self.upsert_relationships(edges)
        logger.info("Graph ingest complete!")

    # ─────────────────────────────────────────────────────────────────────────
    # CRITIC AGENT QUERIES
    # ─────────────────────────────────────────────────────────────────────────

    def get_hanh_vi_full_info(self, hanh_vi_id: str) -> dict:
        """
        Lấy toàn bộ thông tin HanhVi:
          - Hình phạt chính (CHE_TAI_CHINH)
          - Hình phạt bổ sung (CHE_TAI_BO_SUNG)
          - Điều/Khoản quy định hành vi này
        Dùng bởi Critic Agent để kiểm tra thiếu hình phạt phụ.
        """
        query = """
        MATCH (hv:HanhVi {id: $hv_id})
        OPTIONAL MATCH (hv)-[:CHE_TAI_CHINH]->(ct_chinh:CheTai)
        OPTIONAL MATCH (hv)-[:CHE_TAI_BO_SUNG]->(ct_bs:CheTai)
        OPTIONAL MATCH (src)-[:QUY_DINH_HANH_VI]->(hv)
        RETURN hv,
               collect(DISTINCT ct_chinh) AS hinh_phat_chinh,
               collect(DISTINCT ct_bs)    AS hinh_phat_bo_sung,
               collect(DISTINCT src)      AS nguon_quy_dinh
        """
        results = self.run_query(query, {"hv_id": hanh_vi_id})
        if results:
            return dict(results[0])
        return {}

    def get_dieu_with_full_context(self, dieu_id: str) -> dict:
        """
        Lấy toàn bộ ngữ cảnh của một Điều:
          - Khoản, Điểm con
          - Tham chiếu sang Điều khác
          - HanhVi và CheTai liên kết
        """
        query = """
        MATCH (d:Dieu {id: $dieu_id})
        OPTIONAL MATCH (d)-[:CO_KHOAN]->(k:Khoan)
        OPTIONAL MATCH (k)-[:CO_DIEM]->(p:Diem)
        OPTIONAL MATCH (d)-[:THAM_CHIEU]->(ref:Dieu)
        OPTIONAL MATCH (d)-[:QUY_DINH_HANH_VI]->(hv:HanhVi)
        OPTIONAL MATCH (k)-[:QUY_DINH_HANH_VI]->(hv2:HanhVi)
        OPTIONAL MATCH (hv)-[:CHE_TAI_CHINH]->(ct_chinh:CheTai)
        OPTIONAL MATCH (hv)-[:CHE_TAI_BO_SUNG]->(ct_bs:CheTai)
        RETURN d,
               collect(DISTINCT k)        AS khoan_list,
               collect(DISTINCT p)        AS diem_list,
               collect(DISTINCT ref)      AS tham_chieu_list,
               collect(DISTINCT hv)       AS hanh_vi_dieu,
               collect(DISTINCT hv2)      AS hanh_vi_khoan,
               collect(DISTINCT ct_chinh) AS che_tai_chinh,
               collect(DISTINCT ct_bs)    AS che_tai_bo_sung
        """
        results = self.run_query(query, {"dieu_id": dieu_id})
        if results:
            return dict(results[0])
        return {}

    def get_missing_chunks_for_critic(self, retrieved_chunk_ids: list[str]) -> list[dict]:
        """
        Core query cho Critic Agent:
        Từ danh sách chunk_id đã retrieved, tìm các node được THAM_CHIEU
        hoặc liên kết quan trọng nhưng chưa có trong retrieved set.

        Returns:
            list[dict] với keys: missing_chunk_id, node_id, node_ten, relation_type
        """
        query = """
        // Tìm các node trong retrieved chunks
        MATCH (n)
        WHERE n.chunk_id IN $chunk_ids

        // Tìm tất cả neighbors quan trọng (không chỉ THAM_CHIEU)
        OPTIONAL MATCH (n)-[r:THAM_CHIEU|CHE_TAI_BO_SUNG|CO_KHOAN|CO_DIEM]->(neighbor)
        WHERE neighbor.chunk_id IS NOT NULL
          AND NOT neighbor.chunk_id IN $chunk_ids

        RETURN DISTINCT
               neighbor.chunk_id  AS missing_chunk_id,
               neighbor.id        AS node_id,
               neighbor.ten       AS node_ten,
               type(r)            AS relation_type
        ORDER BY relation_type, node_id
        """
        results = self.run_query(query, {"chunk_ids": retrieved_chunk_ids})
        return [dict(r) for r in results if r.get("missing_chunk_id")]

    def get_graph_context_for_critic(self, chunk_ids: list[str]) -> str:
        """
        Trả về text mô tả graph context để đưa vào Critic Agent prompt.
        Đánh dấu rõ node nào ⚠ CHƯA RETRIEVED để agent biết cần bổ sung.
        """
        query = """
        MATCH (n)
        WHERE n.chunk_id IN $chunk_ids
        OPTIONAL MATCH (n)-[r]->(neighbor)
        RETURN
            n.id                AS node_id,
            n.type              AS node_type,
            n.ten               AS node_ten,
            type(r)             AS relation,
            neighbor.id         AS neighbor_id,
            neighbor.type       AS neighbor_type,
            neighbor.ten        AS neighbor_ten,
            neighbor.chunk_id   AS neighbor_chunk_id,
            CASE WHEN neighbor.chunk_id IN $chunk_ids
                 THEN true ELSE false END AS already_retrieved
        ORDER BY n.id, type(r)
        """
        results = self.run_query(query, {"chunk_ids": chunk_ids})

        lines = ["=== Knowledge Graph Context ==="]
        current_node = None

        for row in results:
            node_id = row.get("node_id", "")
            if node_id != current_node:
                current_node = node_id
                node_type = row.get("node_type", "")
                node_ten = row.get("node_ten", node_id)
                lines.append(f"\n[{node_type}] {node_ten} (id={node_id})")

            if row.get("relation"):
                status = "✓" if row.get("already_retrieved") else "⚠ CHƯA RETRIEVED"
                neighbor_ten = row.get("neighbor_ten") or row.get("neighbor_id", "")
                neighbor_type = row.get("neighbor_type", "")
                lines.append(
                    f"  --{row.get('relation')}--> [{neighbor_type}] {neighbor_ten} {status}"
                )

        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    # UTILS
    # ─────────────────────────────────────────────────────────────────────────

    def verify_connection(self) -> bool:
        """Kiểm tra kết nối Neo4j."""
        try:
            result = self.run_query("RETURN 'ok' AS status")
            return bool(result)
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")
            return False

    def get_stats(self) -> dict:
        """Thống kê số node và relationship theo label/type."""
        node_counts = {
            r["label"]: r["count"]
            for r in self.run_query(
                "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count ORDER BY count DESC"
            )
        }
        rel_counts = {
            r["type"]: r["count"]
            for r in self.run_query(
                "MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count ORDER BY count DESC"
            )
        }
        return {
            "nodes": node_counts,
            "relations": rel_counts,
            "total_nodes": sum(node_counts.values()),
            "total_relations": sum(rel_counts.values()),
        }
