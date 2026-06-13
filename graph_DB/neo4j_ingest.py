"""
neo4j_ingest.py
===============
Upsert Knowledge Graph vào Neo4j AuraDB.
Sử dụng MERGE để idempotent (chạy lại không bị duplicate).

Schema:
- Nodes: VanBan, Chuong, Dieu, Khoan, Diem, HanhVi, CheTai, ChuThe, NghiaVu, QuyenHan, KhaiNiem, DieuKien
- Relations: CO_CHUONG, CO_DIEU, CO_KHOAN, CO_DIEM, THAM_CHIEU, QUY_DINH_HANH_VI,
             CHE_TAI_CHINH, CHE_TAI_BO_SUNG, AP_DUNG_VOI, CO_NGHIA_VU, CO_QUYEN,
             DINH_NGHIA, CO_DIEU_KIEN, SUA_DOI, HUONG_DAN
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# NEO4J SCHEMA SETUP
# ─────────────────────────────────────────────────────────────────────────────

# Cypher queries để tạo constraints và indexes
SCHEMA_QUERIES = [
    # Unique constraints (PRIMARY KEY tương đương)
    "CREATE CONSTRAINT van_ban_id IF NOT EXISTS FOR (n:VanBan) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT dieu_id IF NOT EXISTS FOR (n:Dieu) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT khoan_id IF NOT EXISTS FOR (n:Khoan) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT diem_id IF NOT EXISTS FOR (n:Diem) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT hanh_vi_id IF NOT EXISTS FOR (n:HanhVi) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT che_tai_id IF NOT EXISTS FOR (n:CheTai) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT chu_the_id IF NOT EXISTS FOR (n:ChuThe) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT nghia_vu_id IF NOT EXISTS FOR (n:NghiaVu) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT quyen_han_id IF NOT EXISTS FOR (n:QuyenHan) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT khai_niem_id IF NOT EXISTS FOR (n:KhaiNiem) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT dieu_kien_id IF NOT EXISTS FOR (n:DieuKien) REQUIRE n.id IS UNIQUE",

    # Full-text indexes để tìm kiếm
    "CREATE FULLTEXT INDEX hanh_vi_search IF NOT EXISTS FOR (n:HanhVi) ON EACH [n.mo_ta]",
    "CREATE FULLTEXT INDEX khai_niem_search IF NOT EXISTS FOR (n:KhaiNiem) ON EACH [n.ten, n.dinh_nghia]",
    "CREATE FULLTEXT INDEX che_tai_search IF NOT EXISTS FOR (n:CheTai) ON EACH [n.mo_ta]",
]

# Map từ type string → Neo4j Label
TYPE_TO_LABEL = {
    "VanBan": "VanBan",
    "Chuong": "Chuong",
    "Dieu": "Dieu",
    "Khoan": "Khoan",
    "Diem": "Diem",
    "HanhVi": "HanhVi",
    "CheTai": "CheTai",
    "ChuThe": "ChuThe",
    "NghiaVu": "NghiaVu",
    "QuyenHan": "QuyenHan",
    "KhaiNiem": "KhaiNiem",
    "DieuKien": "DieuKien",
}


# ─────────────────────────────────────────────────────────────────────────────
# NEO4J CLIENT
# ─────────────────────────────────────────────────────────────────────────────

class Neo4jGraphIngestor:
    """
    Ingest Legal Knowledge Graph vào Neo4j AuraDB.
    """

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        """
        Args:
            uri: Neo4j URI (ví dụ: neo4j+s://xxxx.databases.neo4j.io)
            user: Username (thường là "neo4j")
            password: Password
            database: Database name (mặc định "neo4j" cho AuraDB)
        """
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
        """Chạy Cypher query và trả về kết quả."""
        with self.driver.session(database=self.database) as session:
            result = session.run(query, params or {})
            return list(result)

    def setup_schema(self):
        """Tạo constraints và indexes."""
        logger.info("Setting up Neo4j schema...")
        for query in SCHEMA_QUERIES:
            try:
                self.run_query(query)
                logger.debug(f"Schema: {query[:60]}...")
            except Exception as e:
                logger.warning(f"Schema query warning (may already exist): {e}")
        logger.info("Schema setup complete.")

    def clear_all(self):
        """Xóa toàn bộ graph (CẢNH BÁO: không thể hoàn tác!)."""
        logger.warning("Clearing all nodes and relationships...")
        self.run_query("MATCH (n) DETACH DELETE n")
        logger.info("Graph cleared.")

    # ─────────────────────────────────────────────────────────────────────────
    # NODE UPSERT
    # ─────────────────────────────────────────────────────────────────────────

    def upsert_node(self, node_data: dict):
        """
        Upsert một node vào Neo4j sử dụng MERGE.
        """
        node_type = node_data.get("type", "")
        label = TYPE_TO_LABEL.get(node_type, node_type)
        node_id = node_data.get("id", "")

        if not label or not node_id:
            logger.warning(f"Invalid node data (missing type or id): {node_data}")
            return

        # Tạo properties dict (loại bỏ 'type' và các giá trị None)
        props = {
            k: v for k, v in node_data.items()
            if k != "type" and v is not None
        }

        query = f"""
        MERGE (n:{label} {{id: $id}})
        SET n += $props
        RETURN n.id
        """
        try:
            self.run_query(query, {"id": node_id, "props": props})
        except Exception as e:
            logger.error(f"Failed to upsert node {node_id}: {e}")

    def upsert_nodes_batch(self, nodes: list[dict], batch_size: int = 100):
        """Batch upsert nodes."""
        logger.info(f"Upserting {len(nodes)} nodes...")

        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]
            for node in batch:
                self.upsert_node(node)
            logger.info(f"Nodes progress: {min(i + batch_size, len(nodes))}/{len(nodes)}")

    # ─────────────────────────────────────────────────────────────────────────
    # EDGE UPSERT
    # ─────────────────────────────────────────────────────────────────────────

    def upsert_edge(self, edge_data: dict):
        """
        Upsert một relationship vào Neo4j sử dụng MERGE.
        """
        source_id = edge_data.get("source", "")
        target_id = edge_data.get("target", "")
        relation_type = edge_data.get("relation", "")
        properties = edge_data.get("properties", {})

        if not source_id or not target_id or not relation_type:
            logger.warning(f"Invalid edge data: {edge_data}")
            return

        # Clean properties
        clean_props = {
            k: v for k, v in properties.items()
            if v is not None and isinstance(v, (str, int, float, bool))
        }

        query = f"""
        MATCH (source {{id: $source_id}})
        MATCH (target {{id: $target_id}})
        MERGE (source)-[r:{relation_type}]->(target)
        SET r += $props
        RETURN r
        """
        try:
            result = self.run_query(query, {
                "source_id": source_id,
                "target_id": target_id,
                "props": clean_props,
            })
            if not result:
                logger.warning(
                    f"Edge not created (nodes not found?): "
                    f"{source_id} -[{relation_type}]-> {target_id}"
                )
        except Exception as e:
            logger.error(f"Failed to upsert edge {source_id} -> {target_id}: {e}")

    def upsert_edges_batch(self, edges: list[dict], batch_size: int = 200):
        """Batch upsert edges."""
        logger.info(f"Upserting {len(edges)} edges...")

        for i in range(0, len(edges), batch_size):
            batch = edges[i:i + batch_size]
            for edge in batch:
                self.upsert_edge(edge)
            logger.info(f"Edges progress: {min(i + batch_size, len(edges))}/{len(edges)}")

    # ─────────────────────────────────────────────────────────────────────────
    # FULL INGEST
    # ─────────────────────────────────────────────────────────────────────────

    def ingest_graph(self, graph_data: dict, clear_first: bool = False):
        """
        Ingest toàn bộ graph vào Neo4j.
        
        Args:
            graph_data: dict {"nodes": [...], "edges": [...]}
            clear_first: Xóa graph cũ trước khi ingest (CẢNH BÁO!)
        """
        if clear_first:
            self.clear_all()

        self.setup_schema()

        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])

        logger.info(f"Ingesting graph: {len(nodes)} nodes, {len(edges)} edges")

        # Upsert nodes trước, sau đó mới edges
        self.upsert_nodes_batch(nodes)
        self.upsert_edges_batch(edges)

        logger.info("Graph ingest complete!")

    # ─────────────────────────────────────────────────────────────────────────
    # CYPHER QUERIES CHO CRITIC AGENT
    # ─────────────────────────────────────────────────────────────────────────

    def get_dieu_with_full_context(self, dieu_id: str) -> dict:
        """
        Lấy toàn bộ thông tin về một Điều luật:
        - Tất cả Khoản và Điểm
        - Tất cả quan hệ THAM_CHIEU
        - Tất cả HanhVi và CheTai liên quan
        """
        query = """
        MATCH (d:Dieu {id: $dieu_id})
        OPTIONAL MATCH (d)-[:CO_KHOAN]->(k:Khoan)
        OPTIONAL MATCH (k)-[:CO_DIEM]->(p:Diem)
        OPTIONAL MATCH (d)-[:THAM_CHIEU]->(ref_d:Dieu)
        OPTIONAL MATCH (k)-[:THAM_CHIEU]->(ref_k)
        OPTIONAL MATCH (d)-[:QUY_DINH_HANH_VI]->(hv:HanhVi)
        OPTIONAL MATCH (k)-[:QUY_DINH_HANH_VI]->(hv2:HanhVi)
        OPTIONAL MATCH (hv)-[:CHE_TAI_CHINH]->(ct_chinh:CheTai)
        OPTIONAL MATCH (hv)-[:CHE_TAI_BO_SUNG]->(ct_bs:CheTai)
        RETURN d, 
               collect(DISTINCT k) as khoan_list,
               collect(DISTINCT p) as diem_list,
               collect(DISTINCT ref_d) as tham_chieu_dieu,
               collect(DISTINCT hv) as hanh_vi_list,
               collect(DISTINCT hv2) as hanh_vi_khoan_list,
               collect(DISTINCT ct_chinh) as che_tai_chinh_list,
               collect(DISTINCT ct_bs) as che_tai_bo_sung_list
        """
        results = self.run_query(query, {"dieu_id": dieu_id})
        if results:
            return dict(results[0])
        return {}

    def get_hanh_vi_full_info(self, hanh_vi_id: str) -> dict:
        """
        Lấy toàn bộ thông tin về một HanhVi:
        - Hình phạt chính
        - Hình phạt bổ sung
        - Điều/Khoản quy định
        """
        query = """
        MATCH (hv:HanhVi {id: $hv_id})
        OPTIONAL MATCH (hv)-[:CHE_TAI_CHINH]->(ct_chinh:CheTai)
        OPTIONAL MATCH (hv)-[:CHE_TAI_BO_SUNG]->(ct_bs:CheTai)
        OPTIONAL MATCH (dieu_or_khoan)-[:QUY_DINH_HANH_VI]->(hv)
        RETURN hv,
               collect(DISTINCT ct_chinh) as hinh_phat_chinh,
               collect(DISTINCT ct_bs) as hinh_phat_bo_sung,
               collect(DISTINCT dieu_or_khoan) as nguon_quy_dinh
        """
        results = self.run_query(query, {"hv_id": hanh_vi_id})
        if results:
            return dict(results[0])
        return {}

    def get_missing_chunks_for_question(self, retrieved_chunk_ids: list[str]) -> list[str]:
        """
        Từ danh sách chunk IDs đã retrieved, tìm các Điều được tham chiếu nhưng chưa có.
        Đây là core query cho Critic Agent.
        
        Returns:
            Danh sách chunk_id cần retrieve thêm
        """
        query = """
        // Tìm tất cả Điều/Khoản được retrieved
        MATCH (n)
        WHERE n.chunk_id IN $chunk_ids
        
        // Tìm các tham chiếu từ những node này
        OPTIONAL MATCH (n)-[:THAM_CHIEU]->(ref)
        
        // Lọc những tham chiếu chưa được retrieved
        WHERE ref.chunk_id IS NOT NULL 
          AND NOT ref.chunk_id IN $chunk_ids
        
        RETURN DISTINCT ref.chunk_id as missing_chunk_id,
                        ref.id as node_id,
                        ref.ten as node_ten
        """
        results = self.run_query(query, {"chunk_ids": retrieved_chunk_ids})
        return [r["missing_chunk_id"] for r in results if r["missing_chunk_id"]]

    def get_graph_context_for_chunks(self, chunk_ids: list[str]) -> str:
        """
        Lấy context từ graph cho một danh sách chunk IDs.
        Trả về string text mô tả graph context cho Critic Agent prompt.
        """
        query = """
        MATCH (n)
        WHERE n.chunk_id IN $chunk_ids
        
        // Lấy tất cả quan hệ liên quan
        OPTIONAL MATCH (n)-[r]->(neighbor)
        
        RETURN n.id as node_id,
               n.type as node_type, 
               n.ten as node_ten,
               type(r) as relation,
               neighbor.id as neighbor_id,
               neighbor.type as neighbor_type,
               neighbor.ten as neighbor_ten,
               neighbor.chunk_id as neighbor_chunk_id,
               CASE WHEN neighbor.chunk_id IN $chunk_ids THEN true ELSE false END as neighbor_already_retrieved
        ORDER BY n.id, type(r)
        """
        results = self.run_query(query, {"chunk_ids": chunk_ids})

        # Format thành text
        lines = ["=== Knowledge Graph Context ==="]
        current_node = None

        for row in results:
            node_id = row.get("node_id", "")
            if node_id != current_node:
                current_node = node_id
                lines.append(f"\n[{row.get('node_type', '')}] {row.get('node_ten', node_id)}")

            if row.get("relation"):
                status = "✓" if row.get("neighbor_already_retrieved") else "⚠ CHƯA RETRIEVED"
                lines.append(
                    f"  -{row.get('relation')}-> "
                    f"[{row.get('neighbor_type', '')}] {row.get('neighbor_ten', row.get('neighbor_id', ''))} "
                    f"{status}"
                )

        return "\n".join(lines)

    def verify_connection(self) -> bool:
        """Kiểm tra kết nối Neo4j."""
        try:
            result = self.run_query("RETURN 'connected' as status")
            return bool(result)
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")
            return False

    def get_stats(self) -> dict:
        """Thống kê số lượng nodes và edges."""
        node_query = "MATCH (n) RETURN labels(n)[0] as label, count(n) as count ORDER BY count DESC"
        edge_query = "MATCH ()-[r]->() RETURN type(r) as type, count(r) as count ORDER BY count DESC"

        node_counts = {r["label"]: r["count"] for r in self.run_query(node_query)}
        edge_counts = {r["type"]: r["count"] for r in self.run_query(edge_query)}

        return {
            "nodes": node_counts,
            "edges": edge_counts,
            "total_nodes": sum(node_counts.values()),
            "total_edges": sum(edge_counts.values()),
        }
