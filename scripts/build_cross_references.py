import os
import re
import logging
from neo4j import GraphDatabase

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class CrossReferenceBuilder:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def build_references(self):
        # Regex patterns để bắt tham chiếu
        # Ví dụ: "khoản 2 Điều 100", "tại Điều 19", "khoản 1, khoản 2 Điều 19"
        # Bắt "Điều X"
        pattern_dieu = re.compile(r"(?:tại\s+)?Điều\s+(\d+)", re.IGNORECASE)
        # Bắt "khoản Y Điều X"
        pattern_khoan_dieu = re.compile(r"khoản\s+(\d+)[^Đ]*Điều\s+(\d+)", re.IGNORECASE)

        query_nodes = """
        MATCH (n)
        WHERE n.mo_ta IS NOT NULL AND (n.mo_ta CONTAINS 'Điều' OR n.mo_ta CONTAINS 'khoản')
        RETURN id(n) AS internal_id, n.id AS node_id, n.van_ban_id AS van_ban_id, labels(n)[0] AS label, n.mo_ta AS mo_ta
        """

        with self.driver.session() as session:
            result = session.run(query_nodes)
            nodes = [record.data() for record in result]

            logger.info(f"🔍 Found {len(nodes)} nodes with potential cross-references in 'mo_ta'.")

            edges_to_create = []

            for node in nodes:
                mo_ta = node['mo_ta']
                node_id = node['node_id']
                van_ban_id = node['van_ban_id'] or ""

                # Tìm khoản Y Điều X trước
                khoan_matches = pattern_khoan_dieu.findall(mo_ta)
                for (khoan, dieu) in khoan_matches:
                    target_id = f"{van_ban_id}_D{dieu}_K{khoan}" if van_ban_id else f"D{dieu}_K{khoan}"
                    edges_to_create.append({"source": node_id, "target": target_id})
                
                # Tìm Điều X (loại trừ những Điều đã bắt ở khoản)
                # Dùng một set để lưu các Điều đã bắt qua khoản
                dieu_from_khoan = {dieu for (khoan, dieu) in khoan_matches}
                
                dieu_matches = pattern_dieu.findall(mo_ta)
                for dieu in dieu_matches:
                    target_id = f"{van_ban_id}_D{dieu}" if van_ban_id else f"D{dieu}"
                    edges_to_create.append({"source": node_id, "target": target_id})

            if not edges_to_create:
                logger.info("Không tìm thấy tham chiếu mới nào để tạo.")
                return

            # Xóa trùng lặp
            unique_edges = []
            seen = set()
            for edge in edges_to_create:
                key = (edge['source'], edge['target'])
                if key not in seen:
                    seen.add(key)
                    unique_edges.append(edge)

            logger.info(f"🔗 Bắt đầu tạo {len(unique_edges)} liên kết THAM_CHIEU tự động...")

            # Batch upsert
            batch_size = 500
            for i in range(0, len(unique_edges), batch_size):
                batch = unique_edges[i:i+batch_size]
                query_merge = """
                UNWIND $batch AS edge
                MATCH (s {id: edge.source})
                MATCH (t {id: edge.target})
                MERGE (s)-[:THAM_CHIEU]->(t)
                """
                session.run(query_merge, batch=batch)
                logger.info(f"  Đã merge {min(i+batch_size, len(unique_edges))}/{len(unique_edges)} edges.")

        logger.info("✅ Hoàn thành Auto-linking Cross References!")

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
        query_orphans = """
        MATCH (n) WHERE count{(n)--()} = 0
        SET n:Orphan
        RETURN count(n) AS orphans
        """
        with self.driver.session() as session:
            result = session.run(query_orphans)
            record = result.single()
            if record:
                logger.info(f"✅ Đã gán nhãn :Orphan cho {record['orphans']} nodes mồ côi.")

def main():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "legal_kg_2024")

    logger.info("=== BẮT ĐẦU TỰ ĐỘNG NỐI THAM CHIẾU VÀ LÀM SẠCH GRAPH ===")
    builder = CrossReferenceBuilder(uri, user, password)
    try:
        builder.build_references()
        builder.propagate_penalties()
        builder.tag_orphans()
    except Exception as e:
        logger.error(f"Lỗi: {e}")
    finally:
        builder.close()

if __name__ == "__main__":
    main()
