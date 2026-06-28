"""
critic_query.py
===============
Cypher queries và helper functions cho Critic Agent.
Critic Agent dùng Knowledge Graph để:
1. Kiểm tra xem retrieved context có đủ thông tin không
2. Xác định các Điều/Khoản bị thiếu và cần retrieve thêm
3. Cung cấp graph context cho LLM critic evaluation
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CYPHER QUERY TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────

class CriticQueryEngine:
    """
    Engine thực thi các Cypher queries phục vụ Critic Agent.
    """

    def __init__(self, neo4j_ingestor):
        """
        Args:
            neo4j_ingestor: Instance của Neo4jGraphIngestor đã kết nối
        """
        self.db = neo4j_ingestor

    # ─────────────────────────────────────────────────────────────────────────
    # CORE CRITIC QUERIES
    # ─────────────────────────────────────────────────────────────────────────

    def find_missing_references(self, retrieved_chunk_ids: list[str]) -> list[dict]:
        """
        Query quan trọng nhất cho Critic Agent.
        
        Từ các chunk đã retrieved, tìm các Điều/Khoản được THAM_CHIEU
        nhưng chưa có trong retrieved set.
        
        Returns:
            List các missing chunk với thông tin đầy đủ
        """
        query = """
        // Tìm tất cả nodes tương ứng với chunk IDs đã retrieved
        MATCH (retrieved)
        WHERE retrieved.chunk_id IN $chunk_ids
        
        // Tìm các tham chiếu đi ra từ những nodes này
        MATCH (retrieved)-[ref_rel:THAM_CHIEU]->(referenced)
        
        // Chỉ lấy những node được tham chiếu mà chưa retrieved
        WHERE referenced.chunk_id IS NOT NULL
          AND NOT referenced.chunk_id IN $chunk_ids
          AND NOT referenced.is_placeholder = true
        
        RETURN DISTINCT
            retrieved.id as from_node_id,
            retrieved.ten as from_node_ten,
            referenced.chunk_id as missing_chunk_id,
            referenced.id as referenced_node_id,
            referenced.ten as referenced_node_ten,
            referenced.type as referenced_node_type,
            ref_rel.ghi_chu as reference_note,
            ref_rel.loai_tham_chieu as reference_type
        ORDER BY referenced.id
        """
        results = self.db.run_query(query, {"chunk_ids": retrieved_chunk_ids})
        return [dict(r) for r in results]

    def find_incomplete_penalty_info(self, retrieved_chunk_ids: list[str]) -> list[dict]:
        """
        Kiểm tra xem thông tin về HanhVi đã có đủ cả hình phạt CHÍNH và BỔ SUNG chưa.
        
        Nếu có HanhVi nhưng chỉ có 1 loại CheTai → cần retrieve thêm.
        """
        query = """
        // Tìm các HanhVi được đề cập trong retrieved chunks
        MATCH (n)-[:QUY_DINH_HANH_VI]->(hv:HanhVi)
        WHERE n.chunk_id IN $chunk_ids
        
        // Kiểm tra hình phạt chính
        OPTIONAL MATCH (hv)-[:CHE_TAI_CHINH]->(ct_chinh:CheTai)
        
        // Kiểm tra hình phạt bổ sung
        OPTIONAL MATCH (hv)-[:CHE_TAI_BO_SUNG]->(ct_bo_sung:CheTai)
        
        WITH hv,
             collect(DISTINCT ct_chinh) as hinh_phat_chinh,
             collect(DISTINCT ct_bo_sung) as hinh_phat_bo_sung
        
        // Lấy chunk nguồn quy định hành vi này
        MATCH (source)-[:QUY_DINH_HANH_VI]->(hv)
        
        // Chỉ trả về những HanhVi có CheTai nhưng chunk nguồn chưa được retrieved
        WHERE source.chunk_id IS NOT NULL
          AND NOT source.chunk_id IN $chunk_ids
        
        RETURN hv.id as hanh_vi_id,
               hv.mo_ta as hanh_vi_mo_ta,
               size(hinh_phat_chinh) as so_hinh_phat_chinh,
               size(hinh_phat_bo_sung) as so_hinh_phat_bo_sung,
               collect(DISTINCT source.chunk_id) as can_retrieve_chunks,
               [c IN hinh_phat_chinh | c.mo_ta] as mo_ta_hinh_phat_chinh,
               [c IN hinh_phat_bo_sung | c.mo_ta] as mo_ta_hinh_phat_bo_sung
        """
        results = self.db.run_query(query, {"chunk_ids": retrieved_chunk_ids})
        return [dict(r) for r in results]

    def get_complete_dieu_info(self, dieu_so: str, van_ban_id: str) -> dict:
        """
        Lấy toàn bộ thông tin một Điều để cung cấp cho Critic Agent context.
        
        Args:
            dieu_so: Số điều (ví dụ: "7", "24")
            van_ban_id: ID văn bản (ví dụ: "van_ban_luat_an_ninh_mang_2025")
        """
        query = """
        MATCH (d:Dieu)
        WHERE d.so = $dieu_so AND d.van_ban_id = $van_ban_id
        
        OPTIONAL MATCH (d)-[:CO_KHOAN]->(k:Khoan)
        OPTIONAL MATCH (k)-[:CO_DIEM]->(p:Diem)
        OPTIONAL MATCH (d)-[:QUY_DINH_HANH_VI]->(hv:HanhVi)
        OPTIONAL MATCH (k)-[:QUY_DINH_HANH_VI]->(hv2:HanhVi)
        WITH d, 
             collect(DISTINCT {id: k.id, so: k.so, content: k.content}) as khoan_list,
             collect(DISTINCT {id: p.id, ky_hieu: p.ky_hieu, content: p.content}) as diem_list,
             collect(DISTINCT {id: hv.id, mo_ta: hv.mo_ta}) as hanh_vi_from_dieu,
             collect(DISTINCT {id: hv2.id, mo_ta: hv2.mo_ta}) as hanh_vi_from_khoan
        
        OPTIONAL MATCH (d)-[:THAM_CHIEU]->(ref)
        
        RETURN d,
               khoan_list,
               diem_list,
               hanh_vi_from_dieu + hanh_vi_from_khoan as all_hanh_vi,
               collect(DISTINCT {id: ref.id, ten: ref.ten, type: ref.type}) as tham_chieu_list
        """
        results = self.db.run_query(query, {
            "dieu_so": dieu_so,
            "van_ban_id": van_ban_id
        })
        if results:
            return dict(results[0])
        return {}

    def check_retrieval_completeness(self, retrieved_chunk_ids: list[str]) -> dict:
        """
        Kiểm tra tổng thể độ đầy đủ của retrieval.
        
        Returns:
            dict với:
            - is_complete: bool
            - missing_references: list chunks bị thiếu do THAM_CHIEU
            - incomplete_penalties: list HanhVi thiếu thông tin chế tài
            - suggestions: list hành động cần làm
        """
        missing_refs = self.find_missing_references(retrieved_chunk_ids)
        incomplete_penalties = self.find_incomplete_penalty_info(retrieved_chunk_ids)

        suggestions = []
        missing_chunk_ids = set()

        for ref in missing_refs:
            chunk_id = ref.get("missing_chunk_id")
            if chunk_id:
                missing_chunk_ids.add(chunk_id)
                suggestions.append({
                    "action": "retrieve",
                    "chunk_id": chunk_id,
                    "reason": f"Chunk '{ref.get('from_node_ten', '')}' tham chiếu đến "
                              f"'{ref.get('referenced_node_ten', ref.get('referenced_node_id', ''))}' "
                              f"nhưng chưa được retrieved. "
                              f"({ref.get('reference_note', '')})"
                })

        for penalty in incomplete_penalties:
            for chunk_id in penalty.get("can_retrieve_chunks", []):
                if chunk_id:
                    missing_chunk_ids.add(chunk_id)
                    suggestions.append({
                        "action": "retrieve",
                        "chunk_id": chunk_id,
                        "reason": f"Hành vi '{penalty.get('hanh_vi_mo_ta', '')}' "
                                  f"có thông tin chế tài chưa đầy đủ trong retrieved context."
                    })

        return {
            "is_complete": len(missing_chunk_ids) == 0,
            "missing_chunk_ids": list(missing_chunk_ids),
            "missing_references": missing_refs,
            "incomplete_penalties": incomplete_penalties,
            "suggestions": suggestions,
        }

    def format_graph_context_for_llm(self, retrieved_chunk_ids: list[str]) -> str:
        """
        Format graph context thành text để đưa vào prompt của Critic Agent LLM.
        """
        completeness = self.check_retrieval_completeness(retrieved_chunk_ids)

        lines = ["=== KNOWLEDGE GRAPH ANALYSIS ===\n"]

        # Status
        if completeness["is_complete"]:
            lines.append("✅ Retrieval đầy đủ: Không phát hiện thông tin bị thiếu từ graph.\n")
        else:
            lines.append(f"⚠️ Phát hiện {len(completeness['missing_chunk_ids'])} chunk cần retrieve thêm:\n")

        # Missing references
        if completeness["missing_references"]:
            lines.append("\n--- Các tham chiếu bị thiếu ---")
            for ref in completeness["missing_references"]:
                lines.append(
                    f"• '{ref.get('from_node_ten', ref.get('from_node_id', ''))}' "
                    f"→[THAM_CHIEU]→ "
                    f"'{ref.get('referenced_node_ten', ref.get('referenced_node_id', ''))}'"
                    f" [chunk: {ref.get('missing_chunk_id', 'N/A')}]"
                    + (f" | Lý do: {ref.get('reference_note', '')}" if ref.get('reference_note') else "")
                )

        # Incomplete penalties
        if completeness["incomplete_penalties"]:
            lines.append("\n--- Hành vi có chế tài chưa đầy đủ ---")
            for p in completeness["incomplete_penalties"]:
                lines.append(
                    f"• Hành vi: {p.get('hanh_vi_mo_ta', '')}\n"
                    f"  Cần retrieve: {p.get('can_retrieve_chunks', [])}"
                )

        # Suggestions
        if completeness["suggestions"]:
            lines.append("\n--- Khuyến nghị retrieve thêm ---")
            for i, sug in enumerate(completeness["suggestions"], 1):
                lines.append(f"{i}. {sug['reason']}")
                lines.append(f"   → chunk_id: {sug['chunk_id']}")

        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    # SEARCH QUERIES
    # ─────────────────────────────────────────────────────────────────────────

    def search_hanh_vi_by_keyword(self, keyword: str, limit: int = 5) -> list[dict]:
        """
        Tìm kiếm HanhVi theo từ khóa (dùng full-text search).
        """
        query = """
        CALL db.index.fulltext.queryNodes('hanh_vi_search', $keyword)
        YIELD node, score
        OPTIONAL MATCH (node)-[:CHE_TAI_CHINH]->(ct_chinh:CheTai)
        OPTIONAL MATCH (node)-[:CHE_TAI_BO_SUNG]->(ct_bo_sung:CheTai)
        RETURN node.id as hanh_vi_id,
               node.mo_ta as hanh_vi_mo_ta,
               score,
               collect(DISTINCT ct_chinh.mo_ta) as hinh_phat_chinh,
               collect(DISTINCT ct_bo_sung.mo_ta) as hinh_phat_bo_sung
        ORDER BY score DESC
        LIMIT $limit
        """
        results = self.db.run_query(query, {"keyword": keyword, "limit": limit})
        return [dict(r) for r in results]

    def get_van_ban_structure(self, van_ban_id: str) -> list[dict]:
        """
        Lấy cấu trúc tổng thể của một văn bản (danh sách Điều và tiêu đề).
        """
        query = """
        MATCH (vb:VanBan {id: $van_ban_id})-[:CO_DIEU]->(d:Dieu)
        RETURN d.so as dieu_so, d.ten as dieu_ten, d.chunk_id as chunk_id
        ORDER BY toInteger(d.so)
        """
        results = self.db.run_query(query, {"van_ban_id": van_ban_id})
        return [dict(r) for r in results]

    def get_all_che_tai_for_van_ban(self, van_ban_id: str) -> list[dict]:
        """
        Lấy tất cả chế tài (hình phạt) trong một văn bản.
        Hữu ích để Critic Agent biết spectrum of penalties.
        """
        query = """
        MATCH (d:Dieu {van_ban_id: $van_ban_id})-[:CO_KHOAN|QUY_DINH_HANH_VI*1..2]->(hv:HanhVi)
        OPTIONAL MATCH (hv)-[:CHE_TAI_CHINH]->(ct_chinh:CheTai)
        OPTIONAL MATCH (hv)-[:CHE_TAI_BO_SUNG]->(ct_bo_sung:CheTai)
        RETURN hv.mo_ta as hanh_vi,
               collect(DISTINCT {loai: 'chinh', mo_ta: ct_chinh.mo_ta, muc_phat_min: ct_chinh.muc_phat_min, muc_phat_max: ct_chinh.muc_phat_max}) as phat_chinh,
               collect(DISTINCT {loai: 'bo_sung', mo_ta: ct_bo_sung.mo_ta}) as phat_bo_sung,
               d.so as dieu_so,
               d.chunk_id as dieu_chunk_id
        ORDER BY toInteger(d.so)
        """
        results = self.db.run_query(query, {"van_ban_id": van_ban_id})
        return [dict(r) for r in results]
