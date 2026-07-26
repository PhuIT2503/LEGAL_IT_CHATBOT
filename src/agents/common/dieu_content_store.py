"""
dieu_content_store.py
======================
Truy xuất toàn văn Điều từ Qdrant (parent collection) — dùng chung bởi
agent_article_expand (mở rộng toàn Điều, không qua KG) và agent_critic (fetch
Điều bị Knowledge Graph phát hiện là thiếu). Xây 2 bảng tra 1 LẦN DUY NHẤT lúc
khởi tạo (dùng lại cho mọi câu hỏi trong phiên), KHÔNG quét lại Qdrant mỗi câu hỏi.
"""

import logging
from typing import Any, Optional

from src.retrieval.qdrant_hybrid_search import _point_id
from src.knowledge_graph.graph_builder import to_dieu_node_id

logger = logging.getLogger(__name__)


class DieuContentStore:
    def __init__(self, qdrant_client, qdrant_parent_col: str, qdrant_child_col: str):
        self.qdrant_client = qdrant_client
        self.qdrant_parent_col = qdrant_parent_col
        self.qdrant_child_col = qdrant_child_col

        # Map id Dieu chuẩn (Neo4j) -> id gốc của parent chunk trong Qdrant (giữ hoa/thường).
        # Cần vì node Dieu trong Neo4j dùng van_ban_id viết thường, còn payload Qdrant
        # giữ nguyên hoa/thường của tên file gốc -> không thể suy ngược 1-1 bằng string.
        self._dieu_parent_lookup: dict = {}
        self._build_dieu_parent_lookup()

        # Map id Dieu chuẩn -> tổng số chunk con (Khoản/Điểm) THẬT trong Qdrant —
        # nguồn đếm ĐỘC LẬP với Neo4j, dùng làm lưới an toàn cho
        # structurally_incomplete (xem docstring _build_dieu_child_chunk_count).
        self._dieu_child_chunk_count: dict = {}
        self._build_dieu_child_chunk_count()

    def _build_dieu_parent_lookup(self):
        """Quét 1 lần collection parent để dựng bảng tra id Dieu chuẩn -> parent id gốc Qdrant."""
        try:
            offset = None
            while True:
                points, offset = self.qdrant_client.scroll(
                    collection_name=self.qdrant_parent_col,
                    with_payload=True,
                    limit=256,
                    offset=offset,
                )
                for point in points:
                    payload = point.payload or {}
                    raw_dieu_id = payload.get("dieu_id", "")
                    raw_van_ban_id = payload.get("van_ban_id", "")
                    if not raw_dieu_id or not raw_van_ban_id:
                        continue
                    canonical = to_dieu_node_id(raw_van_ban_id, raw_dieu_id)
                    if canonical:
                        self._dieu_parent_lookup[canonical] = payload.get("id", "")
                if not offset:
                    break
            logger.info(f"Dieu->parent lookup: {len(self._dieu_parent_lookup)} Điều.")
        except Exception as e:
            logger.warning(f"Không dựng được dieu_parent_lookup (Qdrant parent collection có thể chưa có dữ liệu): {e}")

    def _build_dieu_child_chunk_count(self):
        """
        Quét 1 lần collection child, đếm số chunk_id DUY NHẤT thuộc mỗi Điều —
        dùng làm "tổng số phần" ĐỘC LẬP VỚI NEO4J cho check structurally_incomplete
        (critic_query.find_structurally_incomplete_dieu, tham số total_parts_override).

        LÝ DO CẦN NGUỒN THỨ 2 (phát hiện qua test thật): đếm "tổng số Khoản" bằng
        cách đếm cạnh CO_KHOAN trong Neo4j có thể bị THIẾU nếu bước ingest KG
        (LLM extraction) bỏ sót liên kết cho 1 số Khoản của 1 số Điều — đã xác
        nhận cụ thể: Điều 21 (Nghị định 15/2020) Neo4j chỉ thấy 1 Khoản dù văn
        bản gốc có nhiều hơn; Điều 26 Neo4j chỉ thấy cấu trúc dưới Khoản 1, dù
        Qdrant đã retrieve được cả Khoản 3, Khoản 5 (tức Khoản 3/5 CÓ TỒN TẠI
        trong dữ liệu ingest, chỉ là Neo4j không link tới chúng). Khi Neo4j đếm
        thiếu, "total_parts" bị đánh giá thấp bằng đúng "retrieved_parts" ->
        structurally_incomplete KHÔNG BAO GIỜ trigger được cho Điều đó, dù thực
        tế còn thiếu thật -> Critic Agent bỏ sót (bug đã quan sát qua test thật,
        khiến critic thua graph_blind ở các Điều bị ingest KG thiếu).

        Đếm qua Qdrant child collection thay vì Neo4j vì Qdrant chunk trực tiếp
        từ cấu trúc văn bản gốc lúc ingest (qdrant_local_ingest.py, chunking dựa
        trên regex/cấu trúc Khoản/Điểm thật) — KHÔNG qua bước LLM extraction dễ
        sai sót như lúc build đồ thị Neo4j, nên đáng tin cậy hơn làm "tổng số".
        """
        try:
            offset = None
            seen_chunk_ids: dict = {}
            while True:
                points, offset = self.qdrant_client.scroll(
                    collection_name=self.qdrant_child_col,
                    with_payload=True,
                    limit=256,
                    offset=offset,
                )
                for point in points:
                    payload = point.payload or {}
                    raw_dieu_id = payload.get("dieu_id", "")
                    raw_van_ban_id = payload.get("van_ban_id", "")
                    chunk_id = payload.get("id")
                    if not raw_dieu_id or not raw_van_ban_id or not chunk_id:
                        continue
                    canonical = to_dieu_node_id(raw_van_ban_id, raw_dieu_id)
                    if canonical:
                        seen_chunk_ids.setdefault(canonical, set()).add(chunk_id)
                if not offset:
                    break
            self._dieu_child_chunk_count = {d: len(ids) for d, ids in seen_chunk_ids.items()}
            logger.info(f"Dieu->so chunk con (Qdrant, doc lap Neo4j): {len(self._dieu_child_chunk_count)} Điều.")
        except Exception as e:
            logger.warning(f"Không dựng được dieu_child_chunk_count (Qdrant child collection có thể chưa có dữ liệu): {e}")

    def fetch_parent_content(self, canonical_dieu_id: str) -> Optional[str]:
        """Lấy toàn văn 1 Điều từ Qdrant bằng id Dieu chuẩn (qua bảng tra self._dieu_parent_lookup)."""
        record = self.fetch_parent_record(canonical_dieu_id)
        return record.get("text") if record else None

    def fetch_parent_record(self, canonical_dieu_id: str) -> Optional[dict[str, Any]]:
        """Lấy toàn văn kèm metadata để xếp hạng văn bản và tạo citation."""
        raw_parent_id = self._dieu_parent_lookup.get(canonical_dieu_id)
        if not raw_parent_id:
            return None
        try:
            points = self.qdrant_client.retrieve(
                collection_name=self.qdrant_parent_col,
                ids=[_point_id(raw_parent_id)],
                with_payload=True,
            )
            if points:
                payload = points[0].payload or {}
                metadata = payload.get("metadata") or {}
                return {
                    "dieu_id": canonical_dieu_id,
                    "chunk_id": payload.get("id", raw_parent_id),
                    "text": payload.get("content", ""),
                    "source": metadata.get("source", ""),
                    "metadata": metadata,
                }
        except Exception as e:
            logger.warning(f"Không fetch được parent content cho {canonical_dieu_id}: {e}")
        return None

    def child_chunk_count(self, canonical_dieu_id: str) -> int:
        return self._dieu_child_chunk_count.get(canonical_dieu_id, 0)
