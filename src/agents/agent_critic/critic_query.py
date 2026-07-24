"""
critic_query.py
===============
Cypher queries và helper functions cho Critic Agent.
Critic Agent dùng Knowledge Graph để:
1. Kiểm tra xem retrieved context có đủ thông tin không
2. Xác định các Điều/Khoản bị thiếu và cần retrieve thêm
3. Cung cấp graph context cho LLM critic evaluation

Đơn vị theo dõi "đã retrieve" là ID NODE DIEU CHUẨN của Neo4j
(vd "ngh_nh_15_2020_n_cp_..._D90"), KHÔNG phải chunk_id.

Lý do: retrieval thật (Qdrant) trả về chunk ở cấp Khoản/Điểm, trong khi
property `chunk_id` trên mọi node trong KG lại là chunk PARENT (cả Điều) —
2 giá trị này không bao giờ khớp string. Dùng `graph_builder.to_dieu_node_id()`
để quy đổi (van_ban_id, dieu_id) lấy từ payload Qdrant sang id Dieu chuẩn
trước khi gọi các hàm trong module này.
"""

import logging
import re
import sys
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.knowledge_graph.graph_builder import to_dieu_node_id

logger = logging.getLogger(__name__)

# Khớp id Dieu/Khoan/Diem dạng "{van_ban_id}_D<so>[_K<so>][_P<ky_hieu>]".
# van_ban_id luôn viết thường (get_van_ban_id) nên "_D<digit>" (D hoa) chỉ có thể
# là mốc bắt đầu số Điều, không lẫn với ký tự trong van_ban_id.
_STRUCT_NODE_ID_PATTERN = re.compile(r'^(?P<dieu>.+_D\d+[a-zA-Z]*)(?:_K\d+)?(?:_P[a-zA-Z0-9]+)?$')


def _owning_dieu_id(node_id: str, node_type: str, chunk_id: Optional[str], van_ban_id: Optional[str]) -> Optional[str]:
    """
    Suy ra id Dieu (chuẩn Neo4j) đang "sở hữu" 1 node bất kỳ trong graph.
    - Dieu/Khoan/Diem: suy trực tiếp từ chính id của node (id luôn chứa "_D<so>").
    - Các loại khác (HanhVi, CheTai, ChuThe, NghiaVu, QuyenHan, KhaiNiem, DieuKien):
      không có số Điều trong id, phải suy qua chunk_id (chunk PARENT của Điều đã sinh ra nó).
    """
    if node_type in ("Dieu", "Khoan", "Diem"):
        m = _STRUCT_NODE_ID_PATTERN.match(node_id or "")
        return m.group("dieu") if m else None

    if chunk_id and van_ban_id:
        return to_dieu_node_id(van_ban_id, chunk_id)

    return None


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

    def find_missing_references(
        self, focus_dieu_ids: list[str], all_retrieved_dieu_ids: Optional[list[str]] = None
    ) -> list[dict]:
        """
        Query quan trọng nhất cho Critic Agent.

        LƯU Ý: chỉ xét THAM_CHIEU SANG ĐIỀU KHÁC. Tham chiếu trong CÙNG 1 Điều
        (vd chính ở khoản này, bổ sung ở khoản khác cùng Điều) KHÔNG cần đi qua
        Knowledge Graph — chỉ cần expand lấy toàn văn Điều đó (parent chunk) là đủ,
        không có gì để "phát hiện" ở tầng KG. KG chỉ thực sự cần thiết khi Điều
        đang có KHÔNG THỂ tự cho biết "Điều nào khác" liên quan — đó là lý do
        tồn tại của Critic Agent (retrieval không thể đoán trước Điều khác).

        Xét CẢ HAI CHIỀU:
          - outgoing: đã lấy A, A trích dẫn ra B (Điều khác) chưa lấy (vd "theo Điều 26 Luật này")
          - incoming: B (Điều khác) CHƯA lấy trích dẫn ÁP DỤNG vào A đã lấy — đây là
            chiều quan trọng cho đúng ca "chính/phụ" khi chúng nằm ở 2 VĂN BẢN/ĐIỀU
            khác nhau: khoản quy định hình phạt bổ sung có thể nằm ở Điều khác,
            viết "đối với hành vi quy định tại Điều X" (X = Điều đã lấy).

        Args:
            focus_dieu_ids: id Dieu quét THAM_CHIEU xuất phát từ (thường là tập
                đã lọc theo điểm số — xem agent_critic.critic_score_ratio).
            all_retrieved_dieu_ids: TOÀN BỘ id Dieu retrieval THỰC SỰ đã lấy được
                (không lọc điểm số) — dùng để xác định 1 Điều tham chiếu tới có
                "đã có sẵn" hay chưa. QUAN TRỌNG: phải dùng tập ĐẦY ĐỦ này, không
                phải focus_dieu_ids — nếu không, 1 Điều đã thực sự retrieve được
                (chỉ là bị lọc khỏi focus vì điểm thấp) sẽ bị báo nhầm là "thiếu"
                dù đã có sẵn trong ngữ cảnh rồi. Mặc định = focus_dieu_ids nếu
                không truyền (dùng khi caller không phân biệt 2 tập).

        Returns:
            List dict {missing_dieu_id, missing_node_ten, reason, direction}
        """
        if not focus_dieu_ids:
            return []

        focus_dieu_set = set(focus_dieu_ids)
        all_retrieved_set = set(all_retrieved_dieu_ids) if all_retrieved_dieu_ids is not None else focus_dieu_set
        query = """
        MATCH (a)-[r:THAM_CHIEU]->(b)
        RETURN a.id AS a_id, labels(a)[0] AS a_type, a.chunk_id AS a_chunk, a.van_ban_id AS a_vb, a.ten AS a_ten,
               b.id AS b_id, labels(b)[0] AS b_type, b.chunk_id AS b_chunk, b.van_ban_id AS b_vb, b.ten AS b_ten,
               r.ghi_chu AS note
        """
        rows = [dict(r) for r in self.db.run_query(query)]

        missing: dict[str, dict] = {}
        for row in rows:
            a_dieu = _owning_dieu_id(row["a_id"], row["a_type"], row["a_chunk"], row["a_vb"])
            b_dieu = _owning_dieu_id(row["b_id"], row["b_type"], row["b_chunk"], row["b_vb"])
            if not a_dieu or not b_dieu or a_dieu == b_dieu:
                continue

            a_label = row["a_ten"] or row["a_id"]
            b_label = row["b_ten"] or row["b_id"]

            if a_dieu in focus_dieu_set and b_dieu not in all_retrieved_set:
                missing.setdefault(b_dieu, {
                    "missing_dieu_id": b_dieu,
                    "missing_node_ten": b_label,
                    "direction": "outgoing",
                    "reason": f"Đã lấy '{a_label}' nhưng nó tham chiếu tới '{b_label}' (Điều {b_dieu}) chưa được lấy.",
                })

            if b_dieu in focus_dieu_set and a_dieu not in all_retrieved_set:
                missing.setdefault(a_dieu, {
                    "missing_dieu_id": a_dieu,
                    "missing_node_ten": a_label,
                    "direction": "incoming",
                    "reason": (
                        f"'{a_label}' (Điều {a_dieu}) tham chiếu áp dụng vào '{b_label}' đã lấy, "
                        f"nhưng {a_dieu} chưa được lấy — có thể chứa hình phạt bổ sung/điều kiện liên quan."
                    ),
                })

        return list(missing.values())

    def find_compound_penalty_behaviors(self, retrieved_dieu_ids: list[str]) -> list[dict]:
        """
        Với mỗi Điều đã retrieve, tìm các HanhVi có CẢ hình phạt CHÍNH LẪN BỔ
        SUNG được ghi nhận trong graph (ví dụ kinh điển: "vượt đèn đỏ" vừa bị
        phạt tiền (chính) vừa bị tước giấy phép lái xe (bổ sung)).

        Đây LÀ tín hiệu cốt lõi của Critic Agent — KHÔNG phải kiểm tra graph có
        bị thiếu/lệch dữ liệu hay không (graph do LLM trích xuất trên toàn văn
        Điều nên gần như luôn đầy đủ cả 2 phía). Vấn đề nằm ở RETRIEVAL: pháp
        luật xử phạt hành chính VN gần như luôn tách hình phạt chính và bổ sung
        ở 2 KHOẢN KHÁC NHAU trong cùng 1 Điều, mà retrieval (Qdrant) chỉ lấy
        chunk rời rạc cấp Khoản/Điểm theo độ tương đồng ngữ nghĩa với câu hỏi —
        nếu top-k trúng Khoản có hình phạt chính, rất dễ KHÔNG trúng luôn Khoản
        (khác) chứa hình phạt bổ sung vì 2 đoạn văn này thường không giống nhau
        về mặt ngữ nghĩa. Phát hiện được pattern này -> hành động là lấy lại
        TOÀN VĂN Điều đó để đảm bảo cả 2 phần đều có trong ngữ cảnh.

        LƯU Ý: property chunk_id trên HanhVi/CheTai trỏ tới chunk PARENT (cả
        Điều), không phải Khoản cụ thể — nên KHÔNG THỂ xác định "hình phạt bổ
        sung nằm ở Khoản nào" từ graph, chỉ biết "Điều này CÓ hình phạt bổ
        sung" -> hành động khả thi duy nhất là fetch lại toàn văn Điều.
        """
        if not retrieved_dieu_ids:
            return []

        # (d)-[:CO_KHOAN|CO_DIEM*0..4]->(struc) đi qua mọi độ sâu Khoản/Điểm, kể cả
        # Điểm lồng trong Điểm (vd "D16_K3_Pd_Pd") — *0..4 cho phép struc = d (Dieu
        # có QUY_DINH_HANH_VI trực tiếp) đến sâu 4 cấp.
        query = """
        MATCH (d:Dieu) WHERE d.id IN $dieu_ids
        MATCH (d)-[:CO_KHOAN|CO_DIEM*0..4]->(struc)
        MATCH (struc)-[:QUY_DINH_HANH_VI]->(hv:HanhVi)
        WITH DISTINCT d, hv
        MATCH (hv)-[:CHE_TAI_CHINH]->(ct_chinh:CheTai)
        MATCH (hv)-[:CHE_TAI_BO_SUNG]->(ct_bs:CheTai)
        RETURN d.id AS dieu_id, d.ten AS dieu_ten,
               hv.id AS hanh_vi_id, hv.mo_ta AS hanh_vi_mo_ta,
               collect(DISTINCT ct_chinh.mo_ta) AS mo_ta_hinh_phat_chinh,
               collect(DISTINCT ct_bs.mo_ta) AS mo_ta_hinh_phat_bo_sung
        """
        results = self.db.run_query(query, {"dieu_ids": retrieved_dieu_ids})
        return [dict(r) for r in results]

    def count_dieu_parts_batch(self, dieu_ids: list[str]) -> dict[str, int]:
        """Đếm tổng số Khoản/Điểm (mọi cấp) thuộc mỗi Điều trong graph — dùng làm mẫu số cho find_structurally_incomplete_dieu."""
        if not dieu_ids:
            return {}
        query = """
        MATCH (d:Dieu) WHERE d.id IN $dieu_ids
        OPTIONAL MATCH (d)-[:CO_KHOAN|CO_DIEM*1..4]->(struc)
        RETURN d.id AS dieu_id, count(DISTINCT struc) AS total
        """
        results = self.db.run_query(query, {"dieu_ids": dieu_ids})
        return {r["dieu_id"]: r["total"] for r in results}

    def find_structurally_incomplete_dieu(
        self,
        focus_dieu_ids: list[str],
        retrieved_part_counts: dict[str, int],
        total_parts_override: Optional[dict[str, int]] = None,
    ) -> list[dict]:
        """
        Tổng quát hóa find_compound_penalty_behaviors — KHÔNG giới hạn riêng
        pattern chế tài chính/bổ sung (HanhVi/CheTai), mà xét CẤU TRÚC thuần túy:
        nếu 1 Điều trong phạm vi focus có NHIỀU Khoản/Điểm hơn số phần retrieval
        thực sự lấy được, xem đây là tín hiệu "có thể còn thiếu nội dung liên
        quan trong Khoản khác của CHÍNH Điều này" -> nên lấy lại toàn văn.

        Bao trùm cả trường hợp chế tài kép LẪN các cấu trúc khác mà quan hệ ngữ
        nghĩa (HanhVi/CheTai) không có mặt trong graph để dò riêng — ví dụ 1
        Điều liệt kê nhiều trường hợp miễn trừ trách nhiệm, nhiều điều kiện áp
        dụng, v.v. (mọi Điều nhiều Khoản/Điểm về bản chất có rủi ro retrieval
        chỉ trúng 1 phần, không phụ thuộc nội dung Khoản đó là gì).

        Args:
            focus_dieu_ids: id Dieu trong phạm vi kiểm tra (đã lọc theo điểm số).
            retrieved_part_counts: {dieu_id: số Khoản/Điểm THỰC SỰ đã retrieve
                được, đếm theo chunk_id duy nhất} — do caller (node_critic_check)
                tính từ retrieved_chunks.
            total_parts_override: {dieu_id: tổng số Khoản/Điểm} tính từ nguồn
                ĐỘC LẬP với Neo4j (vd đếm chunk thật trong Qdrant, xây bằng
                to_dieu_node_id trực tiếp từ cấu trúc văn bản gốc — KHÔNG qua
                bước LLM extraction dựng Neo4j nên ít rủi ro thiếu sót hơn). Nếu
                có, LẤY MAX giữa giá trị này và count_dieu_parts_batch (Neo4j)
                cho từng Điều — phòng trường hợp Neo4j đếm THIẾU do bỏ sót liên
                kết CO_KHOAN khi ingest (bug thật đã phát hiện qua test: Điều 21
                Neo4j chỉ thấy 1 Khoản, Điều 26 chỉ thấy Khoản 1, dù Qdrant đã
                retrieve được cả Khoản 3/5 — khiến check này không bao giờ
                trigger được cho các Điều đó, Critic Agent bỏ sót thật).

        Returns:
            List dict {dieu_id, total_parts, retrieved_parts}
        """
        if not focus_dieu_ids:
            return []
        neo4j_totals = self.count_dieu_parts_batch(focus_dieu_ids)
        results = []
        for dieu_id in focus_dieu_ids:
            total = neo4j_totals.get(dieu_id, 0)
            if total_parts_override:
                total = max(total, total_parts_override.get(dieu_id, 0))
            retrieved = retrieved_part_counts.get(dieu_id, 0)
            if total > 1 and retrieved < total:
                results.append({"dieu_id": dieu_id, "total_parts": total, "retrieved_parts": retrieved})
        return results

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

    def check_retrieval_completeness(
        self,
        focus_dieu_ids: list[str],
        all_retrieved_dieu_ids: Optional[list[str]] = None,
        retrieved_part_counts: Optional[dict[str, int]] = None,
        total_parts_override: Optional[dict[str, int]] = None,
    ) -> dict:
        """
        Kiểm tra tổng thể độ đầy đủ của retrieval (retrieval chỉ có top-k chunk
        thô, KHÔNG expand — xem node_critic_check.py). Xét 3 loại thiếu sót:
        - Same-Điều (chế tài): HanhVi trong 1 Điều đã lấy có CẢ hình phạt chính
          lẫn bổ sung trong graph -> 2 phần này thường nằm ở 2 Khoản khác nhau.
        - Same-Điều (tổng quát): Điều trong focus có nhiều Khoản/Điểm hơn số
          phần thực sự đã retrieve được — KHÔNG giới hạn riêng chế tài, áp dụng
          cho MỌI cấu trúc nhiều Khoản (danh sách miễn trừ, điều kiện, v.v.).
        - Cross-Điều: Điều trong focus_dieu_ids tham chiếu sang Điều khác mà
          all_retrieved_dieu_ids KHÔNG có (chưa thực sự lấy được).
        Caller (node_critic_check) chịu trách nhiệm FETCH toàn văn Điều còn
        thiếu dựa trên kết quả này.

        Args:
            focus_dieu_ids: id Dieu quét completeness (thường đã lọc theo điểm số).
            all_retrieved_dieu_ids: TOÀN BỘ id Dieu retrieval thực sự đã lấy —
                dùng để tránh báo nhầm "thiếu" 1 Điều đã có sẵn (chỉ bị lọc khỏi
                focus vì điểm thấp). Mặc định = focus_dieu_ids nếu không truyền.
            retrieved_part_counts: {dieu_id: số Khoản/Điểm thực sự đã retrieve}
                — nếu không truyền, bỏ qua check "same-Điều tổng quát" (chỉ còn
                check chế tài kép + cross-Điều như trước).
            total_parts_override: xem docstring find_structurally_incomplete_dieu
                — nguồn đếm tổng số Khoản/Điểm ĐỘC LẬP với Neo4j (vd từ Qdrant),
                dùng LẤY MAX để tránh bỏ sót khi Neo4j ingest thiếu cạnh CO_KHOAN.

        Returns:
            dict với:
            - is_complete: bool
            - missing_dieu_ids: list id Điều cần retrieve thêm (THAM_CHIEU 2 chiều)
            - missing_references: list chi tiết THAM_CHIEU bị thiếu
            - compound_penalty_behaviors: list HanhVi có cả chính+bổ sung, cần lấy lại toàn văn Điều
            - structurally_incomplete_dieu: list Điều nhiều Khoản/Điểm nhưng chưa lấy đủ phần
            - suggestions: list hành động cần làm
        """
        missing_refs = self.find_missing_references(focus_dieu_ids, all_retrieved_dieu_ids)
        compound_penalties = self.find_compound_penalty_behaviors(focus_dieu_ids)
        structurally_incomplete = (
            self.find_structurally_incomplete_dieu(focus_dieu_ids, retrieved_part_counts, total_parts_override)
            if retrieved_part_counts is not None else []
        )

        suggestions = []
        missing_dieu_ids = set()
        # Điều đã được compound_penalty phát hiện rồi thì khỏi lặp lại lý do
        # tổng quát cho cùng Điều đó (tránh 2 suggestion trùng hành động).
        already_flagged_dieu = set()

        for ref in missing_refs:
            dieu_id = ref.get("missing_dieu_id")
            if dieu_id:
                missing_dieu_ids.add(dieu_id)
                suggestions.append({
                    "action": "retrieve",
                    "dieu_id": dieu_id,
                    "reason": ref.get("reason", ""),
                })

        for cp in compound_penalties:
            dieu_id = cp.get("dieu_id")
            if dieu_id:
                already_flagged_dieu.add(dieu_id)
                suggestions.append({
                    "action": "retrieve",
                    "dieu_id": dieu_id,
                    "reason": f"Hành vi '{cp.get('hanh_vi_mo_ta', '')}' trong Điều {dieu_id} có CẢ hình phạt "
                              f"chính lẫn bổ sung trong tri thức đồ thị — 2 phần này thường nằm ở 2 Khoản "
                              f"khác nhau nên top-k dễ chỉ lấy được 1 phần, cần lấy lại toàn văn Điều.",
                })

        for si in structurally_incomplete:
            dieu_id = si.get("dieu_id")
            if dieu_id and dieu_id not in already_flagged_dieu:
                suggestions.append({
                    "action": "retrieve",
                    "dieu_id": dieu_id,
                    "reason": f"Điều {dieu_id} có {si.get('total_parts')} Khoản/Điểm trong graph nhưng chỉ "
                              f"retrieve được {si.get('retrieved_parts')} phần — nội dung liên quan có thể "
                              f"nằm ở Khoản/Điểm khác chưa lấy được, cần lấy lại toàn văn Điều.",
                })

        return {
            "is_complete": len(missing_dieu_ids) == 0 and len(compound_penalties) == 0 and len(structurally_incomplete) == 0,
            "missing_dieu_ids": list(missing_dieu_ids),
            "missing_references": missing_refs,
            "compound_penalty_behaviors": compound_penalties,
            "structurally_incomplete_dieu": structurally_incomplete,
            "suggestions": suggestions,
        }

    def format_graph_context_for_llm(self, retrieved_dieu_ids: list[str]) -> str:
        """
        Format graph context thành text để đưa vào prompt của Critic Agent LLM.
        """
        completeness = self.check_retrieval_completeness(retrieved_dieu_ids)

        lines = ["=== KNOWLEDGE GRAPH ANALYSIS ===\n"]

        if completeness["is_complete"]:
            lines.append("✅ Retrieval đầy đủ: Không phát hiện thông tin bị thiếu từ graph.\n")
        else:
            lines.append(
                f"⚠️ Phát hiện {len(completeness['missing_dieu_ids'])} Điều cần retrieve thêm, "
                f"{len(completeness['compound_penalty_behaviors'])} hành vi có chế tài kép (chính+bổ sung) "
                f"cần lấy lại toàn văn Điều:\n"
            )

        if completeness["missing_references"]:
            lines.append("\n--- Các tham chiếu bị thiếu ---")
            for ref in completeness["missing_references"]:
                lines.append(f"• {ref.get('reason', '')}")

        if completeness["compound_penalty_behaviors"]:
            lines.append("\n--- Hành vi có chế tài kép chính+bổ sung (cần lấy lại toàn văn Điều) ---")
            for p in completeness["compound_penalty_behaviors"]:
                lines.append(
                    f"• Điều {p.get('dieu_id', '')} - Hành vi: {p.get('hanh_vi_mo_ta', '')}\n"
                    f"  Chính: {p.get('mo_ta_hinh_phat_chinh') or '(không có trong graph)'}\n"
                    f"  Bổ sung: {p.get('mo_ta_hinh_phat_bo_sung') or '(không có trong graph)'}"
                )

        if completeness["suggestions"]:
            lines.append("\n--- Khuyến nghị ---")
            for i, sug in enumerate(completeness["suggestions"], 1):
                lines.append(f"{i}. [{sug['action']}] {sug['reason']} (Điều: {sug['dieu_id']})")

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
