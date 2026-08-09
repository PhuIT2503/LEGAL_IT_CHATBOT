import logging
import re
from typing import Dict, Optional

from src.knowledge_graph.graph_builder import to_dieu_node_id
from src.agents.common.focus_dieu import compute_focus_dieu_ids
from src.agents.agent_critic.relevance_gate import is_candidate_relevant, any_chunk_relevant
from src.agents.agent_critic.state import CriticState

logger = logging.getLogger(__name__)

# Mỗi chunk đã được gắn sẵn tiền tố dạng "[Điều 37, Luật Giao dịch điện tử 2023]"
# (hoặc "[Khoản 1, Điều 16, ...]") lúc chunking — dùng lại làm nhãn đọc được cho
# người dùng thay vì id kỹ thuật (lu_t_giao_d_ch_i_n_t_2023_D37).
_DIEU_LABEL_RE = re.compile(r"^\[[^\]]*?Điều\s+([^,\]]+),\s*([^\]]+)\]")


def _dieu_label(dieu_id: str, content: Optional[str]) -> str:
    match = _DIEU_LABEL_RE.match((content or "").lstrip())
    if not match:
        return dieu_id
    return f"Điều {match.group(1).strip()} — {match.group(2).strip()}"


def critic_check_node(
    state: CriticState,
    *,
    llm_client,
    dieu_content_store,
    critic_query_engine,
    critic_score_ratio: float,
    critic_max_dieu: int,
) -> dict:
    """
    Kịch bản 3, bước 2 — trái tim của khóa luận: dùng Knowledge Graph để
    PHÁT HIỆN xem câu trả lời nháp (dựa trên top-k thuần) có khả năng đang
    thiếu gì không, rồi TỰ ĐI LẤY (fetch) CHÍNH XÁC phần còn thiếu đó — chứ
    KHÔNG bốc tràn lan như Kịch bản 2:

    1. Same-Điều (chế tài): một HanhVi trong Điều đã lấy có CẢ hình phạt
       chính lẫn bổ sung trong graph -> 2 phần này thường nằm ở 2 Khoản
       khác nhau -> fetch toàn văn CHÍNH Điều đó.
    2. Same-Điều (tổng quát): Điều đã lấy có nhiều Khoản/Điểm hơn số phần
       thực sự đã retrieve — KHÔNG giới hạn riêng chế tài, áp dụng cho MỌI
       cấu trúc nhiều Khoản (danh sách miễn trừ, điều kiện, v.v.) mà quan hệ
       HanhVi/CheTai không có mặt trong graph để dò riêng -> fetch toàn văn.
    3. Cross-Điều: Điều đã lấy có tham chiếu (THAM_CHIEU) sang Điều KHÁC mà
       top-k chưa lấy được -> fetch toàn văn Điều kia.
    4. Cross-Điều BẮC CẦU (multi-hop): sau khi fetch xong 1 Điều mới ở check
       #3, quét lại xem CHÍNH Điều mới đó có tham chiếu tiếp sang Điều khác
       nữa không (A -> B -> C), lặp tới khi không còn gì mới hoặc chạm giới
       hạn an toàn (max_hops=2, tổng số Điều fetch <= 2*critic_max_dieu) —
       tránh lặp vô hạn nếu tham chiếu vòng tròn (A<->B) hoặc 1 Điều bị quá
       nhiều Điều khác tham chiếu tới (hub document).

    Cả 4 check trên chỉ phát hiện qua TÍN HIỆU CẤU TRÚC graph (đếm Khoản,
    cạnh THAM_CHIEU/CheTai) — KHÔNG biết nội dung ứng viên có thật liên quan
    tới câu hỏi hay không. Trước khi fetch/bơm vào graph_context, MỖI ứng
    viên còn phải qua `is_candidate_relevant()` (cổng lọc ngữ nghĩa bằng
    LLM, 1 lệnh gọi nhỏ/ứng viên) — ứng viên bị từ chối sẽ KHÔNG xuất hiện
    trong ngữ cảnh cuối (ghi lại ở critic_report["rejected_by_relevance_gate"]).

    Nếu không phát hiện gì (graph_context rỗng) -> workflow sẽ đi thẳng sang
    finalize_draft, KHÔNG tốn thêm 1 lượt gọi LLM vô ích.

    LƯU Ý: chỉ kiểm tra completeness cho các Điều có điểm >= critic_score_ratio
    * điểm Điều cao nhất (KHÔNG cứng "chỉ top-1", KHÔNG chạy tràn lan cho mọi
    Điều trong top-k) — tránh việc 1 Điều nhiễu (lọt vào top-k với điểm thấp
    hẳn, không thật sự liên quan tới câu hỏi) kéo theo tham chiếu riêng của
    nó, làm loãng/lạc đề câu trả lời cuối; đồng thời vẫn giữ được TẤT CẢ nếu
    có nhiều Điều cùng đạt điểm cao gần nhau (câu hỏi thực sự cần đối chiếu
    nhiều Điều), không bỏ sót như cách cắt cứng theo thứ hạng.
    """
    logger.info("Critic Agent (Neo4j): đối chiếu câu trả lời nháp với Knowledge Graph...")
    query = state["query"]
    all_dieu_ids = state.get("retrieved_dieu_ids", [])
    dieu_scores = state.get("dieu_scores", {})

    # focus_dieu_ids: phạm vi QUÉT completeness — TÍNH CHUNG với
    # expand_full_article qua compute_focus_dieu_ids (xem docstring hàm đó —
    # bắt buộc 2 kịch bản dùng cùng 1 tập ứng viên để so sánh công bằng).
    # all_dieu_ids: TOÀN BỘ Điều retrieval THỰC SỰ đã lấy — dùng để biết 1
    # Điều tham chiếu tới có sẵn rồi hay chưa. Phải truyền riêng 2 tập này
    # cho check_retrieval_completeness, KHÔNG được dùng chung 1 tập đã lọc —
    # nếu không, 1 Điều bị lọc khỏi focus (vì điểm thấp) nhưng thực ra ĐÃ
    # được retrieve sẽ bị báo nhầm là "thiếu" (bug đã phát hiện qua test thật).
    focus_dieu_ids = compute_focus_dieu_ids(all_dieu_ids, dieu_scores, score_ratio=critic_score_ratio, max_dieu=critic_max_dieu)
    logger.info(f"Critic Agent: phạm vi kiểm tra {len(focus_dieu_ids)}/{len(all_dieu_ids)} Điều (score_ratio={critic_score_ratio}).")

    if not focus_dieu_ids:
        return {"graph_context": "", "critic_report": {}, "graph_fetched_dieu_ids": []}

    # Đếm số Khoản/Điểm DUY NHẤT (theo chunk_id) thực sự đã retrieve cho mỗi
    # Điều — dùng để phát hiện tổng quát "Điều nhiều Khoản nhưng chưa lấy đủ
    # phần" (không giới hạn riêng chế tài, xem check_retrieval_completeness).
    retrieved_part_counts: Dict[str, set] = {}
    dieu_to_retrieved_text: Dict[str, list] = {}
    for c in state.get("retrieved_chunks", []):
        d = to_dieu_node_id(c["van_ban_id_raw"], c["dieu_id_raw"])
        if d and c.get("chunk_id"):
            retrieved_part_counts.setdefault(d, set()).add(c["chunk_id"])
        if d and c.get("text"):
            dieu_to_retrieved_text.setdefault(d, []).append(c["text"])
    retrieved_part_counts = {d: len(chunk_ids) for d, chunk_ids in retrieved_part_counts.items()}

    # Đếm "tổng số phần" LẤY MAX giữa Neo4j (bên trong check_retrieval_completeness)
    # và Qdrant (dieu_content_store, độc lập, không qua LLM extraction) —
    # phòng Neo4j ingest thiếu cạnh CO_KHOAN cho 1 số Điều (xem
    # DieuContentStore._build_dieu_child_chunk_count và find_structurally_incomplete_dieu).
    total_parts_override = {d: dieu_content_store.child_chunk_count(d) for d in focus_dieu_ids}

    report = critic_query_engine.check_retrieval_completeness(
        focus_dieu_ids, all_dieu_ids, retrieved_part_counts, total_parts_override
    )
    graph_text = ""
    fetched_dieu_ids: set = set()

    rejected_dieu_ids: list = []
    # id Điều -> nhãn đọc được ("Điều 37 — Luật Giao dịch điện tử 2023"), để UI
    # hiển thị cho người dùng thay vì id kỹ thuật. Ghi cho CẢ Điều đã bổ sung
    # lẫn Điều bị relevance gate loại.
    dieu_labels: Dict[str, str] = {}

    def label_of(dieu_id: str, content: Optional[str] = None) -> str:
        """Nhãn đọc được, cache lại — dùng CẢ trong ngữ cảnh gửi cho LLM lẫn
        trong report hiển thị cho người dùng.

        QUAN TRỌNG (phát hiện qua test thật): trước đây header bơm vào ngữ cảnh
        nhúng thẳng id kỹ thuật (vd lu_t_giao_d_ch_i_n_t_2023_D37). LLM đọc phải
        chuỗi đó thì không có cách nào trích dẫn lại cho người dùng, nên bỏ qua
        luôn Điều được bổ sung dù nội dung nằm ngay trong ngữ cảnh — đã quan sát
        đúng ca Điều 37 (điều khoản dẫn chiếu) bị bỏ khỏi câu trả lời.
        """
        if dieu_id not in dieu_labels:
            if content is None:
                content = dieu_content_store.fetch_parent_content(dieu_id)
            dieu_labels[dieu_id] = _dieu_label(dieu_id, content)
        return dieu_labels[dieu_id]

    if report["missing_references"]:
        for ref in report["missing_references"]:
            missing_dieu_id = ref.get("missing_dieu_id")
            if not missing_dieu_id or missing_dieu_id in fetched_dieu_ids:
                continue
            content = dieu_content_store.fetch_parent_content(missing_dieu_id)
            label = label_of(missing_dieu_id, content)
            if content and not is_candidate_relevant(query, content, llm_client=llm_client):
                logger.info(f"Relevance gate: BỎ QUA Điều {missing_dieu_id} (missing_reference) — LLM đánh giá không liên quan tới câu hỏi.")
                rejected_dieu_ids.append(missing_dieu_id)
                continue

            related_id = ref.get("related_dieu_id")
            related = label_of(related_id) if related_id else None
            if related and ref.get("direction") == "incoming":
                # Điều vừa bổ sung DẪN CHIẾU tới Điều đã có sẵn — nói thẳng
                # quan hệ đó ra để câu trả lời dẫn được ĐỦ chuỗi căn cứ, thay
                # vì chỉ trích Điều chứa nội dung chi tiết.
                # Wording dưới đây đã ĐO THẬT: nhắc lại đầy đủ tên cả 2 Điều
                # (không rút gọn thành "Điều 16") thì câu trả lời mới thực sự
                # dẫn đủ chuỗi "Điều 37 dẫn chiếu Điều 16". Bản rút gọn cho
                # tiết kiệm token đã thử và bị model bỏ qua — đừng rút gọn lại.
                graph_text += (
                    f"[{label} — Critic Agent bổ sung qua Knowledge Graph: Điều này DẪN CHIẾU tới "
                    f"{related} (đã có trong ngữ cảnh), tức là căn cứ cho biết vì sao áp dụng "
                    f"{related} vào tình huống đang được hỏi. Nếu dùng {related} để trả lời thì "
                    f"nêu cả {label}.]\n"
                )
            elif related:
                graph_text += (
                    f"[{label} — Critic Agent bổ sung qua Knowledge Graph: {related} (đã có trong "
                    f"ngữ cảnh) tham chiếu tới Điều này.]\n"
                )
            else:
                graph_text += f"[{label} — Critic Agent (Knowledge Graph) phát hiện còn thiếu.]\n"
            if content:
                graph_text += f"{content}\n\n"
                fetched_dieu_ids.add(missing_dieu_id)
            else:
                graph_text += "(Không lấy được toàn văn — Qdrant parent collection chưa có Điều này.)\n\n"

    if report["compound_penalty_behaviors"]:
        for p in report["compound_penalty_behaviors"]:
            dieu_id = p.get("dieu_id")
            if not dieu_id or dieu_id in fetched_dieu_ids:
                continue
            content = dieu_content_store.fetch_parent_content(dieu_id)
            # Dieu nay DA co mat trong focus qua retrieval (co chunk that su
            # da retrieve) - relevance gate phai so voi CHINH CAC CHUNK DO
            # (ngan, cu the, kiem tra RIENG LE qua any_chunk_relevant), KHONG
            # so voi toan van Dieu cat ngan tu dau (gay false-negative loai
            # nham Dieu DUNG - da quan sat o D105/D102) VA KHONG gop chung
            # nhieu chunk lam 1 (gay false-negative khac khi 1 chunk lac de
            # lam nhieu tin hieu chunk dung - da quan sat o D99).
            anchor_texts = dieu_to_retrieved_text.get(dieu_id) or ([content] if content else [])
            label = label_of(dieu_id, content or (anchor_texts[0] if anchor_texts else None))
            if anchor_texts and not any_chunk_relevant(query, anchor_texts, llm_client=llm_client):
                logger.info(f"Relevance gate: BỎ QUA Điều {dieu_id} (compound_penalty) — LLM đánh giá không liên quan tới câu hỏi.")
                rejected_dieu_ids.append(dieu_id)
                continue
            graph_text += (
                f"[Toàn văn {label} do Critic Agent (Knowledge Graph) tự bổ sung — phát hiện hành vi "
                f"'{p.get('hanh_vi_mo_ta', '')}' có CẢ hình phạt chính lẫn bổ sung/biện pháp khắc phục hậu quả "
                f"(2 phần này thường nằm ở Khoản khác nhau, top-k dễ chỉ trúng 1 phần) — đảm bảo không bỏ sót]\n"
            )
            if content:
                graph_text += f"{content}\n\n"
                fetched_dieu_ids.add(dieu_id)
            else:
                graph_text += "(Không lấy được toàn văn Điều này từ Qdrant.)\n\n"

    if report["structurally_incomplete_dieu"]:
        for si in report["structurally_incomplete_dieu"]:
            dieu_id = si.get("dieu_id")
            if not dieu_id or dieu_id in fetched_dieu_ids:
                continue
            content = dieu_content_store.fetch_parent_content(dieu_id)
            # Tuong tu compound_penalty o tren: kiem tra RIENG LE tung chunk
            # DA retrieve (khong gop chung, khong so toan van cat ngan).
            anchor_texts = dieu_to_retrieved_text.get(dieu_id) or ([content] if content else [])
            label = label_of(dieu_id, content or (anchor_texts[0] if anchor_texts else None))
            if anchor_texts and not any_chunk_relevant(query, anchor_texts, llm_client=llm_client):
                logger.info(f"Relevance gate: BỎ QUA Điều {dieu_id} (structurally_incomplete) — LLM đánh giá không liên quan tới câu hỏi.")
                rejected_dieu_ids.append(dieu_id)
                continue
            graph_text += (
                f"[Toàn văn {label} do Critic Agent (Knowledge Graph) tự bổ sung — Điều này có "
                f"{si.get('total_parts')} Khoản/Điểm nhưng chỉ retrieve được {si.get('retrieved_parts')} phần, "
                f"có thể còn nội dung liên quan ở Khoản/Điểm khác chưa lấy được]\n"
            )
            if content:
                graph_text += f"{content}\n\n"
                fetched_dieu_ids.add(dieu_id)
            else:
                graph_text += "(Không lấy được toàn văn Điều này từ Qdrant.)\n\n"

    # VÒNG LẶP BỔ SUNG (multi-hop tham chiếu): Điều VỪA fetch ở 3 check trên
    # có thể TỰ NÓ tham chiếu sang 1 Điều khác nữa (A -> B -> C) mà vòng đầu
    # chưa biết tới — find_missing_references() ở trên chỉ quét THAM_CHIEU
    # xuất phát từ focus_dieu_ids BAN ĐẦU, chưa quét từ B (Điều vừa fetch).
    # Sau khi fetch B, quét lại THAM_CHIEU xuất phát từ CHÍNH B để tìm C,
    # lặp tới khi không còn gì mới hoặc chạm max_hops. Đây CHÍNH LÀ phần
    # "đánh giá tiếp, khi nào đủ thì trả kết quả cuối cùng" — nhưng lặp lại
    # CHECK CẤU TRÚC (khách quan, dựa trên graph) chứ KHÔNG lặp lại việc để
    # LLM tự phán xét câu trả lời của chính nó (đã thử kiểu đó ở
    # regenerate và REVERT vì model 7B tự vứt bỏ luôn cả thông tin
    # ĐÚNG — xem lịch sử đo). Giới hạn max_hops + MAX_TOTAL_FETCH để tránh
    # lặp vô hạn nếu 2 Điều tham chiếu vòng tròn lẫn nhau (vd A<->B) hoặc 1
    # Điều được tham chiếu bởi quá nhiều Điều khác (hub document).
    all_known_dieu_ids = set(all_dieu_ids) | fetched_dieu_ids
    frontier = list(fetched_dieu_ids)
    multi_hop_refs: list = []
    max_hops = 2
    MAX_TOTAL_FETCH = critic_max_dieu * 2
    for _hop in range(max_hops):
        if not frontier or len(fetched_dieu_ids) >= MAX_TOTAL_FETCH:
            break
        deeper_refs = critic_query_engine.find_missing_references(frontier, list(all_known_dieu_ids))
        next_frontier = []
        for ref in deeper_refs:
            if len(fetched_dieu_ids) >= MAX_TOTAL_FETCH:
                break
            missing_dieu_id = ref.get("missing_dieu_id")
            if not missing_dieu_id or missing_dieu_id in all_known_dieu_ids:
                continue
            all_known_dieu_ids.add(missing_dieu_id)
            content = dieu_content_store.fetch_parent_content(missing_dieu_id)
            label = label_of(missing_dieu_id, content)
            if content and not is_candidate_relevant(query, content, llm_client=llm_client):
                logger.info(f"Relevance gate: BỎ QUA Điều {missing_dieu_id} (tham chiếu bắc cầu) — LLM đánh giá không liên quan tới câu hỏi.")
                rejected_dieu_ids.append(missing_dieu_id)
                continue
            related_id = ref.get("related_dieu_id")
            related = label_of(related_id) if related_id else ", ".join(label_of(d) for d in frontier)
            graph_text += (
                f"[{label} — Critic Agent bổ sung qua Knowledge Graph (tham chiếu bắc cầu): "
                f"liên hệ với {related} đã có trong ngữ cảnh.]\n"
            )
            if content:
                graph_text += f"{content}\n\n"
                fetched_dieu_ids.add(missing_dieu_id)
                next_frontier.append(missing_dieu_id)
            else:
                graph_text += "(Không lấy được toàn văn — Qdrant parent collection chưa có Điều này.)\n\n"
        multi_hop_refs.extend(deeper_refs)
        frontier = next_frontier

    if multi_hop_refs:
        report["multi_hop_references"] = multi_hop_refs

    # PHÒNG NGỪA "LOST IN THE MIDDLE" liên văn bản (phát hiện qua test thật
    # — case cat4_01 "Luật Dữ liệu 2024 hiệu lực thi hành"): khi ngữ cảnh
    # top-k trộn lẫn NHIỀU văn bản khác nhau có điều khoản CẤU TRÚC GIỐNG
    # NHAU (vd "Hiệu lực thi hành" ở 3 luật khác nhau), draft dễ tự PHỦ
    # NHẬN thông tin ĐÚNG dù nó nằm ngay trong context — chỉ vì bị nhiễu
    # bởi đoạn tương tự của văn bản khác đứng gần đó. Đây KHÔNG phải lỗi
    # "thiếu dữ liệu" (3 check completeness ở trên đúng khi báo is_complete)
    # mà là nguy cơ "hiểu sai dữ liệu đã có" — 1 loại lỗi khác hẳn, cần tín
    # hiệu khác hẳn để phát hiện (không phải hỏi LLM tự nghi ngờ — đã thử
    # kiểu đó ở regenerate và REVERT vì phản tác dụng). Tín hiệu dùng ở
    # đây KHÁCH QUAN, không qua LLM: nếu top-k có chunk từ >= 2 văn bản
    # khác nhau VÀ không có gì để fetch qua 4 check trên (sắp đi thẳng
    # finalize_draft, dùng nguyên draft) -> chủ động LẶP LẠI toàn văn Điều
    # điểm cao nhất, buộc quy trình rẽ sang regenerate (sinh lại có ngữ
    # cảnh được củng cố) thay vì dùng thẳng draft chưa được kiểm chứng lại
    # — mô phỏng đúng cơ chế "lặp lại thông tin đúng" mà article_expand có
    # sẵn (luôn re-dump toàn văn), nhưng CÓ ĐIỀU KIỆN, chỉ khi thật sự có
    # nguy cơ nhầm lẫn liên văn bản, không làm tràn lan như article_expand.
    if not fetched_dieu_ids and all_dieu_ids:
        distinct_van_ban = {
            c.get("van_ban_id_raw") for c in state.get("retrieved_chunks", []) if c.get("van_ban_id_raw")
        }
        if len(distinct_van_ban) >= 2:
            top_dieu_id = all_dieu_ids[0]
            content = dieu_content_store.fetch_parent_content(top_dieu_id)
            if content:
                logger.info(
                    f"Critic Agent: ngữ cảnh trộn {len(distinct_van_ban)} văn bản khác nhau, không phát hiện "
                    f"thiếu gì về cấu trúc — chủ động lặp lại toàn văn Điều {top_dieu_id} để phòng nhầm lẫn."
                )
                graph_text += (
                    f"[Toàn văn {label_of(top_dieu_id, content)} — Critic Agent CHỦ ĐỘNG LẶP LẠI để phòng ngừa nhầm lẫn: "
                    f"ngữ cảnh có nội dung từ {len(distinct_van_ban)} văn bản khác nhau, dễ nhầm điều khoản "
                    f"cấu trúc giống nhau giữa các văn bản (vd \"Hiệu lực thi hành\")]\n{content}\n\n"
                )
                fetched_dieu_ids.add(top_dieu_id)
                report["reinforced_top_dieu_multi_van_ban"] = top_dieu_id

    if rejected_dieu_ids:
        report["rejected_by_relevance_gate"] = rejected_dieu_ids
    if dieu_labels:
        report["dieu_labels"] = dieu_labels

    return {"graph_context": graph_text, "critic_report": report, "graph_fetched_dieu_ids": list(fetched_dieu_ids)}
