import logging
from typing import List

logger = logging.getLogger(__name__)


def is_candidate_relevant(query: str, candidate_content: str, *, llm_client) -> bool:
    """
    Cổng lọc NGỮ NGHĨA thật (dùng LLM) trước khi fetch/bơm 1 ứng viên vào
    ngữ cảnh — đây là mảnh còn thiếu của Critic Agent: 3 check trong
    critic_query.py (missing_references, compound_penalty, structurally_incomplete)
    CHỈ dựa vào tín hiệu CẤU TRÚC/TOPOLOGY của graph (đếm Khoản, cạnh
    THAM_CHIEU, cạnh CheTai) — hoàn toàn KHÔNG biết nội dung ứng viên có thật
    sự nói về ĐÚNG hành vi trong câu hỏi hay không. critic_score_ratio cũng
    không giúp gì ở đây vì nó tái dùng lại chính điểm retrieval gốc (thứ đã
    để lọt Điều nhiễu vào top-k từ đầu), không phải 1 bộ lọc độc lập.

    Cố ý ĐẶT Ở ĐÂY (trước khi fetch) thay vì dặn dò trong prompt sinh câu trả
    lời cuối (đã thử ở regenerate và REVERT — xem lịch sử đo: model 7B không
    đủ tin cậy tự lọc khi phải làm đồng thời 2 việc — vừa viết câu trả lời
    vừa thẩm định nhiều khối nội dung trộn lẫn trong 1 ngữ cảnh lớn,
    completeness_rate giảm từ 0.6 xuống 0.317 trên bộ test thật). Tách thành
    1 lệnh gọi LLM NHỎ, CHUYÊN ĐÚNG 1 VIỆC (yes/no trên ĐÚNG 1 ứng viên, không
    lẫn việc sinh văn bản) — cùng dạng câu hỏi phân loại đơn giản đã kiểm
    chứng hoạt động ổn định ở judge_fact_covered (score_evaluation.py). Nhờ
    lọc TRƯỚC khi fetch, ứng viên bị loại sẽ KHÔNG bao giờ xuất hiện trong
    graph_context — vừa giảm nhiễu vừa giảm kích thước ngữ cảnh so với
    article_expand (vốn không lọc gì cả).

    LƯU Ý (đã thử và sửa, phát hiện qua test trên bộ đa dạng 4 category):
    bản prompt đầu tiên chỉ hỏi "có nói về ĐÚNG hành vi/tình huống" — khung
    "hành vi" này khớp tốt với câu hỏi kiểu chế tài/xử phạt (nhóm
    same_dieu_compound_penalty) nhưng gây FALSE NEGATIVE có hệ thống với câu
    hỏi thủ tục/cấu trúc/quyền lợi (nhóm structural_multi_part, cross_reference)
    — quan sát cụ thể: Điều 65 "Kết quả hòa giải" (nêu đúng nội dung văn bản
    hòa giải cần có) bị gate loại dù khớp hoàn toàn câu hỏi, chỉ vì nó không
    mô tả 1 "hành vi vi phạm". Sửa bằng cách mở rộng tiêu chí sang "cần
    thiết/liên quan để trả lời đúng câu hỏi" nói chung, không giới hạn ở
    hành vi vi phạm.
    """
    if not candidate_content:
        return False
    prompt = (
        "Bạn là trợ lý kiểm tra độ liên quan, CHỈ làm đúng 1 việc: xác định 1 đoạn văn bản pháp luật có "
        "THỰC SỰ CẦN THIẾT để trả lời ĐÚNG câu hỏi hay không. Có thể là quy định về đúng hành vi/tình huống "
        "được hỏi (nếu câu hỏi về chế tài, xử phạt), HOẶC đúng chủ thể/thủ tục/điều kiện/thành phần nội dung "
        "mà câu hỏi yêu cầu (nếu câu hỏi về quyền, nghĩa vụ, quy trình, cấu trúc văn bản...) — KHÔNG bắt buộc "
        "phải là hành vi vi phạm. Chỉ trả lời KHÔNG liên quan nếu đoạn văn bản nói về CHỦ ĐỀ KHÁC hẳn, hoặc "
        "hành vi/đối tượng tương tự nhưng khác điều kiện, khác loại giấy phép/dịch vụ so với câu hỏi.\n\n"
        f"CÂU HỎI của người dùng: {query}\n\n"
        f"ĐOẠN VĂN BẢN ỨNG VIÊN (hệ thống tự động phát hiện qua cấu trúc Knowledge Graph, CHƯA xác nhận có "
        f"thật sự liên quan):\n{candidate_content[:3000]}\n\n"
        "Đoạn văn bản này có cần thiết/liên quan để trả lời ĐÚNG câu hỏi trên không? "
        "Chỉ trả lời đúng 1 từ: 'yes' hoặc 'no'."
    )
    try:
        resp = llm_client.invoke(prompt, tag="gate")
        return "yes" in resp.content.strip().lower()
    except Exception as e:
        logger.warning(f"Relevance gate lỗi, mặc định coi là liên quan (fail-open): {e}")
        return True


def any_chunk_relevant(query: str, texts: List[str], *, llm_client) -> bool:
    """
    Kiểm tra từng chunk RIÊNG LẺ (không gộp chung) — chỉ cần 1 chunk khớp là
    đủ coi Điều đó liên quan. BẮT BUỘC kiểm tra riêng lẻ thay vì nối chuỗi
    rồi hỏi 1 lần: 1 Điều thường có NHIỀU chunk được retrieve ứng với NHIỀU
    Khoản/hành vi khác nhau trong cùng Điều — nếu gộp chung, 1 chunk lạc đề
    (vd Khoản 1 nói về nghĩa vụ khác) có thể làm loãng/nhiễu tín hiệu của
    chunk ĐÚNG (vd Khoản 3 đúng hành vi câu hỏi), khiến LLM đánh giá sai
    "không liên quan" cho cả Điều dù có ít nhất 1 chunk thật sự khớp (bug đã
    quan sát được: Điều 99 có chunk Khoản 3 đúng + chunk Khoản 1 khác hành vi
    gộp chung bị loại nhầm).
    """
    for text in texts:
        if text and is_candidate_relevant(query, text, llm_client=llm_client):
            return True
    return False
