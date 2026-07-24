import logging

logger = logging.getLogger(__name__)


def make_node_agent_regenerate(generation_agent):
    """
    Kịch bản 3, bước 3: sinh LẠI câu trả lời cuối cùng TỪ ĐẦU bằng đúng
    prompt/cơ chế của generate_final (context_texts gốc + graph_context Critic
    Agent vừa fetch/lọc) — KHÔNG "sửa" draft_response bằng cách yêu cầu LLM hợp
    nhất 2 khối văn bản riêng biệt (draft cũ + phần bổ sung mới).

    LƯU Ý (đã thử và bỏ, phát hiện qua test thật): cách cũ đưa CẢ draft_response
    LẪN graph_context, yêu cầu LLM "tích hợp tự nhiên" 2 phần lại — đây là tác
    vụ khó hơn hẳn "đọc ngữ cảnh rồi trả lời thẳng": model 7B phải vừa đọc lại
    draft, vừa đọc graph_context mới, vừa suy luận phần nào đã có/chưa có/
    trùng lặp rồi hợp nhất — dễ ĐÁNH RƠI thông tin dù graph_context đã có ĐỦ
    và ĐÚNG (quan sát cụ thể: graph_context có đủ 9 Khoản Điều 98, gồm cả
    Khoản 5 "buộc thu hồi/hoàn trả tên miền", nhưng câu trả lời HỢP NHẤT lại bỏ
    sót đúng Khoản 5 đó — trong khi sinh lại từ đầu với CÙNG NGỮ CẢNH, y hệt
    cách Kịch bản 1/2 làm, thì KHÔNG bỏ sót). Sinh lại từ đầu bằng chung 1 CƠ
    CHẾ SINH CÂU TRẢ LỜI CUỐI với Kịch bản 1/2 cũng giúp phép so sánh 3 kịch
    bản công bằng hơn: giờ cả 3 chỉ còn khác nhau đúng ở NỘI DUNG ngữ cảnh
    được đưa vào, không còn khác nhau ở CÁCH SINH câu trả lời từ ngữ cảnh đó.
    """
    def node(state: dict) -> dict:
        logger.info("Critic Agent: sinh lại câu trả lời cuối từ ngữ cảnh gốc + phần Critic Agent bổ sung...")
        return generation_agent.run_final(state)
    return node
