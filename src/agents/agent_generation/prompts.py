from typing import List


def format_context_block(context_texts: List[str]) -> str:
    context_text = "\n\n".join(f"Tài liệu {i+1} (độ liên quan giảm dần):\n{t}" for i, t in enumerate(context_texts))
    return context_text or "Không tìm thấy tài liệu liên quan."


def build_answer_prompt(query: str, context_text: str) -> str:
    """
    Prompt sinh câu trả lời — dùng CHUNG cho cả 3 kịch bản (naive/
    article_expand qua node_generate_final; critic qua node_generate_draft
    VÀ qua node_generate_final khi mode="critic" ở bước regenerate — draft
    và regenerate BẮT BUỘC dùng giống nhau để không lẫn biến số giữa "sinh
    nháp" và "sinh lại").

    LƯU Ý (đã thử và REVERT — thí nghiệm CoT 4 bước + few-shot 1 ví dụ):
    từng đổi hẳn sang bắt model viết "Suy luận" theo 4 bước trước khi kết
    luận, kỳ vọng giảm nhầm lẫn Khoản/văn bản. Đo trên bộ test thật cho
    kết quả TỆ HƠN hẳn ở CẢ 3 mode: completeness_rate cat1 (10 mẫu) giảm
    mạnh — article_expand 0.642 -> 0.442, critic 0.758 -> 0.475. Nguyên
    nhân: bước "lọc Tài liệu" khiến model 7B quá tay, tự đánh giá 1 Tài
    liệu ĐÚNG là "không đề cập trực tiếp"/"không có quy định cụ thể" chỉ
    vì cách diễn đạt câu hỏi không khớp Y HỆT câu chữ trong Tài liệu. Đây
    là CÙNG BẢN CHẤT lỗi với thí nghiệm "nghi ngờ graph_context" đã revert
    ở regenerate trước đó (completeness giảm 0.6 -> 0.317) — chỉ khác ở
    chỗ lần này lỗi đến từ 1 bước suy luận có cấu trúc thay vì 1 câu dặn
    trực tiếp. Kết luận: KHÔNG dùng CoT/few-shot dạng "tự lọc rồi mới trả
    lời" cho model 7B ở tác vụ này; giữ nguyên prompt ngắn, ra lệnh trực
    tiếp cho CẢ 3 mode.
    """
    return (
        "Bạn là một chuyên gia tư vấn pháp lý xuất sắc. "
        "Hãy trả lời câu hỏi của người dùng một cách chính xác, đầy đủ và dễ hiểu nhất dựa trên NGỮ CẢNH được cung cấp.\n\n"
        "Các Tài liệu được sắp theo thứ tự ĐỘ LIÊN QUAN GIẢM DẦN — Tài liệu 1 khớp với hành vi trong câu hỏi nhất. "
        "Chỉ dùng Tài liệu nếu nó nói về ĐÚNG hành vi trong câu hỏi; đừng lẫn sang hành vi tương tự nhưng khác. "
        "Mỗi Tài liệu có ghi rõ TÊN VĂN BẢN trong ngoặc vuông ở đầu — luôn kiểm tra kỹ tên văn bản đó có khớp "
        "ĐÚNG văn bản pháp luật được hỏi hay không TRƯỚC KHI dùng nội dung; nhiều văn bản luật khác nhau có "
        "điều khoản CẤU TRÚC GIỐNG NHAU (vd \"Hiệu lực thi hành\", \"Giải thích từ ngữ\", \"Phạm vi điều "
        "chỉnh\") — đừng vì thấy cấu trúc/hành văn quen thuộc mà tưởng nhầm là văn bản khác hoặc phủ nhận "
        "thông tin đúng chỉ vì có Tài liệu tương tự của văn bản khác đứng gần đó.\n\n"
        "Trong ngữ cảnh có thể có Điều chỉ DẪN CHIẾU sang Điều khác, bản thân nó không chứa quy định cụ thể "
        "(vd \"việc ... được thực hiện theo quy định tại các điều 15, 16, 17 và 18 của Luật này\"). Trước khi "
        "trả lời, hãy rà xem ngữ cảnh có Điều nào dẫn chiếu tới Điều mà bạn định dùng làm căn cứ hay không; "
        "nếu có, BẮT BUỘC nêu cả hai theo dạng \"Điều <dẫn chiếu> dẫn chiếu đến Điều <chi tiết>\" — Điều dẫn "
        "chiếu cho biết VÌ SAO quy định đó áp dụng vào đúng tình huống người hỏi đang gặp, còn Điều được dẫn "
        "chiếu mới là nội dung trả lời.\n"
        "Nếu NGỮ CẢNH không chứa thông tin để trả lời, hãy nói rằng bạn không biết, đừng tự bịa ra.\n"
        "Một hành vi vi phạm có thể có cả hình phạt chính LẪN hình phạt bổ sung/biện pháp khắc phục hậu quả — "
        "đọc kỹ TOÀN BỘ ngữ cảnh và liệt kê ĐẦY ĐỦ tất cả các loại chế tài liên quan, không chỉ hình phạt chính.\n\n"
        f"================ NGỮ CẢNH ================\n{context_text}\n\n"
        f"Câu hỏi: {query}\n\n"
        # Nhắc lại NGAY SAU câu hỏi (không chỉ ở phần dặn dò đầu prompt): khối
        # ngữ cảnh ở giữa dài vài nghìn token nên chỉ dẫn đặt trước nó bị loãng
        # — đã đo: đặt trước, model bỏ qua Điều dẫn chiếu ở cả 2 lần thử.
        "Nhắc lại: rà kỹ ngữ cảnh xem có Điều nào DẪN CHIẾU tới Điều bạn dùng làm căn cứ không "
        "(kiểu \"... được thực hiện theo quy định tại các điều ... của Luật này\"); nếu có thì phải "
        "nêu cả Điều dẫn chiếu lẫn Điều chứa quy định chi tiết."
    )
