"""
focus_dieu.py
==============
Hàm thuần (không phụ thuộc state ẩn) tính tập Điều "trong phạm vi xem xét" —
lọc theo tỷ lệ điểm so với Điều điểm cao nhất, cap tối đa max_dieu Điều.

Dùng chung bởi agent_retrieval (suy article_expand_dieu_ids từ tập rộng, dùng
article_expand_score_ratio + max_dieu=top_k) và agent_critic (suy focus_dieu_ids
cho completeness check, dùng critic_score_ratio + critic_max_dieu) — MỖI caller
BẮT BUỘC truyền tường minh score_ratio/max_dieu của riêng mình (không có giá trị
mặc định ẩn), để 2 nơi gọi không vô tình lẫn ngưỡng của nhau.

Lý do dùng TỶ LỆ thay vì cắt cứng theo thứ hạng: nếu top-k lẫn phải Điều không
thật sự liên quan (nhiễu retrieval, điểm thấp hẳn so với Điều đứng đầu), mỗi
Điều đó lại kéo theo tham chiếu RIÊNG của nó -> càng nhiều Điều bị kiểm tra,
càng dễ fetch phải nội dung lạc đề, khiến câu trả lời cuối bị "Lost in the
Middle" (đã quan sát thực tế: Điều 74 về nhãn hiệu bị fetch nhầm khi hỏi về
sáng chế, do lọt top-k với điểm thấp). Ngược lại, nếu có NHIỀU Điều cùng đạt
điểm cao gần nhau (câu hỏi thực sự cần đối chiếu ≥2 Điều), tỷ lệ này vẫn giữ
được TẤT CẢ — không bỏ sót như cách cắt cứng "chỉ top-1" trước đây. max_dieu
là van an toàn, tránh trường hợp hiếm khi quá nhiều Điều cùng đạt điểm cao.
"""

from typing import Dict, List


def compute_focus_dieu_ids(
    retrieved_dieu_ids: List[str],
    dieu_scores: Dict[str, float],
    score_ratio: float,
    max_dieu: int,
) -> List[str]:
    if not retrieved_dieu_ids:
        return []
    # Thứ tự Điều có thể đã qua cross-encoder/diversity selection nên không
    # được giả định score gốc của phần tử đầu tiên là lớn nhất.
    top_score = max(
        (float(dieu_scores.get(dieu_id, 0.0)) for dieu_id in retrieved_dieu_ids),
        default=0.0,
    )
    focus_dieu_ids = []
    for d in retrieved_dieu_ids:
        if top_score > 0 and dieu_scores.get(d, 0.0) < top_score * score_ratio:
            continue
        focus_dieu_ids.append(d)
        if len(focus_dieu_ids) >= max_dieu:
            break
    return focus_dieu_ids
