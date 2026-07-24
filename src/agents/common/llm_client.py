"""
llm_client.py
==============
Wrapper DUY NHẤT quanh việc gọi LLM — mọi agent (router, generation, critic)
dùng CHUNG 1 instance của LLMClient để cộng dồn token usage vào cùng 1 chỗ
(tổng + tách riêng theo tag), thay vì mỗi agent tự đếm riêng.

`llm` là thuộc tính có thể gán lại BẤT CỨ LÚC NÀO (vd app.py đổi model LLM
theo lựa chọn người dùng mỗi request) — invoke() luôn đọc self.llm tại thời
điểm gọi, không cache lại instance cũ.
"""

from typing import Dict


class LLMClient:
    def __init__(self, llm):
        self.llm = llm
        self.token_usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "call_count": 0}
        self.token_usage_by_tag: Dict[str, Dict[str, int]] = {}

    def reset_usage(self) -> None:
        """Reset đếm token — gọi ở đầu mỗi ChatbotWorkflow.run(), mỗi câu hỏi độc lập, không cộng dồn qua các câu."""
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "call_count": 0}
        self.token_usage_by_tag = {}

    def invoke(self, prompt: str, tag: str = "other"):
        """Gọi LLM và cộng dồn token usage vào self.token_usage (tổng) VÀ
        self.token_usage_by_tag[tag] (tách riêng theo loại lệnh gọi).

        tag dùng để xác định lệnh gọi nào THỰC SỰ sinh ra câu trả lời cuối cùng:
        - "router"/"gate": không bao giờ là câu trả lời cuối.
        - "draft": là câu trả lời cuối CHỈ KHI critic không phát hiện thiếu gì
          (finalize_draft — không có lệnh regenerate nào chạy thêm).
        - "final_generate": generate_final — luôn LÀ câu trả lời cuối bất cứ khi
          nào được gọi (naive/article_expand gọi trực tiếp; critic gọi qua
          regenerate khi phát hiện thiếu).
        - "chit_chat": câu trả lời cuối khi câu hỏi được route thành chit-chat.
        """
        resp = self.llm.invoke(prompt)
        usage = None
        if getattr(resp, "usage_metadata", None):
            um = resp.usage_metadata
            usage = (um.get("input_tokens", 0), um.get("output_tokens", 0), um.get("total_tokens", 0))
        elif isinstance(getattr(resp, "response_metadata", None), dict):
            tu = resp.response_metadata.get("token_usage") or resp.response_metadata.get("usage")
            if tu:
                usage = (tu.get("prompt_tokens", 0), tu.get("completion_tokens", 0), tu.get("total_tokens", 0))
        if usage:
            self.token_usage["prompt_tokens"] += usage[0]
            self.token_usage["completion_tokens"] += usage[1]
            self.token_usage["total_tokens"] += usage[2]
            self.token_usage["call_count"] += 1
            bucket = self.token_usage_by_tag.setdefault(
                tag, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "call_count": 0}
            )
            bucket["prompt_tokens"] += usage[0]
            bucket["completion_tokens"] += usage[1]
            bucket["total_tokens"] += usage[2]
            bucket["call_count"] += 1
        return resp
