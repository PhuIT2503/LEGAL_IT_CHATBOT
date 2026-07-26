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

import logging
import time
from typing import Callable, Dict, Iterable, Optional, Set


logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, llm):
        self.llm = llm
        self.token_usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "call_count": 0}
        self.token_usage_by_tag: Dict[str, Dict[str, int]] = {}
        self._stream_callback: Optional[Callable[[str], None]] = None
        self._stream_tags: Set[str] = set()

    def set_stream_callback(self, callback: Optional[Callable[[str], None]], *, tags: Optional[Iterable[str]] = None) -> None:
        """Bật streaming cho đúng lệnh sinh câu trả lời cuối của request."""

        self._stream_callback = callback
        self._stream_tags = set(tags or ()) if callback else set()

    @staticmethod
    def _chunk_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        return str(content or "")

    def reset_usage(self) -> None:
        """Reset đếm token — gọi ở đầu mỗi ChatbotWorkflow.run(), mỗi câu hỏi độc lập, không cộng dồn qua các câu."""
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "call_count": 0}
        self.token_usage_by_tag = {}

    def invoke(self, prompt: str, tag: str = "other", **model_kwargs):
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
        started_at = time.monotonic()
        logger.debug(
            "[llm] start tag=%s prompt_chars=%s",
            tag,
            len(prompt),
        )
        try:
            if self._stream_callback and tag in self._stream_tags:
                combined = None
                for chunk in self.llm.stream(prompt, **model_kwargs):
                    token = self._chunk_text(getattr(chunk, "content", ""))
                    if token:
                        self._stream_callback(token)
                    combined = chunk if combined is None else combined + chunk
                # Một provider không trả chunk là bất thường; fallback để
                # API cũ vẫn hoạt động thay vì trả message rỗng.
                resp = (
                    combined
                    if combined is not None
                    else self.llm.invoke(prompt, **model_kwargs)
                )
            else:
                resp = self.llm.invoke(prompt, **model_kwargs)
        except Exception:
            logger.warning(
                "[llm] failed tag=%s elapsed=%.2fs",
                tag,
                time.monotonic() - started_at,
            )
            raise
        logger.info(
            "[llm] complete tag=%s elapsed=%.2fs",
            tag,
            time.monotonic() - started_at,
        )
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
