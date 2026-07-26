"""Chuẩn hoá thứ tự căn cứ và định dạng câu trả lời pháp lý.

Module này cố ý không phụ thuộc Chainlit/LangGraph để có thể dùng
chung trong pipeline, CLI đánh giá và unit test.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable


INCOMPLETE_WARNING = (
    "Câu trả lời dưới đây có thể chưa đầy đủ do chưa truy xuất được "
    "toàn bộ căn cứ pháp luật liên quan."
)

_CITATION_RE = re.compile(
    r"\[\s*(?:Điểm\s+(?P<point>[^,\]]+)\s*,\s*)?"
    r"(?:Khoản\s+(?P<clause>[^,\]]+)\s*,\s*)?"
    r"Điều\s+(?P<article>[^,\]]+)\s*,\s*(?P<document>[^\]]+)\]",
    re.IGNORECASE,
)
_SOURCE_HEADING_RE = re.compile(r"(?im)^#{1,3}\s*Căn cứ pháp lý\s*$")
_TECHNICAL_LINE_RE = re.compile(
    r"^\s*(?:\[retrieve\]|is_complete\s*[:=]|graph\s*[:=]|embedding\s*[:=]|"
    r"rerank\s*[:=]|score\s*[:=]|token(?:s|_usage)?\s*[:=])",
    re.IGNORECASE,
)


def _ascii_fold(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value or "")
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn").lower().replace("đ", "d")


def document_priority(document: str, content: str = "") -> int:
    """Thứ tự ưu tiên: Luật, Bộ luật, NĐ chuyên ngành, NĐ xử phạt, Thông tư."""

    title = _ascii_fold(document).strip()
    body = _ascii_fold(content[:4000])
    if title.startswith("luat "):
        return 1
    if title.startswith("bo luat "):
        return 2
    if title.startswith("nghi dinh "):
        penalty_markers = (
            "xu phat",
            "vi pham hanh chinh",
            "phat tien",
            "hinh thuc xu phat",
            "muc phat",
        )
        return 4 if any(marker in title or marker in body for marker in penalty_markers) else 3
    if title.startswith("thong tu "):
        return 5
    return 6


def context_document(text: str, fallback: str = "") -> str:
    match = _CITATION_RE.search(text or "")
    if match:
        return match.group("document").strip().removesuffix(".docx")
    return (fallback or "").strip().removesuffix(".docx")


def sort_context_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sắp căn cứ theo hiệu lực/loại văn bản, sau đó mới theo score."""

    indexed = list(enumerate(records))

    def key(item: tuple[int, dict[str, Any]]) -> tuple[int, float, int]:
        index, record = item
        text = str(record.get("text") or "")
        document = context_document(text, str(record.get("source") or ""))
        score = float(record.get("score") or 0.0)
        return document_priority(document, text), -score, index

    return [record for _, record in sorted(indexed, key=key)]


def extract_legal_sources(context_texts: Iterable[str], limit: int = 20) -> list[dict[str, Any]]:
    """Trích Tên văn bản/Điều/Khoản/Điểm từ prefix chunk đã ingest."""

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for text in context_texts:
        for match in _CITATION_RE.finditer(text or ""):
            document = match.group("document").strip().removesuffix(".docx")
            article = match.group("article").strip()
            key = (document, article)
            bucket = grouped.setdefault(
                key,
                {
                    "document": document,
                    "article": article,
                    "clauses": set(),
                    "points": set(),
                    "priority": document_priority(document, text),
                },
            )
            if match.group("clause"):
                bucket["clauses"].add(match.group("clause").strip())
            if match.group("point"):
                bucket["points"].add(match.group("point").strip())

    def natural(value: str) -> tuple[int, str]:
        number = re.search(r"\d+", value)
        return (int(number.group()) if number else 10**9, value)

    sources = sorted(
        grouped.values(),
        key=lambda source: (
            source["priority"],
            _ascii_fold(source["document"]),
            natural(source["article"]),
        ),
    )
    for source in sources:
        source["clauses"] = sorted(source["clauses"], key=natural)
        source["points"] = sorted(source["points"], key=natural)
    return sources[:limit]


def format_legal_sources(sources: Iterable[dict[str, Any]]) -> str:
    lines = ["## Căn cứ pháp lý", ""]
    source_list = list(sources)
    if not source_list:
        lines.append("- Chưa xác định được căn cứ pháp lý đủ tin cậy từ dữ liệu đã truy xuất.")
        return "\n".join(lines)

    for source in source_list:
        detail = [f"Điều {source['article']}"]
        if source.get("clauses"):
            detail.append("Khoản " + ", ".join(source["clauses"]))
        if source.get("points"):
            detail.append("Điểm " + ", ".join(source["points"]))
        lines.append(f"- **{source['document']}** — " + "; ".join(detail) + ".")
    return "\n".join(lines)


def _sources_used_in_answer(answer: str, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chỉ hiển thị văn bản mà phần lập luận thực sự viện dẫn."""

    folded_answer = _ascii_fold(answer)
    used = []
    for source in sources:
        document = source["document"]
        # Phần tên trước ngoặc thường là tên chính thức được model
        # nhắc lại; bỏ chi tiết "sửa đổi, bổ sung..." để match bền vững.
        base_document = document.split("(", 1)[0].strip()
        article_pattern = re.compile(rf"\bdieu\s+{re.escape(_ascii_fold(source['article']))}\b")
        if _ascii_fold(base_document) in folded_answer and article_pattern.search(folded_answer):
            cited = dict(source)
            cited["clauses"] = [
                clause
                for clause in source.get("clauses", [])
                if re.search(rf"\bkhoan\s+{re.escape(_ascii_fold(clause))}\b", folded_answer)
            ]
            cited["points"] = [
                point
                for point in source.get("points", [])
                if re.search(rf"\bdiem\s+{re.escape(_ascii_fold(point))}\b", folded_answer)
            ]
            used.append(cited)
    if used:
        return used

    # Model đôi khi chỉ viện dẫn "Điều X" sau khi tên văn bản đã
    # nằm trong chính câu hỏi. Khi không match được tên, chỉ giữ căn cứ
    # ưu tiên cao nhất thay vì dump toàn bộ candidate retrieval.
    return sources[:1]


def _strip_technical_lines(answer: str) -> str:
    return "\n".join(line for line in (answer or "").splitlines() if not _TECHNICAL_LINE_RE.match(line))


def finalize_legal_answer(answer: str, *, is_complete: bool, context_texts: Iterable[str]) -> str:
    """Bảo đảm warning, không lộ log nội bộ và có mục căn cứ xác định."""

    cleaned = _strip_technical_lines(answer).strip()
    existing_source_heading = _SOURCE_HEADING_RE.search(cleaned)
    if existing_source_heading:
        cleaned = cleaned[: existing_source_heading.start()].rstrip()

    sections: list[str] = []
    if not is_complete:
        sections.append(f"> **Lưu ý:** {INCOMPLETE_WARNING}")
    if cleaned:
        sections.append(cleaned)
    sources = extract_legal_sources(context_texts)
    sections.append(format_legal_sources(_sources_used_in_answer(cleaned, sources)))
    return "\n\n".join(sections).strip()
