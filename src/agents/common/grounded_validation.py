"""Grounding contract giữa retrieval và generation.

Model chỉ được tham chiếu các ``SOURCE_ID`` do module này cấp.  Trích luật,
trích dẫn hiển thị và danh mục căn cứ cuối bài đều được dựng lại từ dữ liệu
retrieval, thay vì tin vào tên văn bản/Điều/Khoản do model tự gõ.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Iterable, Sequence
import unicodedata

from src.agents.common.legal_response import document_priority
from src.agents.common.retrieval_ranking import lexical_relevance


INSUFFICIENT_GROUNDS = (
    "Chưa đủ căn cứ pháp lý để kết luận do chưa truy xuất được điều luật liên quan."
)
INCOMPLETE_GROUNDS_WARNING = (
    "Câu trả lời có thể chưa đầy đủ vì chưa truy xuất được toàn bộ căn cứ pháp luật."
)

logger = logging.getLogger(__name__)

_PREFIX_RE = re.compile(
    r"\[\s*(?:Điểm\s+(?P<point>[^,\]]+)\s*,\s*)?"
    r"(?:Khoản\s+(?P<clause>[^,\]]+)\s*,\s*)?"
    r"Điều\s+(?P<article>[^,\]]+)\s*,\s*(?P<document>[^\]]+)\]",
    re.IGNORECASE,
)
_MARKER_RE = re.compile(r"\[\[(?P<kind>CITE|QUOTE):(?P<ids>S\d+(?:\s*,\s*S\d+)*)\]\]")
_SOURCE_HEADING_RE = re.compile(r"(?im)^#{1,3}\s*Căn cứ pháp lý\s*$")
_TECHNICAL_LINE_RE = re.compile(
    r"^\s*(?:\[retrieve\]|is_complete\s*[:=]|graph\s*[:=]|embedding\s*[:=]|"
    r"rerank\s*[:=]|score\s*[:=]|token(?:s|_usage)?\s*[:=])",
    re.IGNORECASE,
)
_LEGAL_CONCLUSION_RE = re.compile(
    r"\b(?:vi phạm|thuộc trường hợp|bị nghiêm cấm|không được|phải\b|có quyền|"
    r"chịu trách nhiệm|hậu quả pháp lý|xử phạt|phạt tiền|áp dụng|kết luận)\b",
    re.IGNORECASE,
)
_SANCTION_RE = re.compile(
    r"\b(?:phạt|xử phạt|triệu|tỷ|đồng|đình chỉ|tước quyền|khắc phục hậu quả|"
    r"cá nhân|tổ chức)\b",
    re.IGNORECASE,
)
_NO_SANCTION_RE = re.compile(
    r"(?:chưa đủ căn cứ|không đủ căn cứ|không có căn cứ|không đề cập|chưa thể xác định)",
    re.IGNORECASE,
)
_NUMBERED_DOCUMENT_RE = re.compile(
    r"\b(?:Nghị định|Thông tư)\s+(?:số\s+)?\d{1,4}[/.-]\d{4}[^\s,.;)]*",
    re.IGNORECASE,
)
_DATED_LAW_RE = re.compile(
    r"\b(?:Bộ luật|Luật)\s+[^\n,.();:]{2,80}?\s+\d{4}\b",
    re.IGNORECASE,
)
_LEGAL_EFFECT_TERMS = (
    "xử lý hình sự",
    "xử lý hành chính",
    "bị xử lý",
    "truy cứu",
    "phạt tiền",
    "bồi thường",
    "đình chỉ",
    "tước quyền",
    "buộc ",
    "phải dừng",
)
_MISSING_EFFECT_RE = re.compile(
    r"(?:source|đoạn|nguồn|căn cứ).{0,50}(?:không nêu|chưa nêu|không quy định|"
    r"chưa đủ căn cứ|chưa thể xác định)",
    re.IGNORECASE,
)
_COPYING_ONLY_RE = re.compile(
    r"(?:đối chiếu với tình tiết|không bổ sung|chỉ xác nhận(?: nội dung)?|"
    r"nội dung được giới hạn|chỉ có thể xác nhận|không mở rộng)",
    re.IGNORECASE,
)
_GENERIC_ANALYSIS_RE = re.compile(
    r"(?:tình tiết(?: nêu trên)? thuộc đúng nhóm (?:hoạt động|đối tượng)|"
    r"(?:đây|nội dung(?: trong)? đoạn trích|đoạn luật).{0,35}là căn cứ trực tiếp|"
    r"có liên hệ trực tiếp|"
    r"\bcó dấu hiệu\b|"
    r"tình tiết và phạm vi điều chỉnh.{0,35}liên hệ)",
    re.IGNORECASE,
)
_CONFIDENCE_RE = re.compile(
    r"(?:🟢\s*Đủ căn cứ|🟡\s*Chưa đủ điều kiện kết luận|🔴\s*Chưa đủ căn cứ)"
)
_DIRECT_ANSWER_RE = re.compile(
    r"^(?:có\b|không\b|chưa đủ căn cứ\b|không tìm thấy căn cứ\b|"
    r"có dấu hiệu vi phạm\b|phải\b|không phải\b)",
    re.IGNORECASE,
)
_NO_SOURCE_ANSWER_RE = re.compile(
    r"(?:chưa đủ căn cứ|không đủ căn cứ|không tìm thấy căn cứ|"
    r"không có căn cứ|chưa truy xuất được)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GroundedSource:
    source_id: str
    document: str
    article: str
    clause: str | None
    point: str | None
    text: str
    body: str


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    issues: tuple[str, ...]
    used_source_ids: tuple[str, ...]


def _clean_document(value: str) -> str:
    return value.strip().removesuffix(".docx")


def _normalise(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value or "").casefold().split())


def extract_user_questions(query: str, limit: int = 6) -> list[str]:
    """Giữ nguyên các câu hỏi người dùng để synthesis trả lời đúng ý định."""

    query = unicodedata.normalize("NFC", query or "")
    text = "\n".join(line.strip() for line in query.splitlines() if line.strip())
    numbered = [
        match.group(1).strip()
        for match in re.finditer(
            r"(?m)^\s*(?:\d+[.)]|[-*])\s+(.+?)\s*$",
            text,
        )
        if re.search(
            r"(?:\?|có\b|không\b|nào\b|gì\b|bao nhiêu|như thế nào|phải\b)",
            match.group(1),
            re.IGNORECASE,
        )
    ]
    if numbered:
        return numbered[:limit]

    # Người dùng thường gõ "1. ...? 2. ...?" trên cùng một dòng trong ô chat.
    # Tách theo marker số trước khi fallback theo dấu hỏi để phần mô tả tình
    # huống không bị dính vào câu hỏi thứ nhất.
    inline_numbered = [
        match.group(1).strip()
        for match in re.finditer(
            r"(?s)(?:^|\s)\d+[.)]\s+(.+?)(?=(?:\s+\d+[.)]\s+)|$)",
            text,
        )
        if re.search(
            r"(?:\?|có\b|không\b|nào\b|gì\b|bao nhiêu|như thế nào|phải\b)",
            match.group(1),
            re.IGNORECASE,
        )
    ]
    if inline_numbered:
        return inline_numbered[:limit]

    line_questions = [
        " ".join(line.split()).strip()
        for line in text.splitlines()
        if "?" in line and line.strip()
    ]
    if line_questions:
        return line_questions[:limit]

    questions = [
        " ".join(match.group(0).split()).strip()
        for match in re.finditer(r"[^?]+\?", text)
        if match.group(0).strip()
    ]
    if questions:
        return questions[-limit:]
    return [text] if text else []


def _document_key(value: str) -> str:
    """So tên văn bản bền vững qua '/', '-', '_' và khác biệt dấu tiếng Việt."""

    decomposed = unicodedata.normalize("NFD", value or "")
    ascii_value = "".join(
        character
        for character in decomposed
        if unicodedata.category(character) != "Mn"
    ).casefold().replace("đ", "d")
    return "".join(character for character in ascii_value if character.isalnum())


def _clean_body(value: str) -> str:
    body = " ".join((value or "").split()).strip(" -–—:;\n")
    return body


def build_grounded_sources(
    context_texts: Iterable[str],
    limit: int = 24,
    *,
    query: str = "",
) -> list[GroundedSource]:
    """Tách mỗi prefix retrieved thành một nguồn nhỏ, có ID ổn định.

    Parent chunk có thể chứa nhiều Khoản/Điểm. Tách theo từng prefix giúp
    validator biết chính xác model đang dùng phần nào thay vì coi cả Điều là
    một khối căn cứ mơ hồ.
    """

    raw: list[dict[str, str | None | int]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    order = 0
    for context_index, context in enumerate(context_texts):
        text = str(context or "")
        matches = list(_PREFIX_RE.finditer(text))
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            document = _clean_document(match.group("document"))
            article = match.group("article").strip()
            clause = match.group("clause").strip() if match.group("clause") else None
            point = match.group("point").strip() if match.group("point") else None
            body = _clean_body(text[match.end() : end])
            exact = (match.group(0) + (" " + body if body else "")).strip()
            key = (document, article, clause or "", point or "", body)
            if key in seen:
                continue
            seen.add(key)
            raw.append(
                {
                    "document": document,
                    "article": article,
                    "clause": clause,
                    "point": point,
                    "body": body,
                    "text": exact,
                    "priority": document_priority(document, exact),
                    "context_index": context_index,
                    "order": order,
                }
            )
            order += 1

    if query:
        def ranking_key(item: dict[str, str | None | int]) -> tuple[int, float, int]:
            return (
                int(item["priority"]),
                -lexical_relevance(query, f"{item['document']}\n{item['text']}"),
                int(item["order"]),
            )

        ranked = sorted(raw, key=ranking_key)

        # Một parent context (thường là toàn bộ một Điều) có thể chứa hàng
        # chục Khoản/Điểm. Nếu chỉ sort rồi cắt ``limit``, một Điều luật dài
        # sẽ chiếm hết quota và làm rơi Điều xử phạt đã retrieve. Giữ trước
        # đoạn phù hợp nhất của từng context, sau đó mới lấp phần quota còn lại.
        representatives: dict[int, dict[str, str | None | int]] = {}
        for item in ranked:
            representatives.setdefault(int(item["context_index"]), item)
        representative_ids = {id(item) for item in representatives.values()}
        raw = sorted(representatives.values(), key=ranking_key) + [
            item for item in ranked if id(item) not in representative_ids
        ]
    else:
        raw.sort(key=lambda item: (int(item["priority"]), int(item["order"])))
    return [
        GroundedSource(
            source_id=f"S{index}",
            document=str(item["document"]),
            article=str(item["article"]),
            clause=str(item["clause"]) if item["clause"] else None,
            point=str(item["point"]) if item["point"] else None,
            text=str(item["text"]),
            body=str(item["body"]),
        )
        for index, item in enumerate(raw[:limit], start=1)
    ]


def citation_label(source: GroundedSource, *, markdown: bool = False) -> str:
    document = f"**{source.document}**" if markdown else source.document
    parts = [document, f"Điều {source.article}"]
    if source.clause:
        parts.append(f"Khoản {source.clause}")
    if source.point:
        parts.append(f"Điểm {source.point}")
    return ", ".join(parts)


def format_grounded_context(sources: Sequence[GroundedSource]) -> str:
    if not sources:
        return "KHÔNG CÓ SOURCE PHÁP LUẬT ĐÃ RETRIEVE."
    return "\n\n".join(
        f'<SOURCE id="{source.source_id}" citation="{citation_label(source)}">\n'
        f"{source.text}\n</SOURCE>"
        for source in sources
    )


def _marker_ids(marker_text: str) -> list[str]:
    match = _MARKER_RE.fullmatch(marker_text.strip())
    return [item.strip() for item in match.group("ids").split(",")] if match else []


def _section(answer: str, heading: str, next_headings: Sequence[str]) -> str:
    start_match = re.search(rf"(?im)^#{{1,2}}\s+{re.escape(heading)}\s*$", answer)
    if not start_match:
        return ""
    end = len(answer)
    for next_heading in next_headings:
        match = re.search(
            rf"(?im)^#{{1,2}}\s+{re.escape(next_heading)}\s*$",
            answer[start_match.end() :],
        )
        if match:
            end = min(end, start_match.end() + match.start())
    return answer[start_match.end() : end].strip()


def _paragraphs(value: str) -> list[str]:
    return [block.strip() for block in re.split(r"\n\s*\n", value or "") if block.strip()]


def _legal_assertion_text(paragraph: str) -> str:
    """Loại dòng mô tả dữ kiện người dùng khỏi phép kiểm hallucination.

    ``Hành vi`` có thể chứa chính câu hỏi như "có bị xử lý không". Đây là
    dữ kiện/yêu cầu do người dùng đưa vào, không phải hậu quả pháp lý do model
    khẳng định, nên không được làm validator loại một câu trả lời có căn cứ.
    Các dòng Phân tích, Đánh giá và phần trả lời tổng hợp vẫn được kiểm tra đầy đủ.
    """

    return "\n".join(
        line
        for line in (paragraph or "").splitlines()
        if not re.search(r"^\s*###\s+\d+[.)]", line)
        and not re.search(
            r"\*\*(?:Hành vi(?: phù hợp)?|Tình tiết):\*\*",
            line,
            re.IGNORECASE,
        )
    )


def _explicit_coordinates(value: str, label: str) -> set[str]:
    return {
        _normalise(match.group(1))
        for match in re.finditer(rf"\b{label}\s+(\d+[a-zA-ZđĐ]?)\b", value, re.IGNORECASE)
    }


def _cited_sources_in_text(
    value: str, source_map: dict[str, GroundedSource]
) -> list[GroundedSource]:
    selected: list[GroundedSource] = []
    for marker in _MARKER_RE.finditer(value):
        if marker.group("kind") != "CITE":
            continue
        for source_id in (item.strip() for item in marker.group("ids").split(",")):
            source = source_map.get(source_id)
            if source and source not in selected:
                selected.append(source)
    return selected


def validate_grounded_draft(
    draft: str,
    sources: Sequence[GroundedSource],
    *,
    is_complete: bool = True,
    query: str = "",
) -> ValidationResult:
    """Kiểm tra bản nháp trước khi bất kỳ nội dung pháp lý nào tới UI."""

    answer = (draft or "").strip()
    if answer == INSUFFICIENT_GROUNDS:
        if not sources:
            return ValidationResult(True, (), ())
        return ValidationResult(
            False,
            ("Đã có SOURCE liên quan nên không được từ chối toàn bộ câu trả lời.",),
            (),
        )

    issues: list[str] = []
    required_headings = (
        "Tóm tắt tình huống",
        "Các vấn đề pháp lý",
        "Phân tích",
        "Chế tài",
        "Trả lời câu hỏi của người dùng",
    )
    for heading in required_headings:
        if not re.search(rf"(?im)^#{{1,2}}\s+{re.escape(heading)}\s*$", answer):
            issues.append(f"Thiếu heading bắt buộc: {heading}.")

    if _SOURCE_HEADING_RE.search(answer):
        issues.append("Model tự tạo mục Căn cứ pháp lý; mục này phải do hệ thống dựng.")

    source_map = {source.source_id: source for source in sources}
    marker_matches = list(_MARKER_RE.finditer(answer))
    used_ids: list[str] = []
    for marker in marker_matches:
        for source_id in (item.strip() for item in marker.group("ids").split(",")):
            if source_id not in source_map:
                issues.append(f"Marker tham chiếu SOURCE_ID không tồn tại: {source_id}.")
            elif source_id not in used_ids:
                used_ids.append(source_id)

    stray_markers = re.findall(r"\[\[(?:CITE|QUOTE):[^\]]+\]\]", answer)
    if len(stray_markers) != len(marker_matches):
        issues.append("Có marker trích dẫn sai cú pháp.")
    if not sources:
        issues.append("Không có nguồn retrieval để tạo kết luận pháp lý.")
    if not any(match.group("kind") == "CITE" for match in marker_matches):
        issues.append("Không có citation theo kết luận.")

    synthesis_heading = "Trả lời câu hỏi của người dùng"
    analysis = _section(answer, "Phân tích", ("Chế tài", synthesis_heading))
    synthesis = _section(answer, synthesis_heading, ())
    sanctions = _section(answer, "Chế tài", (synthesis_heading,))

    analysis_items = [item for item in re.split(r"(?m)^###\s+", analysis) if item.strip()]
    if not analysis_items:
        issues.append("Phần Phân tích chưa chia thành từng vấn đề cấp 3.")
    reasoning_texts: list[str] = []
    for item in analysis_items:
        if not re.search(r"\[\[QUOTE:S\d+(?:\s*,\s*S\d+)*\]\]", item):
            issues.append("Mỗi hành vi trong phần Phân tích phải có trích đoạn retrieved.")
        if not re.search(r"\[\[CITE:S\d+(?:\s*,\s*S\d+)*\]\]", item):
            issues.append("Mỗi hành vi trong phần Phân tích phải có căn cứ inline.")
        for field in (
            "Nội dung điều luật",
            "Hành vi phù hợp",
            "Phân tích",
            "Điều kiện còn thiếu",
            "Căn cứ pháp lý",
            "Đánh giá",
        ):
            if not re.search(rf"\*\*{re.escape(field)}:\*\*", item, re.IGNORECASE):
                issues.append(f"Mỗi vấn đề thiếu trường bắt buộc: {field}.")
        summary_match = re.search(
            r"(?im)^\s*-\s*\*\*Nội dung điều luật:\*\*\s*(.+)$",
            item,
        )
        behavior_match = re.search(
            r"(?im)^\s*-\s*\*\*Hành vi phù hợp:\*\*\s*(.+)$",
            item,
        )
        reasoning_match = re.search(
            r"(?im)^\s*-\s*\*\*Phân tích:\*\*\s*(.+)$",
            item,
        )
        missing_match = re.search(
            r"(?im)^\s*-\s*\*\*Điều kiện còn thiếu:\*\*\s*(.+)$",
            item,
        )
        if not summary_match or len(summary_match.group(1).strip()) < 35:
            issues.append("Chưa tóm tắt nội dung riêng của điều luật bằng ngôn ngữ tự nhiên.")
        if not behavior_match or len(behavior_match.group(1).strip()) < 20:
            issues.append("Chưa chỉ rõ hành vi cụ thể trong tình huống phù hợp với điều luật.")
        if not reasoning_match or len(reasoning_match.group(1).strip()) < 70:
            issues.append("Phần Phân tích chưa giải thích cụ thể vì sao điều luật được áp dụng.")
        elif reasoning_match:
            reasoning_texts.append(_normalise(reasoning_match.group(1)))
        if not missing_match or len(missing_match.group(1).strip()) < 25:
            issues.append("Chưa nêu rõ điều kiện còn thiếu hoặc xác nhận không còn điều kiện thiếu.")
        if query and lexical_relevance(query, item) < 0.55:
            issues.append("Phần Phân tích chưa áp dụng vào tình tiết cụ thể của người dùng.")
        for evaluation_line in re.findall(r"(?im)^.*\*\*Đánh giá:\*\*.*$", item):
            if "[[CITE:" not in evaluation_line:
                issues.append("Dòng Đánh giá thiếu citation trực tiếp.")
            if not _CONFIDENCE_RE.search(evaluation_line):
                issues.append("Dòng Đánh giá thiếu mức 🟢/🟡/🔴 hợp lệ.")

    if len(reasoning_texts) != len(set(reasoning_texts)):
        issues.append("Nhiều điều luật đang dùng lại nguyên văn cùng một mẫu phân tích.")

    if _COPYING_ONLY_RE.search(analysis):
        issues.append("Câu trả lời đang lặp SOURCE/câu máy móc thay vì áp dụng vào tình huống.")
    if _GENERIC_ANALYSIS_RE.search(analysis):
        issues.append("Phân tích dùng câu chung chung thay vì đối chiếu nội dung riêng của điều luật.")
    if not is_complete and "🟢 Đủ căn cứ" in answer:
        issues.append("Retrieval chưa đầy đủ nên không được gắn mức 🟢 Đủ căn cứ.")

    expected_questions = extract_user_questions(query) if query else []
    synthesis_items = [
        item for item in re.split(r"(?m)^###\s+", synthesis) if item.strip()
    ]
    if not synthesis_items:
        issues.append("Phần trả lời cuối chưa chia theo từng câu hỏi của người dùng.")
    if expected_questions and len(synthesis_items) != len(expected_questions):
        issues.append(
            "Số câu trả lời tổng hợp không khớp số câu hỏi người dùng: "
            f"cần {len(expected_questions)}, nhận {len(synthesis_items)}."
        )
    for index, item in enumerate(synthesis_items, start=1):
        title = item.splitlines()[0].strip() if item.splitlines() else ""
        title_without_number = re.sub(r"^\d+[.)]\s*", "", title)
        if re.match(
            r"(?:điều\s+\d+|luật\b|bộ luật\b|nghị định\b)",
            title_without_number,
            re.IGNORECASE,
        ):
            issues.append("Phần trả lời cuối đang kết luận theo điều luật thay vì câu hỏi.")
        for field in ("Trả lời trực tiếp", "Vì sao", "Căn cứ", "Còn thiếu"):
            if not re.search(rf"\*\*{re.escape(field)}:\*\*", item, re.IGNORECASE):
                issues.append(f"Câu trả lời số {index} thiếu trường: {field}.")
        direct_match = re.search(
            r"(?im)^\s*-\s*\*\*Trả lời trực tiếp:\*\*\s*(.+)$", item
        )
        why_match = re.search(r"(?im)^\s*-\s*\*\*Vì sao:\*\*\s*(.+)$", item)
        grounds_match = re.search(r"(?im)^\s*-\s*\*\*Căn cứ:\*\*\s*(.+)$", item)
        missing_match = re.search(r"(?im)^\s*-\s*\*\*Còn thiếu:\*\*\s*(.+)$", item)
        direct = direct_match.group(1).strip() if direct_match else ""
        if not direct or not _DIRECT_ANSWER_RE.search(direct):
            issues.append(
                f"Câu trả lời số {index} chưa mở đầu bằng kết luận Có/Không/Chưa đủ căn cứ."
            )
        no_source_answer = bool(_NO_SOURCE_ANSWER_RE.search(item))
        if (
            no_source_answer
            and re.match(r"^không\b", direct, re.IGNORECASE)
            and not re.match(r"^không tìm thấy căn cứ\b", direct, re.IGNORECASE)
        ):
            issues.append(
                f"Câu trả lời số {index} suy ra 'Không' chỉ từ việc không tìm thấy căn cứ."
            )
        if not no_source_answer and "[[CITE:" not in item:
            issues.append(f"Câu trả lời số {index} có kết luận nhưng thiếu citation.")
        if not why_match or len(why_match.group(1).strip()) < 40:
            issues.append(f"Câu trả lời số {index} chưa giải thích ngắn gọn vì sao.")
        elif "[[CITE:" not in why_match.group(1) and not _NO_SOURCE_ANSWER_RE.search(why_match.group(1)):
            issues.append(f"Câu trả lời số {index} có lý do nhưng thiếu citation trên cùng dòng.")
        if grounds_match and "[[CITE:" not in grounds_match.group(1) and not _NO_SOURCE_ANSWER_RE.search(grounds_match.group(1)):
            issues.append(f"Câu trả lời số {index} chưa nêu căn cứ đã dùng hoặc việc không có căn cứ.")
        if not missing_match or len(missing_match.group(1).strip()) < 15:
            issues.append(f"Câu trả lời số {index} chưa nêu phần căn cứ/dữ kiện còn thiếu.")

    for paragraph in _paragraphs(analysis + "\n\n" + synthesis):
        assertion_text = _legal_assertion_text(paragraph)
        if (
            _LEGAL_CONCLUSION_RE.search(assertion_text)
            and "[[CITE:" not in assertion_text
            and not _NO_SOURCE_ANSWER_RE.search(assertion_text)
        ):
            issues.append("Có kết luận pháp lý chưa gắn citation trong cùng đoạn.")
        cited = _cited_sources_in_text(assertion_text, source_map)
        if cited:
            cited_corpus = " ".join(_normalise(source.text) for source in cited)
            inline_allowed = {
                "điều": {_normalise(source.article) for source in cited},
                "khoản": {_normalise(source.clause) for source in cited if source.clause},
                "điểm": {_normalise(source.point) for source in cited if source.point},
            }
            for label in ("Điều", "Khoản", "Điểm"):
                for value in _explicit_coordinates(assertion_text, label):
                    coordinate = _normalise(f"{label} {value}")
                    if (
                        value not in inline_allowed[label.casefold()]
                        and coordinate not in cited_corpus
                    ):
                        issues.append(
                            f"{label} {value} không khớp SOURCE_ID được cite trong cùng đoạn."
                        )
            for known_source in sources:
                if (
                    _normalise(known_source.document) in _normalise(assertion_text)
                    and known_source.document not in {source.document for source in cited}
                ):
                    issues.append(
                        f"Văn bản {known_source.document} không khớp SOURCE_ID được cite trong cùng đoạn."
                    )
            for line in assertion_text.splitlines():
                if _SANCTION_RE.search(line) and not _NO_SANCTION_RE.search(line):
                    for number in re.findall(r"\b\d+(?:[.,]\d+)?\b", line):
                        if _normalise(number) not in cited_corpus:
                            issues.append(
                                f"Con số chế tài {number} không tồn tại trong SOURCE_ID được cite."
                            )
            for effect in _LEGAL_EFFECT_TERMS:
                if (
                    effect in _normalise(assertion_text)
                    and effect not in cited_corpus
                    and not _MISSING_EFFECT_RE.search(assertion_text)
                ):
                    issues.append(
                        f"Hậu quả/lập luận '{effect.strip()}' không xuất hiện trong SOURCE_ID được cite."
                    )

        for line in paragraph.splitlines():
            if (
                re.search(r"\*\*Hậu quả pháp lý:\*\*", line, re.IGNORECASE)
                and not _MISSING_EFFECT_RE.search(line)
                and "[[CITE:" not in line
            ):
                issues.append("Kết luận về hậu quả pháp lý thiếu citation trên cùng dòng.")

    has_applied_sanction = (
        sanctions
        and _SANCTION_RE.search(sanctions)
        and ("[[CITE:" in sanctions or not _NO_SANCTION_RE.search(sanctions))
    )
    if has_applied_sanction:
        if "[[QUOTE:" not in sanctions or "[[CITE:" not in sanctions:
            issues.append("Phần Chế tài có nội dung áp dụng nhưng thiếu trích luật/citation.")
        required_terms = ("hành vi", "cá nhân", "tổ chức", "điều kiện")
        for term in required_terms:
            if term not in sanctions.casefold():
                issues.append(f"Phần Chế tài chưa giải thích: {term}.")
        cited_sanction_sources = _cited_sources_in_text(sanctions, source_map)
        cited_corpus = " ".join(_normalise(source.text) for source in cited_sanction_sources)
        for number in re.findall(r"\b\d+(?:[.,]\d+)?\b", sanctions):
            if _normalise(number) not in cited_corpus:
                issues.append(
                    f"Con số chế tài {number} không tồn tại trong SOURCE_ID được cite."
                )
        for line in sanctions.splitlines():
            if (
                (_SANCTION_RE.search(line) or re.search(r"\b\d+(?:[.,]\d+)?\b", line))
                and not _NO_SANCTION_RE.search(line)
                and not line.lstrip().startswith("[[QUOTE:")
                and "[[CITE:" not in line
            ):
                issues.append("Thông tin chế tài thiếu citation trên cùng dòng.")

    # Chỉ kiểm tra phần lập luận pháp lý. Tóm tắt có thể nhắc lại nguyên văn
    # Điều/văn bản do người dùng hỏi nhưng đó chưa phải citation của chatbot.
    legal_body = "\n\n".join((analysis, sanctions, synthesis))

    # Điều/Khoản/Điểm được model tự viết cũng phải tồn tại trong retrieval.
    allowed = {
        "điều": {_normalise(source.article) for source in sources},
        "khoản": {_normalise(source.clause) for source in sources if source.clause},
        "điểm": {_normalise(source.point) for source in sources if source.point},
    }
    retrieved_corpus = " ".join(_normalise(source.text) for source in sources)
    for label in ("Điều", "Khoản", "Điểm"):
        for value in _explicit_coordinates(legal_body, label):
            # Marker/citation hướng dẫn không chứa chữ Điều; mọi match ở đây là
            # model tự gõ và phải khớp dữ liệu retrieved.
            coordinate = _normalise(f"{label} {value}")
            if (
                value
                and value not in allowed[label.casefold()]
                and coordinate not in retrieved_corpus
            ):
                issues.append(f"{label} {value} không tồn tại trong nguồn retrieval.")

    known_documents = [_document_key(source.document) for source in sources]
    for pattern in (_NUMBERED_DOCUMENT_RE, _DATED_LAW_RE):
        for match in pattern.finditer(legal_body):
            candidate = _document_key(match.group(0))
            if not any(candidate in document or document in candidate for document in known_documents):
                issues.append(f"Văn bản không tồn tại trong nguồn retrieval: {match.group(0)}.")

    return ValidationResult(not issues, tuple(dict.fromkeys(issues)), tuple(used_ids))


def salvage_grounded_draft(
    *, query: str, draft: str, sources: Sequence[GroundedSource]
) -> str:
    """Giữ từng hành vi hợp lệ thay vì bỏ cả câu trả lời vì một phần lỗi.

    Mỗi block được validate độc lập. Phần chế tài không đạt contract bị hạ
    riêng về thiếu căn cứ; chỉ khi không còn block nào hợp lệ mới fail closed.
    """

    if not sources:
        return INSUFFICIENT_GROUNDS
    source_map = {source.source_id: source for source in sources}
    selected_ids: list[str] = []
    for marker in _MARKER_RE.finditer(draft):
        if marker.group("kind") != "QUOTE":
            continue
        for source_id in (item.strip() for item in marker.group("ids").split(",")):
            if source_id in source_map and source_id not in selected_ids:
                selected_ids.append(source_id)
    if not selected_ids:
        return INSUFFICIENT_GROUNDS
    return build_extractive_grounded_draft(
        query=query,
        sources=[source_map[source_id] for source_id in selected_ids],
    )


_MATCH_STOPWORDS = {
    "các", "của", "cho", "chưa", "công", "đồng", "được", "không", "liệu",
    "một", "nhân", "những", "người",
    "này", "pháp", "quy", "định", "theo", "thì", "trong", "trên", "và",
    "việc", "với", "luật", "điều", "khoản", "điểm", "source", "retrieved",
}


def _compact_source_body(source: GroundedSource, max_chars: int = 300) -> str:
    body = " ".join((source.body or source.text).split()).strip(" .,:;")
    if len(body) <= max_chars:
        return body
    shortened = body[:max_chars]
    boundary = max(shortened.rfind(". "), shortened.rfind("; "))
    if boundary >= 100:
        shortened = shortened[: boundary + 1]
    return shortened.rstrip(" ,;:") + "…"


def _is_penalty_source(source: GroundedSource) -> bool:
    return document_priority(source.document, source.text) == 4 or bool(
        re.search(
            r"\b(?:phạt|xử phạt|đình chỉ|tước quyền|khắc phục hậu quả)\b",
            source.body or source.text,
            re.IGNORECASE,
        )
    )


def _natural_source_summary(source: GroundedSource) -> str:
    """Tóm tắt an toàn: đổi cách diễn đạt nhưng không thêm quy tắc ngoài nguồn."""

    body = _compact_source_body(source)
    folded = body.casefold()
    if _is_penalty_source(source):
        lead = "Quy định này mô tả chế tài và đối tượng áp dụng đối với"
    elif re.search(r"\b(?:nghiêm cấm|cấm|không được)\b", folded):
        lead = "Quy định này đặt ra giới hạn hoặc lệnh cấm đối với"
    elif re.search(r"\b(?:phải|nghĩa vụ|có trách nhiệm)\b", folded):
        lead = "Quy định này xác lập nghĩa vụ hoặc trách nhiệm về"
    elif re.search(r"\b(?:được hiểu là|là hành vi|bao gồm)\b", folded):
        lead = "Quy định này giải thích hoặc xác định"
    else:
        lead = "Trọng tâm của quy định là"
    return f"{lead}: {body}."


def _query_fact_candidates(query: str) -> list[str]:
    query = unicodedata.normalize("NFC", query or "")
    candidates = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", query.strip()):
        sentence = sentence.strip()
        if not sentence or re.match(
            r"^(?:hãy|vui lòng|xin)\s+(?:phân tích|tư vấn|giải thích)",
            sentence,
            re.IGNORECASE,
        ):
            continue
        cleaned = re.sub(
            r"(?:\s*(?:và|,)?\s*)?(?:có vi phạm|có thể bị xử lý|bị xử lý|"
            r"xử lý như thế nào|mức phạt|chế tài|phải làm gì|hãy tư vấn|kết luận)\b.*$",
            "",
            sentence,
            flags=re.IGNORECASE,
        ).strip(" ,;:.?!")
        # Câu kết hợp dữ kiện + yêu cầu ("Công ty chia sẻ ... và bị xử lý
        # thế nào?") vẫn giữ phần dữ kiện sau khi cắt vế hỏi. Câu hỏi thuần
        # túy không được biến thành "hành vi" để áp dụng Điều luật.
        if sentence.endswith("?") and cleaned == sentence.strip(" ,;:.?!"):
            continue
        if cleaned:
            candidates.append(cleaned)
    factual = [
        sentence
        for sentence in candidates
        if not re.search(
            r"(?:có vi phạm|bị xử lý|xử lý như thế nào|mức phạt|chế tài|"
            r"phải làm gì|kết luận|tư vấn)",
            sentence,
            re.IGNORECASE,
        )
    ]
    return factual or candidates or [query.strip()]


def _matching_behavior(query: str, source: GroundedSource) -> str:
    facts = _query_fact_candidates(query)
    return max(
        facts,
        key=lambda fact: lexical_relevance(fact, source.text),
        default=query.strip(),
    )


def _shared_terms(query: str, source: GroundedSource, limit: int = 6) -> list[str]:
    query_tokens = re.findall(r"[\w]+", _normalise(query), re.UNICODE)
    source_text = _normalise(source.text)
    output: list[str] = []
    # Cụm hai/ba từ mô tả hành vi có giá trị hơn token rời như "dữ liệu".
    for width in (3, 2):
        for index in range(len(query_tokens) - width + 1):
            phrase_tokens = query_tokens[index : index + width]
            phrase = " ".join(phrase_tokens)
            if (
                any(token not in _MATCH_STOPWORDS for token in phrase_tokens)
                and phrase in source_text
                and phrase not in output
            ):
                output.append(phrase)
            if len(output) >= limit:
                return output
    source_tokens = set(re.findall(r"[\w]+", source_text, re.UNICODE))
    for token in query_tokens:
        if (
            len(token) >= 4
            and token not in _MATCH_STOPWORDS
            and token in source_tokens
            and token not in output
            and not any(token in phrase.split() for phrase in output)
        ):
            output.append(token)
        if len(output) >= limit:
            break
    return output


def _specific_application(
    query: str,
    source: GroundedSource,
    behavior: str,
    summary: str,
) -> str:
    matched = _shared_terms(query, source)
    if matched:
        elements = ", ".join(f'“{term}”' for term in matched)
        return (
            f'Tình huống mô tả “{behavior}”. {summary} Khi đặt hai nội dung cạnh '
            f"nhau, các yếu tố cần đối chiếu cụ thể là {elements}; việc áp dụng "
            "chỉ giới hạn ở các yếu tố được chính đoạn luật này mô tả."
        )
    return (
        f'Tình huống mô tả “{behavior}”, trong khi {summary[0].casefold() + summary[1:]} '
        "Đoạn đã truy xuất chưa cho thấy một yếu tố hành vi trùng khớp bằng câu "
        "chữ; vì vậy chỉ có thể sử dụng quy định nếu dữ kiện thực tế thỏa đúng "
        "mô tả vừa nêu."
    )


def _missing_conditions(
    query: str,
    source: GroundedSource,
    behavior: str,
) -> str:
    body = _compact_source_body(source)
    condition = re.search(
        r"\b(?:nếu|khi|trường hợp)\b[^.;]{10,180}", body, re.IGNORECASE
    )
    if condition:
        return (
            "Cần xác minh tình huống có đáp ứng điều kiện mà nguồn nêu: "
            f'“{condition.group(0).strip()}”.'
        )
    if _is_penalty_source(source):
        return (
            f'Cần xác minh chủ thể thực hiện và hành vi “{behavior}” có khớp đầy '
            f'đủ với mô tả hành vi bị xử phạt trong nguồn: “{body}”.'
        )
    if not _shared_terms(query, source):
        return (
            f'Còn thiếu dữ kiện chứng minh hành vi “{behavior}” thỏa nội dung cụ '
            f'thể mà nguồn quy định: “{body}”.'
        )
    return (
        "Đoạn trích không nêu thêm một điều kiện độc lập; kết luận hiện chỉ "
        f'giới hạn ở việc đối chiếu hành vi “{behavior}” với nội dung đã trích.'
    )


def build_extractive_grounded_draft(
    *, query: str, sources: Sequence[GroundedSource], max_articles: int = 4
) -> str:
    """Câu trả lời tối thiểu từ top SOURCE khi model làm hỏng toàn bộ contract.

    Không kết luận vượt quá nội dung trích; vẫn cho người dùng thấy căn cứ đúng
    thay vì trả một câu fallback dù retrieval đã có dữ liệu.
    """

    if not sources:
        return INSUFFICIENT_GROUNDS

    selected: list[GroundedSource] = []
    seen_articles: set[tuple[str, str]] = set()
    scored_candidates = [
        (
            source,
            lexical_relevance(query, f"{source.document}\n{source.text}"),
        )
        for source in sources
    ]
    best_relevance = max((score for _, score in scored_candidates), default=0.0)
    # Không đưa mọi SOURCE có chung từ rất rộng như "dữ liệu" vào câu trả lời.
    # Ngưỡng tương đối giữ các căn cứ gần với hành vi mạnh nhất và loại các
    # Điều lạc chủ đề (ví dụ sinh trắc học trong câu hỏi về quảng cáo/vị trí).
    candidates = [
        source
        for source, score in scored_candidates
        if score > 0 and score >= best_relevance * 0.55
    ]
    asks_sanction = bool(
        re.search(
            r"\b(?:xử phạt|mức phạt|phạt tiền|chế tài|bị phạt|bị xử lý|"
            r"xử lý như thế nào)\b",
            query,
            re.IGNORECASE,
        )
    )
    penalty_source = next(
        (
            source
            for source, score in scored_candidates
            if _is_penalty_source(source)
            and score > 0
        ),
        None,
    ) if asks_sanction else None
    core_limit = max(0, max_articles - (1 if penalty_source else 0))

    for source in candidates:
        if source is penalty_source:
            continue
        article_key = (source.document, source.article)
        if article_key in seen_articles:
            continue
        if lexical_relevance(query, f"{source.document}\n{source.text}") <= 0:
            continue
        selected.append(source)
        seen_articles.add(article_key)
        if len(selected) >= core_limit:
            break
    if penalty_source and len(selected) < max_articles:
        selected.append(penalty_source)
        seen_articles.add((penalty_source.document, penalty_source.article))
    for source in candidates:
        if len(selected) >= max_articles:
            break
        article_key = (source.document, source.article)
        if article_key not in seen_articles:
            selected.append(source)
            seen_articles.add(article_key)
    if not selected:
        return INSUFFICIENT_GROUNDS

    blocks: list[str] = []
    for index, source in enumerate(selected, start=1):
        quote_marker = f"[[QUOTE:{source.source_id}]]"
        cite_marker = f"[[CITE:{source.source_id}]]"
        is_penalty = _is_penalty_source(source)
        summary = _natural_source_summary(source)
        behavior = _matching_behavior(query, source)
        application = _specific_application(query, source, behavior, summary)
        missing = _missing_conditions(query, source, behavior)
        confidence = (
            f"🔴 Chưa đủ căn cứ: {missing}"
            if is_penalty
            else f"🟡 Chưa đủ điều kiện kết luận: {missing}"
        )
        blocks.append(
            f"""### {source.document} — Điều {source.article}
{quote_marker}

- **Nội dung điều luật:** {summary}
- **Hành vi phù hợp:** {behavior}
- **Phân tích:** {application}
- **Điều kiện còn thiếu:** {missing}
- **Căn cứ pháp lý:** {cite_marker}
- **Đánh giá:** {confidence} {cite_marker}"""
        )

    if penalty_source and penalty_source in selected:
        penalty_quote = f"[[QUOTE:{penalty_source.source_id}]]"
        penalty_cite = f"[[CITE:{penalty_source.source_id}]]"
        penalty_behavior = _matching_behavior(query, penalty_source)
        penalty_summary = _natural_source_summary(penalty_source)
        penalty_application = _specific_application(
            query, penalty_source, penalty_behavior, penalty_summary
        )
        penalty_missing = _missing_conditions(query, penalty_source, penalty_behavior)
        sanction_section = f"""{penalty_quote}

- **Nội dung chế tài:** {penalty_summary} {penalty_cite}
- **Hành vi cần đối chiếu:** {penalty_behavior} {penalty_cite}
- **Phân tích áp dụng:** {penalty_application} {penalty_cite}
- **Cá nhân hay tổ chức:** Chỉ xác định theo đúng đối tượng được nêu trong đoạn trích; nếu tình huống chưa xác định chủ thể thì chưa thể chọn mức áp dụng. {penalty_cite}
- **Điều kiện áp dụng:** {penalty_missing} {penalty_cite}
- **Cộng dồn:** Chưa đủ căn cứ để khẳng định các mức được cộng dồn. {penalty_cite}
- **Đánh giá:** 🔴 Chưa đủ căn cứ để áp dụng mức chế tài cụ thể cho tình huống. {penalty_cite}"""
    else:
        sanction_section = """- **Đã xác định được:** Có căn cứ liên quan đến hành vi cần đánh giá.
- **Chưa xác định được:** Chưa có đủ nội dung để xác định toàn bộ hậu quả và cách áp dụng.
- **Muốn kết luận cần:** Bổ sung căn cứ trực tiếp về hậu quả và các điều kiện áp dụng còn thiếu."""

    questions = extract_user_questions(query) or [query.strip()]
    all_cite = f"[[CITE:{','.join(source.source_id for source in selected)}]]"
    source_summary = " ".join(
        _natural_source_summary(source) for source in selected[:2]
    )
    source_corpus = _normalise(" ".join(source.text for source in selected))
    synthesis_items: list[str] = []
    for index, question in enumerate(questions, start=1):
        folded_question = _normalise(question)
        asks_sanction = bool(
            re.search(r"(?:xử phạt|mức phạt|phạt|chế tài|bị xử lý)", folded_question)
        )
        asks_specific_obligation = bool(
            re.search(r"(?:phải|nghĩa vụ|trách nhiệm|xóa|xoá|gỡ bỏ|bồi thường)", folded_question)
        )
        requested_actions = re.findall(
            r"(?:xóa dữ liệu|xoá dữ liệu|xóa|xoá|gỡ bỏ|bồi thường|hoàn trả|"
            r"thông báo|xin sự đồng ý)",
            folded_question,
        )
        question_terms = [
            token
            for token in re.findall(r"[\w]+", folded_question, re.UNICODE)
            if len(token) >= 4 and token not in _MATCH_STOPWORDS
        ]
        has_specific_support = any(term in source_corpus for term in question_terms)

        if asks_sanction and penalty_source:
            direct = "Chưa đủ căn cứ để kết luận mức xử phạt cụ thể cho hành vi được hỏi."
            why = (
                f"Nguồn đã truy xuất có quy định chế tài, nhưng việc chọn mức phụ thuộc "
                f"vào đúng hành vi, chủ thể và điều kiện được mô tả trong nguồn. {all_cite}"
            )
            grounds = all_cite
            missing = (
                "Cần làm rõ chủ thể thực hiện, đầy đủ dấu hiệu hành vi và điều kiện "
                "áp dụng mức phạt trong đoạn đã trích."
            )
        elif asks_specific_obligation and (
            not has_specific_support
            or any(action not in source_corpus for action in requested_actions)
        ):
            direct = (
                "Không tìm thấy căn cứ trong các nguồn hiện có để khẳng định nghĩa vụ cụ thể "
                "mà câu hỏi yêu cầu."
            )
            why = (
                "Không tìm thấy căn cứ nào trong các nguồn hiện có mô tả nghĩa vụ "
                "được hỏi; suy ra nghĩa vụ đó từ quy định về hành vi khác sẽ vượt "
                "quá dữ liệu retrieval."
            )
            grounds = "Không tìm thấy căn cứ trong các nguồn hiện có cho câu hỏi này."
            missing = (
                "Cần truy xuất quy định trực tiếp về nghĩa vụ được hỏi và các điều "
                "kiện phát sinh nghĩa vụ đó."
            )
        else:
            direct = "Chưa đủ căn cứ để kết luận dứt khoát câu hỏi này."
            why = (
                f"Các nguồn hiện có xác định được một phần quy tắc liên quan: "
                f"{source_summary} Việc kết luận chỉ được giới hạn trong các nội dung đó. {all_cite}"
            )
            grounds = all_cite
            missing = (
                "Cần bổ sung dữ kiện đáp ứng các điều kiện được nêu trong từng nguồn "
                "và căn cứ trực tiếp cho phần câu hỏi chưa được bao phủ."
            )

        synthesis_items.append(
            f"""### {index}. {question}
- **Trả lời trực tiếp:** {direct}
- **Vì sao:** {why}
- **Căn cứ:** {grounds}
- **Còn thiếu:** {missing}"""
        )

    draft = f"""## Tóm tắt tình huống
{query}

## Các vấn đề pháp lý
- Xác định bản chất pháp lý của các hành vi được mô tả.
- Xác định điều kiện áp dụng và hậu quả pháp lý trong phạm vi căn cứ đã truy xuất.

## Phân tích
{(chr(10) * 2).join(blocks)}

## Chế tài
{sanction_section}

## Trả lời câu hỏi của người dùng
{(chr(10) * 2).join(synthesis_items)}"""
    validation = validate_grounded_draft(draft, sources, query=query)
    if validation.is_valid:
        return draft

    # Một SOURCE có câu dẫn chiếu/phần chế tài phức tạp có thể khiến strict
    # validator loại cả bản extractive. Thử giữ từng căn cứ độc lập trước khi
    # chuyển sang fallback bảo thủ; không để một block xấu làm mất block tốt.
    if len(selected) > 1:
        for source in selected:
            single_source_draft = build_extractive_grounded_draft(
                query=query,
                sources=[source],
                max_articles=1,
            )
            if single_source_draft != INSUFFICIENT_GROUNDS:
                return single_source_draft

    logger.warning(
        "Extractive grounded fallback không qua strict validation: %s",
        "; ".join(validation.issues),
    )
    return INSUFFICIENT_GROUNDS


def _missing_for_question(question: str) -> str:
    folded = _normalise(question)
    if re.search(r"(?:xử phạt|mức phạt|phạt|chế tài|hành chính|hình sự)", folded):
        return (
            "Cần thêm căn cứ trực tiếp về loại trách nhiệm, hành vi bị xử lý, "
            "chủ thể và đầy đủ điều kiện áp dụng."
        )
    if re.search(r"(?:quyền|xâm phạm)", folded):
        return (
            "Cần thêm căn cứ trực tiếp xác định từng quyền được bảo vệ và các "
            "tình tiết làm phát sinh việc xâm phạm quyền đó."
        )
    if re.search(r"(?:phải|nghĩa vụ|trách nhiệm|xóa|xoá|bồi thường)", folded):
        return (
            "Cần thêm căn cứ trực tiếp về nghĩa vụ hoặc trách nhiệm được hỏi "
            "và điều kiện làm phát sinh nghĩa vụ đó."
        )
    return (
        "Cần thêm căn cứ điều chỉnh trực tiếp hành vi được hỏi và các dữ kiện "
        "đáp ứng đầy đủ điều kiện của quy định đó."
    )


def build_conservative_grounded_draft(
    *, query: str, sources: Sequence[GroundedSource], max_articles: int = 3
) -> str:
    """Fallback cuối chỉ trình bày trích đoạn thật và kết luận thiếu căn cứ.

    Hàm này không diễn giải quy tắc hay mức phạt. Nó được dùng khi cả generation,
    repair và strict extractive fallback đều lỗi, nhờ vậy nguồn còn tồn tại sẽ
    không bị biến thành một câu từ chối trống.
    """

    if not sources:
        return INSUFFICIENT_GROUNDS

    selected: list[GroundedSource] = []
    seen_articles: set[tuple[str, str]] = set()
    for source in sources:
        article_key = (source.document, source.article)
        if article_key in seen_articles:
            continue
        selected.append(source)
        seen_articles.add(article_key)
        if len(selected) >= max_articles:
            break
    if not selected:
        return INSUFFICIENT_GROUNDS

    all_cite = f"[[CITE:{','.join(source.source_id for source in selected)}]]"
    source_blocks = []
    for index, source in enumerate(selected, start=1):
        source_blocks.append(
            f"""### Căn cứ hiện có {index}
[[QUOTE:{source.source_id}]]

- **Nội dung có thể xác nhận:** Trích đoạn phía trên là nội dung pháp luật hiện có cho việc xem xét tình huống. [[CITE:{source.source_id}]]
- **Giới hạn phân tích:** Chưa đủ cơ sở để mở rộng nội dung trích đoạn thành một kết luận pháp lý dứt khoát cho toàn bộ câu hỏi. [[CITE:{source.source_id}]]"""
        )

    questions = extract_user_questions(query) or [query.strip()]
    synthesis_items = []
    for index, question in enumerate(questions, start=1):
        synthesis_items.append(
            f"""### {index}. {question}
- **Trả lời trực tiếp:** Chưa đủ căn cứ để kết luận dứt khoát câu hỏi này.
- **Vì sao:** Các căn cứ hiện có mới xác nhận được những nội dung được trích dẫn ở phần Phân tích; chưa đủ để kết luận vượt ra ngoài các nội dung đó. {all_cite}
- **Căn cứ:** {all_cite}
- **Còn thiếu:** {_missing_for_question(question)}"""
        )

    return f"""## Tóm tắt tình huống
{query}

## Các vấn đề pháp lý
{chr(10).join(f'- {question}' for question in questions)}

## Phân tích
{(chr(10) * 2).join(source_blocks)}

## Chế tài
- **Đã xác định được:** Có các trích đoạn pháp luật liên quan được trình bày ở phần Phân tích. {all_cite}
- **Chưa xác định được:** Chưa đủ căn cứ để xác định loại trách nhiệm hoặc mức chế tài cụ thể.
- **Muốn kết luận cần:** Bổ sung căn cứ trực tiếp về chế tài và đầy đủ điều kiện áp dụng cho hành vi trong tình huống.

## Trả lời câu hỏi của người dùng
{(chr(10) * 2).join(synthesis_items)}"""


def _validate_conservative_grounded_draft(
    draft: str,
    sources: Sequence[GroundedSource],
    *,
    query: str,
) -> bool:
    """Validation tối thiểu cho fallback không đưa ra kết luận khẳng định."""

    if not sources or not draft or draft == INSUFFICIENT_GROUNDS:
        return False
    source_ids = {source.source_id for source in sources}
    markers = list(_MARKER_RE.finditer(draft))
    if not markers or any(
        source_id not in source_ids
        for marker in markers
        for source_id in (item.strip() for item in marker.group("ids").split(","))
    ):
        return False
    synthesis = _section(draft, "Trả lời câu hỏi của người dùng", ())
    items = [item for item in re.split(r"(?m)^###\s+", synthesis) if item.strip()]
    questions = extract_user_questions(query) or [query.strip()]
    return len(items) == len(questions) and all(
        "**Trả lời trực tiếp:** Chưa đủ căn cứ" in item
        and "**Vì sao:**" in item
        and "**Căn cứ:**" in item
        and "**Còn thiếu:**" in item
        and "[[CITE:" in item
        for item in items
    )


def build_safe_grounded_fallback(
    *,
    query: str,
    sources: Sequence[GroundedSource],
    is_complete: bool,
) -> str:
    """Luôn giữ phần nguồn thật nếu retrieval còn ít nhất một SOURCE."""

    extractive = build_extractive_grounded_draft(query=query, sources=sources)
    if extractive != INSUFFICIENT_GROUNDS and validate_grounded_draft(
        extractive,
        sources,
        is_complete=is_complete,
        query=query,
    ).is_valid:
        return extractive

    conservative = build_conservative_grounded_draft(query=query, sources=sources)
    if _validate_conservative_grounded_draft(
        conservative,
        sources,
        query=query,
    ):
        logger.warning(
            "Dùng conservative grounded fallback vì generation/repair/extractive đều không hợp lệ."
        )
        return conservative
    return INSUFFICIENT_GROUNDS


def _short_quote(source: GroundedSource, max_chars: int = 320) -> str:
    text = source.body or source.text
    if len(text) <= max_chars:
        return text
    candidate = text[:max_chars]
    sentence_end = max(candidate.rfind(". "), candidate.rfind("; "))
    if sentence_end >= 100:
        candidate = candidate[: sentence_end + 1]
    return candidate.rstrip(" ,;:") + "…"


def _render_marker(match: re.Match[str], source_map: dict[str, GroundedSource]) -> str:
    source_ids = [item.strip() for item in match.group("ids").split(",")]
    selected = [source_map[source_id] for source_id in source_ids if source_id in source_map]
    if match.group("kind") == "QUOTE":
        blocks = []
        for source in selected:
            quote = _short_quote(source).replace("\n", " ")
            blocks.append(
                f"> “{quote}”\n>\n> (Trích từ: {citation_label(source, markdown=True)})"
            )
        return "\n\n".join(blocks)
    labels = "; ".join(citation_label(source, markdown=True) for source in selected)
    return f"(Căn cứ: {labels})"


def render_grounded_answer(
    draft: str,
    sources: Sequence[GroundedSource],
    *,
    is_complete: bool,
) -> str:
    """Thay marker bằng dữ liệu xác định và dựng danh mục nguồn đã dùng."""

    cleaned = "\n".join(
        line for line in (draft or "").splitlines() if not _TECHNICAL_LINE_RE.match(line)
    ).strip()
    source_heading = _SOURCE_HEADING_RE.search(cleaned)
    if source_heading:
        cleaned = cleaned[: source_heading.start()].rstrip()

    source_map = {source.source_id: source for source in sources}
    used_ids: list[str] = []
    for marker in _MARKER_RE.finditer(cleaned):
        for source_id in (item.strip() for item in marker.group("ids").split(",")):
            if source_id in source_map and source_id not in used_ids:
                used_ids.append(source_id)
    rendered = _MARKER_RE.sub(lambda match: _render_marker(match, source_map), cleaned)

    sections: list[str] = []
    # Khi hoàn toàn không có nguồn, yêu cầu product là trả đúng duy nhất câu
    # safe fallback. Warning completeness chỉ gắn cho câu trả lời còn có phần
    # căn cứ đã retrieve nhưng recursive retrieval chưa hoàn tất.
    if (
        cleaned != INSUFFICIENT_GROUNDS
        and not is_complete
        and (used_ids or sources)
    ):
        sections.append(f"> **Lưu ý:** {INCOMPLETE_GROUNDS_WARNING}")
    sections.append(rendered or INSUFFICIENT_GROUNDS)
    if used_ids:
        lines = ["## Căn cứ pháp lý", ""]
        for source_id in used_ids:
            lines.append(f"- {citation_label(source_map[source_id], markdown=True)}.")
        sections.append("\n".join(lines))
    return "\n\n".join(section for section in sections if section).strip()


def build_repair_prompt(
    *,
    query: str,
    draft: str,
    issues: Sequence[str],
    sources: Sequence[GroundedSource],
    is_complete: bool,
) -> str:
    issue_text = "\n".join(f"- {issue}" for issue in issues)
    completeness = "đầy đủ" if is_complete else "chưa đầy đủ"
    return f"""
Bạn là bộ sửa bản nháp pháp lý theo Grounded Reasoning. Dữ liệu retrieval đang {completeness}.
Chỉ sửa lỗi được liệt kê; không thêm kiến thức ngoài SOURCE. Không tự tạo tên văn bản,
Điều, Khoản, Điểm hay mức phạt. Nếu có dù chỉ một SOURCE liên quan, phải phân tích
phần đã có và tách phần còn thiếu; không được từ chối toàn bộ.

LỖI VALIDATION:
{issue_text}

SOURCE HỢP LỆ:
{format_grounded_context(sources)}

CÂU HỎI:
{query}

BẢN NHÁP CẦN SỬA:
{draft}

Xuất lại toàn bộ câu trả lời theo đúng 5 heading cấp 2 sau (không đổi tên):
## Tóm tắt tình huống
## Các vấn đề pháp lý
## Phân tích
## Chế tài
## Trả lời câu hỏi của người dùng

Trong ## Phân tích, mỗi vấn đề là heading cấp 3 và có đủ sáu trường:
**Nội dung điều luật**, **Hành vi phù hợp**, **Phân tích**,
**Điều kiện còn thiếu**, **Căn cứ pháp lý**, **Đánh giá**. Mỗi điều luật phải
được tóm tắt và đối chiếu riêng với tình tiết, không dùng lại một mẫu chung.
Không dùng các câu "tình tiết thuộc đúng nhóm hoạt động", "đây là căn cứ trực
tiếp", "có liên hệ trực tiếp", "có dấu hiệu". Đặt [[QUOTE:Sx]]
trước phân tích; **Đánh giá** bắt đầu bằng 🟢 Đủ căn cứ /
🟡 Chưa đủ điều kiện kết luận /
🔴 Chưa đủ căn cứ và kết thúc [[CITE:Sx]]. Nếu thiếu dữ liệu, chia rõ:
Đã xác định được / Chưa xác định được / Muốn kết luận cần. Không tạo mục
Căn cứ pháp lý cuối bài; hệ thống tự dựng.

Trong ## Trả lời câu hỏi của người dùng, quay lại từng câu hỏi gốc và tạo một
heading cấp 3 đánh số cho mỗi câu hỏi. Mỗi mục có đúng bốn trường:
**Trả lời trực tiếp**, **Vì sao**, **Căn cứ**, **Còn thiếu**. Kết luận mở đầu
bằng Có / Không / Có dấu hiệu vi phạm [hành vi cụ thể] / Chưa đủ căn cứ /
Không tìm thấy căn cứ. Không đặt tên Điều luật làm heading và không kết luận
lần lượt theo SOURCE. Không được kết luận "Không" chỉ vì không tìm thấy nguồn;
phải nói "Không tìm thấy căn cứ trong các nguồn hiện có...". Trường **Còn thiếu** phải
giải thích cụ thể, không chỉ ghi "Không có". Không tự tính con số chế tài chưa
ghi nguyên văn trong SOURCE.
""".strip()


def chunk_markdown_for_streaming(text: str, target_size: int = 48) -> list[str]:
    """Chia bản đã validate ở biên whitespace, giữ nguyên Markdown tuyệt đối."""

    if not text:
        return []
    chunks = re.findall(r"\S+\s*|\s+", text)
    output: list[str] = []
    current = ""
    for chunk in chunks:
        current += chunk
        if len(current) >= target_size or "\n\n" in current:
            output.append(current)
            current = ""
    if current:
        output.append(current)
    return output
