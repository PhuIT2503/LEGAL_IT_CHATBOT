"""Taxonomy, corpus registry và bộ chọn domain pháp luật cho Retrieval Phase 1.

Module này cố ý không dùng LLM: Phase 1 cần một cổng domain nhẹ, xác định và
dễ kiểm thử.  Registry theo văn bản là nguồn sự thật khi ingest; bộ rule theo
query chỉ quyết định những domain nào được phép đi vào Hybrid Retrieval.

Domain là multi-label.  Một nghị định xử phạt về an ninh mạng, ví dụ, có thể
đồng thời mang ``cybersecurity`` và ``administrative_penalty``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable


SUPPORTED_LEGAL_DOMAINS: tuple[str, ...] = (
    "cybersecurity",
    "personal_data",
    "civil_personality",
    "advertising",
    "intellectual_property",
    "criminal",
    "administrative_penalty",
    "consumer_protection",
    "digital_technology",
    "artificial_intelligence",
    "data_governance",
    "electronic_transactions",
    "e_commerce",
    "telecommunications",
    "digital_content",
    "finance_payment",
    "general_legal",
)

# Chỉ dùng khi query không khớp rule nào. Giới hạn năm domain cốt lõi của
# corpus để vẫn giữ recall thực dụng mà không quay lại tìm toàn bộ kho luật.
DEFAULT_QUERY_DOMAINS: tuple[str, ...] = (
    "cybersecurity",
    "personal_data",
    "digital_technology",
    "electronic_transactions",
    "telecommunications",
)


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFD", str(text or "").casefold())
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return re.sub(r"\s+", " ", text.replace("đ", "d")).strip()


# Pattern được so với tên file đã bỏ dấu. Thứ tự domain trong từng tuple ổn
# định để payload và log có thể so sánh giữa các lần ingest.
DOCUMENT_DOMAIN_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (r"2023_361|so huu tri tue", ("intellectual_property",)),
    (r"luat an ninh mang", ("cybersecurity",)),
    (r"luat bao ve du lieu ca nhan", ("personal_data",)),
    (
        r"luat bao ve quyen loi nguoi tieu dung",
        ("consumer_protection", "advertising"),
    ),
    (
        r"luat (?:cntt|cong nghe thong tin)",
        ("digital_technology", "cybersecurity"),
    ),
    (
        r"luat cong nghiep cong nghe so",
        ("digital_technology", "artificial_intelligence"),
    ),
    (r"luat du lieu 2024", ("data_governance",)),
    (r"luat giao dich dien tu", ("electronic_transactions",)),
    (r"luat vien thong", ("telecommunications",)),
    (
        r"nghi dinh 147 2024",
        ("cybersecurity", "telecommunications", "digital_content", "advertising"),
    ),
    (
        r"nghi dinh 15 2020",
        (
            "administrative_penalty",
            "cybersecurity",
            "telecommunications",
            "electronic_transactions",
            "advertising",
        ),
    ),
    (r"nghi dinh 17 2023", ("intellectual_property",)),
    (
        r"nghi dinh 211 2025",
        ("cybersecurity", "administrative_penalty", "digital_technology"),
    ),
    (
        r"nghi dinh 52 2013",
        (
            "e_commerce",
            "electronic_transactions",
            "consumer_protection",
            "personal_data",
        ),
    ),
    (
        r"nghi dinh 52 2024",
        ("finance_payment", "electronic_transactions"),
    ),
    (r"nghi dinh 53 2022", ("cybersecurity",)),
    (r"nghi dinh 71 2007", ("digital_technology",)),
    (r"nghi dinh 85 2016", ("cybersecurity",)),
)


def document_legal_domains(source: str) -> list[str]:
    """Trả về multi-domain ổn định cho một văn bản trong corpus.

    Văn bản mới chưa có registry vẫn nhận ``general_legal`` thay vì payload
    rỗng. Khi thêm tài liệu, chỉ cần bổ sung một rule ở bảng phía trên.
    """

    folded_source = _fold(source)
    domains: list[str] = []
    for pattern, matched_domains in DOCUMENT_DOMAIN_RULES:
        if re.search(pattern, folded_source):
            for domain in matched_domains:
                if domain not in domains:
                    domains.append(domain)
    return domains or ["general_legal"]


# (cụm từ đã bỏ dấu, trọng số). Rule hành vi cụ thể có trọng số cao hơn từ
# ngữ cảnh rộng; nhờ đó "deepfake quảng cáo" không kích hoạt SHTT nếu không có
# dữ kiện về tác phẩm/bản quyền/bản ghi có sẵn.
QUERY_DOMAIN_SIGNALS: dict[str, tuple[tuple[str, float], ...]] = {
    "cybersecurity": (
        ("sql injection", 6.0),
        ("deepfake", 5.0),
        ("tan cong mang", 5.0),
        ("truy cap trai phep", 5.0),
        ("khai thac lo hong", 5.0),
        ("lo hong bao mat", 4.0),
        ("gia mao video", 4.0),
        ("gia mao giong noi", 4.0),
        ("ma doc", 4.0),
        ("malware", 4.0),
        ("hack", 4.0),
        ("an ninh mang", 4.0),
        ("an toan thong tin", 3.0),
    ),
    "personal_data": (
        ("du lieu ca nhan", 6.0),
        ("thong tin ca nhan", 5.0),
        ("du lieu sinh trac", 5.0),
        ("khuon mat", 4.0),
        ("giong noi", 3.0),
        ("hinh anh ca nhan", 4.0),
        ("chu the du lieu", 5.0),
        ("dong y", 2.0),
        ("deepfake", 2.0),
    ),
    "civil_personality": (
        ("quyen hinh anh", 6.0),
        ("hinh anh nguoi khac", 5.0),
        ("nguoi noi tieng", 4.0),
        ("danh du", 4.0),
        ("nhan pham", 4.0),
        ("uy tin", 3.0),
        ("doi tu", 4.0),
        ("deepfake", 3.0),
    ),
    "advertising": (
        ("quang cao", 6.0),
        ("tiep thi", 4.0),
        ("marketing", 4.0),
        ("quang ba san pham", 5.0),
        ("dai dien thuong hieu", 4.0),
    ),
    "intellectual_property": (
        ("so huu tri tue", 6.0),
        ("quyen tac gia", 6.0),
        ("ban quyen", 6.0),
        ("tac pham", 4.0),
        ("nhan hieu", 5.0),
        ("sang che", 5.0),
        ("sao chep video", 5.0),
        ("su dung ban ghi", 5.0),
    ),
    "criminal": (
        ("truy cuu hinh su", 6.0),
        ("trach nhiem hinh su", 6.0),
        ("toi pham", 5.0),
        ("bo luat hinh su", 6.0),
        ("phat tu", 5.0),
    ),
    "administrative_penalty": (
        ("xu phat", 5.0),
        ("muc phat", 5.0),
        ("phat tien", 5.0),
        ("che tai", 4.0),
        ("bi phat", 4.0),
        ("vi pham hanh chinh", 5.0),
    ),
    "consumer_protection": (
        ("nguoi tieu dung", 6.0),
        ("lua doi khach hang", 5.0),
        ("gay nham lan", 3.0),
        ("quang cao sai su that", 5.0),
    ),
    "digital_technology": (
        ("cong nghe so", 5.0),
        ("cong nghe thong tin", 5.0),
        ("phan mem", 3.0),
        ("nen tang so", 3.0),
        ("he thong thong tin", 3.0),
    ),
    "artificial_intelligence": (
        ("tri tue nhan tao", 6.0),
        ("deepfake", 5.0),
        ("ai tao sinh", 5.0),
        ("generative ai", 5.0),
        ("mo hinh ai", 4.0),
    ),
    "data_governance": (
        ("quan tri du lieu", 5.0),
        ("du lieu so", 4.0),
        ("co so du lieu", 2.0),
        ("trung tam du lieu", 5.0),
        ("chia se du lieu", 3.0),
    ),
    "electronic_transactions": (
        ("giao dich dien tu", 6.0),
        ("chu ky dien tu", 5.0),
        ("chu ky so", 5.0),
        ("thong diep du lieu", 5.0),
        ("hop dong dien tu", 5.0),
    ),
    "e_commerce": (
        ("thuong mai dien tu", 6.0),
        ("san thuong mai", 5.0),
        ("website ban hang", 5.0),
        ("ban hang online", 4.0),
    ),
    "telecommunications": (
        ("vien thong", 6.0),
        ("sim", 4.0),
        ("so dien thoai", 3.0),
        ("nha mang", 5.0),
        ("tan so vo tuyen", 5.0),
        ("dich vu internet", 3.0),
    ),
    "digital_content": (
        ("mang xa hoi", 5.0),
        ("thong tin tren mang", 4.0),
        ("noi dung so", 4.0),
        ("tin gia", 5.0),
        ("phat tan", 2.0),
    ),
    "finance_payment": (
        ("thanh toan khong dung tien mat", 6.0),
        ("vi dien tu", 5.0),
        ("dich vu thanh toan", 5.0),
        ("tai khoan thanh toan", 5.0),
    ),
}


@dataclass(frozen=True)
class DomainSelection:
    selected: tuple[str, ...]
    filtered: tuple[str, ...]
    scores: dict[str, float]
    used_fallback: bool = False


def select_legal_domains(
    query: str,
    *,
    max_domains: int = 5,
    minimum_score: float = 2.0,
) -> DomainSelection:
    """Chọn tối đa ``max_domains`` bằng tín hiệu pháp lý trong query gốc."""

    folded_query = _fold(query)
    scores: dict[str, float] = {}
    for domain, signals in QUERY_DOMAIN_SIGNALS.items():
        score = sum(weight for phrase, weight in signals if phrase in folded_query)
        if score > 0:
            scores[domain] = score

    ranked = sorted(scores, key=lambda domain: (-scores[domain], domain))
    selected = [domain for domain in ranked if scores[domain] >= minimum_score][
        : max(1, max_domains)
    ]
    used_fallback = not selected
    if used_fallback:
        selected = list(DEFAULT_QUERY_DOMAINS[: max(1, max_domains)])

    selected_set = set(selected)
    filtered = tuple(
        domain for domain in SUPPORTED_LEGAL_DOMAINS if domain not in selected_set
    )
    return DomainSelection(
        selected=tuple(selected),
        filtered=filtered,
        scores=scores,
        used_fallback=used_fallback,
    )


def count_candidates_by_domain(
    records: Iterable[dict], selected_domains: Iterable[str]
) -> dict[str, int]:
    """Đếm candidate theo multi-domain; một candidate có thể thuộc nhiều nhóm."""

    counts = {domain: 0 for domain in selected_domains}
    for record in records:
        metadata = record.get("metadata") or {}
        candidate_domains = set(metadata.get("legal_domains") or [])
        for domain in counts:
            if domain in candidate_domains:
                counts[domain] += 1
    return counts
