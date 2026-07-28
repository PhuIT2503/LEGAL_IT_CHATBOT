"""Mở rộng truy vấn theo miền pháp lý, chỉ để tăng recall.

Expansion không được biến một từ chung (ví dụ ``dữ liệu`` trong ``cơ sở dữ
liệu``) thành một kết luận về miền pháp luật (``dữ liệu cá nhân``). Các nhóm
dưới đây vì vậy dùng trigger đủ đặc hiệu và bổ sung thuật ngữ mô tả cùng hành
vi; query gốc vẫn là nguồn duy nhất cho reranking/final relevance.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

from src.retrieval.legal_event import (
    CanonicalLegalEvent,
    extract_canonical_legal_event,
)


def _fold(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value or "")
    return "".join(
        char for char in decomposed if unicodedata.category(char) != "Mn"
    ).casefold().replace("đ", "d")


@dataclass(frozen=True)
class ExpansionRule:
    """Một intent pháp lý với trigger đủ đặc hiệu và các thuật ngữ recall."""

    triggers: tuple[str, ...]
    terms: tuple[str, ...]
    excluded_when: tuple[str, ...] = ()

    def matches(self, folded_query: str) -> bool:
        return (
            any(trigger in folded_query for trigger in self.triggers)
            and not any(exclusion in folded_query for exclusion in self.excluded_when)
        )


_RULES: tuple[ExpansionRule, ...] = (
    # An ninh mạng phải đứng trước privacy: "tải cơ sở dữ liệu" trong một vụ
    # SQL Injection là hành vi xâm nhập/chiếm đoạt, không mặc nhiên là xử lý
    # dữ liệu cá nhân dựa trên sự đồng ý.
    ExpansionRule(
        triggers=(
            "sql injection",
            "injection",
            "lo hong",
            "xam nhap",
            "tan cong mang",
            "tan cong website",
            "hack website",
            "truy cap trai phep",
            "tai xuong co so du lieu",
            "chiem doat du lieu",
        ),
        terms=(
            "xâm nhập trái phép hệ thống thông tin",
            "khai thác điểm yếu lỗ hổng bảo mật",
            "tấn công mạng",
            "truy nhập trái phép thông tin",
            "chiếm đoạt dữ liệu trong hệ thống thông tin",
            "phá hoại hệ thống thông tin",
        ),
    ),
    ExpansionRule(
        triggers=(
            "du lieu ca nhan",
            "thong tin ca nhan",
            "chu the du lieu",
            "danh sach khach hang",
            "email khach hang",
            "so dien thoai khach hang",
            "vi tri nguoi dung",
            "quyen rieng tu",
        ),
        terms=(
            "dữ liệu cá nhân",
            "xử lý dữ liệu cá nhân",
            "thu thập dữ liệu cá nhân",
            "sử dụng dữ liệu cá nhân",
            "tiết lộ dữ liệu cá nhân",
            "chia sẻ dữ liệu cá nhân",
            "chuyển giao dữ liệu cá nhân",
            "sự đồng ý của chủ thể dữ liệu",
            "nghĩa vụ bảo vệ dữ liệu cá nhân",
        ),
    ),
    ExpansionRule(
        triggers=("copy", "sao chep", "usb", "nhan vien", "danh sach khach hang"),
        terms=(
            "sao chép dữ liệu",
            "bí mật kinh doanh",
            "chiếm đoạt thông tin",
            "nghĩa vụ bảo mật của nhân viên",
        ),
    ),
    ExpansionRule(
        triggers=("quang cao", "doi tac quang cao", "tiep thi"),
        terms=(
            "kinh doanh dịch vụ quảng cáo",
            "sử dụng dữ liệu cho quảng cáo",
            "chia sẻ dữ liệu cho bên thứ ba",
        ),
    ),
    ExpansionRule(
        triggers=("lua dao", "gia mao", "chiem doat tai san"),
        terms=("lừa đảo trên không gian mạng", "giả mạo thông tin", "chiếm đoạt tài sản"),
    ),
    ExpansionRule(
        triggers=("hop dong", "giao dich", "chu ky so"),
        terms=("giao dịch điện tử", "hiệu lực hợp đồng", "chữ ký điện tử", "nghĩa vụ các bên"),
    ),
    ExpansionRule(
        triggers=("xu phat", "muc phat", "bi xu ly", "che tai", "phat tien"),
        terms=(
            "xử phạt vi phạm hành chính",
            "hành vi bị xử phạt",
            "mức phạt cá nhân tổ chức",
            "biện pháp khắc phục hậu quả",
        ),
    ),
)


def expand_legal_query(
    query: str,
    max_terms: int = 18,
    *,
    event: CanonicalLegalEvent | None = None,
) -> tuple[str, list[str]]:
    """Trả query recall và thuật ngữ thêm vào; giữ nguyên API hiện tại."""

    folded = " ".join(re.findall(r"[a-z0-9]+", _fold(query)))
    event = event or extract_canonical_legal_event(query)
    additions: list[str] = []
    for rule in _RULES:
        if rule.matches(folded):
            additions.extend(rule.terms)
    if "sell_personal_data" in event.actions:
        additions.extend(
            (
                "mua bán dữ liệu cá nhân",
                "nghiêm cấm mua bán dữ liệu cá nhân",
                "thu lợi từ mua bán dữ liệu cá nhân",
            )
        )
    if "retain_personal_data" in event.actions:
        additions.extend(
            (
                "lưu giữ dữ liệu cá nhân trái quy định",
                "sử dụng dữ liệu cá nhân của người khác",
                "chiếm đoạt dữ liệu cá nhân",
            )
        )
    if event.liability_intent and "personal_data" in event.objects:
        additions.extend(
            (
                "xử lý vi phạm dữ liệu cá nhân",
                "bồi thường thiệt hại dữ liệu cá nhân",
                "truy cứu trách nhiệm hình sự dữ liệu cá nhân",
            )
        )
    additions = list(dict.fromkeys(additions))[:max_terms]
    if not additions:
        return query, []
    return f"{query}\nThuật ngữ pháp lý liên quan: " + "; ".join(additions), additions
