"""Canonical, deterministic legal-event extraction shared by retrieval stages.

The event is deliberately descriptive rather than adjudicative: it records
what the user states and maps clear concepts to retrieval labels.  It never
decides whether an act is lawful and never invents consent, authorisation,
amounts, dates, or affected-person counts.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Any


def fold_legal_text(value: Any) -> str:
    text = unicodedata.normalize("NFD", str(value or "").casefold())
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return re.sub(r"\s+", " ", text.replace("đ", "d")).strip()


def _unique(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _number(value: str) -> int | None:
    digits = re.sub(r"\D", "", value or "")
    return int(digits) if digits else None


def _money_vnd(folded: str) -> int | None:
    patterns = (
        r"(?:ban|gia|doi lay|nhan|thu|thu loi|kiem duoc).{0,35}?"
        r"(?P<number>\d+(?:[.,]\d+)?)\s*(?P<unit>trieu|ty|nghin|dong|vnd)\b",
        r"(?P<number>\d+(?:[.,]\d+)?)\s*(?P<unit>trieu|ty|nghin|dong|vnd)"
        r".{0,25}?(?:de ban|tien ban|thu loi|nhan duoc)",
    )
    for pattern in patterns:
        match = re.search(pattern, folded)
        if not match:
            continue
        raw = match.group("number").replace(",", ".")
        try:
            amount = float(raw)
        except ValueError:
            continue
        multiplier = {
            "ty": 1_000_000_000,
            "trieu": 1_000_000,
            "nghin": 1_000,
            "dong": 1,
            "vnd": 1,
        }[match.group("unit")]
        return int(amount * multiplier)
    return None


def _affected_subject_count(folded: str) -> int | None:
    match = re.search(
        r"(?P<count>\d{1,3}(?:[.\s]\d{3})+|\d+)\s*"
        r"(?:khach hang|nguoi dung|ca nhan|chu the du lieu)\b",
        folded,
    )
    return _number(match.group("count")) if match else None


@dataclass(frozen=True)
class CanonicalLegalEvent:
    actions: tuple[str, ...] = ()
    objects: tuple[str, ...] = ()
    purposes: tuple[str, ...] = ()
    conditions: tuple[str, ...] = ()
    actor_roles: tuple[str, ...] = ()
    data_fields: tuple[str, ...] = ()
    affected_subject_count: int | None = None
    sale_amount_vnd: int | None = None
    recipient_type: str | None = None
    received_payment: bool = False
    required_domains: tuple[str, ...] = ()
    supporting_domains: tuple[str, ...] = ()
    liability_intent: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "actions": list(self.actions),
            "objects": list(self.objects),
            "purposes": list(self.purposes),
            "conditions": list(self.conditions),
            "actor_roles": list(self.actor_roles),
            "data_fields": list(self.data_fields),
            "affected_subject_count": self.affected_subject_count,
            "sale_amount_vnd": self.sale_amount_vnd,
            "recipient_type": self.recipient_type,
            "received_payment": self.received_payment,
            "required_domains": list(self.required_domains),
            "supporting_domains": list(self.supporting_domains),
            "liability_intent": self.liability_intent,
        }


def extract_canonical_legal_event(query: str) -> CanonicalLegalEvent:
    """Extract one shared event from the original user query without an LLM."""

    folded = fold_legal_text(query)
    actions: list[str] = []
    objects: list[str] = []
    purposes: list[str] = []
    conditions: list[str] = []
    actor_roles: list[str] = []
    data_fields: list[str] = []
    required_domains: list[str] = []
    supporting_domains: list[str] = []

    customer_context = bool(
        re.search(
            r"(?:khach hang|nguoi dung|chu the du lieu|danh sach khach|"
            r"file (?:data|du lieu|khach)|du lieu khach|thong tin khach)",
            folded,
        )
    )
    field_patterns = (
        ("full_name", r"\b(?:ho va ten|ho ten|ten khach hang)\b"),
        ("phone_number", r"\b(?:so dien thoai|dien thoai)\b"),
        ("email", r"\bemail\b"),
        ("purchase_history", r"\b(?:lich su mua hang|lich su giao dich)\b"),
        ("address", r"\b(?:dia chi|noi o)\b"),
        ("identifier", r"\b(?:cccd|cmnd|so dinh danh)\b"),
    )
    for key, pattern in field_patterns:
        if re.search(pattern, folded):
            data_fields.append(key)

    explicit_personal_data = bool(
        re.search(
            r"\b(?:du lieu ca nhan|thong tin ca nhan|chu the du lieu)\b",
            folded,
        )
    )
    personal_data = explicit_personal_data or bool(
        customer_context
        and (
            data_fields
            or re.search(r"\b(?:file|data|du lieu|danh sach|co so du lieu)\b", folded)
        )
    )
    if personal_data:
        objects.append("personal_data")
        required_domains.append("personal_data")

    sale = bool(
        re.search(
            r"\b(?:ban|rao ban|mua ban|doi lay tien)\b.{0,45}"
            r"(?:du lieu|data|file|danh sach|thong tin|khach hang)",
            folded,
        )
        or re.search(
            r"(?:du lieu|data|file|danh sach|thong tin|khach hang).{0,45}"
            r"\b(?:ban|rao ban|mua ban|doi lay tien)\b",
            folded,
        )
    )
    transfer = bool(
        re.search(
            r"\b(?:chuyen|chuyen giao|chia se|cung cap|gui|dua)\b.{0,45}"
            r"(?:du lieu|data|file|danh sach|thong tin|khach hang)",
            folded,
        )
        or re.search(
            r"(?:du lieu|data|file|danh sach|thong tin|khach hang).{0,45}"
            r"\b(?:cho|sang)\s+(?:ben|doi tac|cong ty|nguoi)\b",
            folded,
        )
    )
    retention = bool(
        re.search(r"\b(?:giu lai|luu giu|mang theo|sao chep)\b", folded)
        and customer_context
        and re.search(
            r"\b(?:nghi viec|sau khi nghi|nhan vien cu|cham dut cong viec|"
            r"cham dut hop dong)\b",
            folded,
        )
    )
    if personal_data and sale:
        actions.extend(("sell_personal_data", "share_personal_data"))
        purposes.append("commercial_gain")
    elif personal_data and transfer:
        actions.append("share_personal_data")
    if personal_data and retention:
        actions.append("retain_personal_data")

    former_employee = bool(
        re.search(r"\b(?:nhan vien cu|cuu nhan vien|nguoi lao dong cu)\b", folded)
        or re.search(r"\b(?:nhan vien|nguoi lao dong)\b.{0,30}\bnghi viec\b", folded)
    )
    if former_employee:
        actor_roles.append("former_employee")

    explicit_no_consent = bool(
        re.search(
            r"\b(?:khong|chua)\b.{0,20}"
            r"(?:dong y|xin phep|cho phep|uy quyen|duoc phep)",
            folded,
        )
        or re.search(
            r"\b(?:khong duoc cong ty|khong duoc khach hang|"
            r"khong duoc chu the du lieu).{0,20}(?:cho phep|dong y|uy quyen)",
            folded,
        )
    )
    if explicit_no_consent:
        conditions.extend(("without_consent", "without_authorization"))

    advertising = bool(re.search(r"\b(?:quang cao|tiep thi|marketing)\b", folded))
    if advertising:
        purposes.append("advertising")
        supporting_domains.append("advertising")
    if customer_context:
        supporting_domains.append("consumer_protection")

    recipient_type = None
    if re.search(r"(?:cong ty|doi tac).{0,25}(?:quang cao|tiep thi|marketing)", folded):
        recipient_type = "advertising_company"
    elif re.search(r"\bdoi tac\b", folded):
        recipient_type = "business_partner"
    elif re.search(r"\bben thu ba\b", folded):
        recipient_type = "third_party"

    liability_intent = bool(
        re.search(
            r"\b(?:trach nhiem|dan su|hanh chinh|hinh su|che tai|"
            r"xu ly|hau qua phap ly|vi pham gi|van de phap ly)\b",
            folded,
        )
    )
    received_payment = bool(
        re.search(r"\b(?:nhan|thu)\s+(?:duoc\s+)?(?:mot\s+)?khoan tien\b", folded)
        or _money_vnd(folded) is not None
    )
    if received_payment and "commercial_gain" not in purposes:
        purposes.append("commercial_gain")

    return CanonicalLegalEvent(
        actions=_unique(actions),
        objects=_unique(objects),
        purposes=_unique(purposes),
        conditions=_unique(conditions),
        actor_roles=_unique(actor_roles),
        data_fields=_unique(data_fields),
        affected_subject_count=_affected_subject_count(folded),
        sale_amount_vnd=_money_vnd(folded) if sale else None,
        recipient_type=recipient_type,
        received_payment=received_payment,
        required_domains=_unique(required_domains),
        supporting_domains=_unique(supporting_domains),
        liability_intent=liability_intent,
    )
