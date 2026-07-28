"""Deterministic fact preservation for long-form legal scenarios.

The extractor records only facts explicitly stated by the user.  Legal
interpretations are kept in a separate inference bucket and unresolved legal
elements in a third bucket, so downstream stages cannot silently turn an
inference into a stated fact.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Any, Iterable, Mapping

from src.retrieval.legal_event import (
    CanonicalLegalEvent,
    extract_canonical_legal_event,
)


def _fold(value: Any) -> str:
    text = unicodedata.normalize("NFD", str(value or "").casefold())
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return re.sub(r"\s+", " ", text.replace("đ", "d")).strip()


def _normalize_question(question: str) -> str:
    return " ".join(unicodedata.normalize("NFC", str(question or "")).split())


def _question_sections(question: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFC", str(question or "")).strip()
    return tuple(
        section.strip()
        for section in re.split(r"(?<=[.!?])\s+|\n+", normalized)
        if section.strip()
    )


_AMOUNT = r"(?P<amount>\d{1,3}(?:[.\s]\d{3})+|\d+)\s*(?:dong|vnd)"


def _amount(pattern: str, folded: str) -> int | None:
    match = re.search(pattern.replace("{amount}", _AMOUNT), folded)
    if not match:
        return None
    digits = re.sub(r"\D", "", match.group("amount"))
    return int(digits) if digits else None


@dataclass(frozen=True)
class LegalScenarioFactState:
    normalized_question: str
    question_sections: tuple[str, ...]
    stated_facts: Mapping[str, Any]
    supported_inferences: Mapping[str, Any]
    unknown_legal_elements: Mapping[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "normalized_question": self.normalized_question,
            "question_sections": list(self.question_sections),
            "stated_facts": dict(self.stated_facts),
            "supported_inferences": dict(self.supported_inferences),
            "unknown_legal_elements": dict(self.unknown_legal_elements),
        }


def extract_legal_scenario_facts(
    question: str,
    *,
    event: CanonicalLegalEvent | None = None,
) -> LegalScenarioFactState:
    """Extract explicit scenario facts without using an LLM."""

    normalized = _normalize_question(question)
    folded = _fold(normalized)
    event = event or extract_canonical_legal_event(question)

    requested_amount = _amount(
        r"(?:yeu cau|de nghi|lenh)\s+(?:ke toan\s+)?(?:chuyen|thanh toan)"
        r"(?:\s+(?:tien|du|so tien))?\s+{amount}",
        folded,
    )
    transferred_amount = _amount(
        r"(?:ke toan|nguoi nhan lenh|cong ty).{0,30}"
        r"\bda\s+(?:chuyen|thanh toan)"
        r"(?:\s+(?:du|tien|so tien))*\s+{amount}",
        folded,
    ) or _amount(
        r"\bda\s+(?:chuyen|thanh toan)"
        r"(?:\s+(?:du|tien|so tien))*\s+{amount}",
        folded,
    )
    frozen_amount = _amount(
        r"(?:phong toa|dong bang|tam giu)(?:\s+duoc)?\s+{amount}",
        folded,
    )
    onward_amount = _amount(
        r"{amount}\s+(?:da\s+)?duoc\s+(?:chuyen tiep|chuyen sang)",
        folded,
    ) or _amount(
        r"(?:chuyen tiep|chuyen sang)(?:\s+duoc)?\s+{amount}",
        folded,
    )

    used_ai = bool(re.search(r"\b(?:ai|tri tue nhan tao)\b", folded))
    voice_impersonation = bool(
        re.search(
            r"(?:gia giong|gia mao giong noi|giong noi gia mao|giong noi gia)",
            folded,
        )
    )
    role_match = re.search(
        r"(?:gia giong|gia mao giong noi)\s+(?P<role>"
        r"giam doc|lanh dao|tong giam doc|truong phong|nguoi quan ly)",
        folded,
    )
    called_accountant = bool(
        re.search(r"(?:goi|lien lac)(?:\s+(?:cho|voi))?\s+ke toan", folded)
    )
    requested_transfer = bool(
        requested_amount is not None
        or re.search(r"(?:yeu cau|de nghi|lenh).{0,35}(?:chuyen|thanh toan)", folded)
    )
    transfer_executed = bool(
        transferred_amount is not None
        or re.search(
            r"(?:ke toan|nguoi nhan lenh).{0,25}\bda\s+"
            r"(?:chuyen|thanh toan)(?:\s+(?:tien|du|xong))?",
            folded,
        )
    )
    actor_claimed_experiment = bool(
        re.search(
            r"(?:khai|noi|cho rang).{0,25}(?:chi\s+)?(?:thu nghiem|test)"
            r"(?:\s+cong nghe)?",
            folded,
        )
    )
    actor_claimed_money_not_used = bool(
        re.search(
            r"(?:khai|noi|cho rang)?.{0,30}(?:chua|khong)"
            r"\s+(?:su dung|dung).{0,20}(?:so )?tien",
            folded,
        )
    )
    act_was_executed = transfer_executed

    stated: dict[str, Any] = {}
    role = role_match.group("role") if role_match else None
    role = {
        "giam doc": "giám đốc",
        "tong giam doc": "tổng giám đốc",
        "lanh dao": "lãnh đạo",
        "truong phong": "trưởng phòng",
        "nguoi quan ly": "người quản lý",
    }.get(role, role)
    candidates = {
        "used_ai": used_ai,
        "voice_impersonation": voice_impersonation,
        "impersonated_person_role": role,
        "called_accountant": called_accountant,
        "requested_transfer": requested_transfer,
        "requested_amount_vnd": requested_amount,
        "transfer_executed": transfer_executed,
        "transferred_amount_vnd": transferred_amount,
        "frozen_amount_vnd": frozen_amount,
        "onward_transferred_amount_vnd": onward_amount,
        "act_was_executed": act_was_executed,
        "actor_claimed_experiment": actor_claimed_experiment,
        "actor_claimed_money_not_used": actor_claimed_money_not_used,
    }
    for key, value in candidates.items():
        if value is not None and value is not False:
            stated[key] = value

    event_facts = {
        "former_employee": "former_employee" in event.actor_roles,
        "retained_customer_data": "retain_personal_data" in event.actions,
        "sold_personal_data": "sell_personal_data" in event.actions,
        "shared_personal_data": "share_personal_data" in event.actions,
        "affected_subject_count": event.affected_subject_count,
        "personal_data_fields": list(event.data_fields) or None,
        "recipient_type": event.recipient_type,
        "data_sale_proceeds_vnd": event.sale_amount_vnd,
        "received_payment": event.received_payment,
    }
    for key, value in event_facts.items():
        if value is not None and value is not False:
            stated[key] = value
    if "sell_personal_data" in event.actions or "share_personal_data" in event.actions:
        stated["act_was_executed"] = True
    if "without_authorization" in event.conditions:
        stated["company_authorization"] = False
    if "without_consent" in event.conditions:
        stated["data_subject_consent"] = False

    induced_transfer = bool(
        voice_impersonation and requested_transfer and transfer_executed
    )
    possible_appropriation_intent = bool(
        voice_impersonation and requested_transfer and transferred_amount
    )
    possible_financial_loss = bool(
        transferred_amount
        and (
            frozen_amount is None
            or transferred_amount > frozen_amount
            or bool(onward_amount)
        )
    )
    supported_inferences = {
        key: value
        for key, value in {
            "possible_appropriation_intent": possible_appropriation_intent,
            "possible_financial_loss": possible_financial_loss,
            "impersonation_used_to_induce_transfer": induced_transfer,
        }.items()
        if value
    }

    unknown: dict[str, str] = {}
    if transfer_executed:
        unknown["final_unrecoverable_loss"] = (
            "Chưa xác định số tiền hoặc tổn thất cuối cùng không thể thu hồi."
        )
    if possible_appropriation_intent:
        unknown["full_appropriation_intent_evidence"] = (
            "Chưa có đầy đủ chứng cứ để kết luận chắc chắn ý định chiếm đoạt."
        )
    if re.search(r"\b(?:vai tro|chu the khac|dong pham)\b", folded):
        unknown["other_actor_roles"] = (
            "Chưa xác định vai trò pháp lý của các chủ thể khác nếu có."
        )
    if re.search(r"\b(?:trach nhiem|dan su|hanh chinh|hinh su)\b", folded):
        unknown["liability_type"] = (
            "Chưa đủ căn cứ xác định đầy đủ loại trách nhiệm pháp lý."
        )
    if re.search(r"\b(?:che tai|muc phat|bien phap|xu ly)\b", folded):
        unknown["complete_sanction_basis"] = (
            "Chưa có đầy đủ căn cứ trực tiếp về chế tài và biện pháp xử lý."
        )
    if event.actions and "personal_data" in event.objects:
        if "without_authorization" not in event.conditions:
            unknown["company_authorization"] = (
                "Chưa xác định người thực hiện có được doanh nghiệp cho phép hay không."
            )
        if "without_consent" not in event.conditions:
            unknown["data_subject_consent"] = (
                "Chưa xác định có sự đồng ý hoặc căn cứ xử lý dữ liệu phù hợp hay không."
            )
        unknown["event_time"] = (
            "Chưa xác định thời điểm hành vi để đối chiếu hiệu lực văn bản pháp luật."
        )
    return LegalScenarioFactState(
        normalized_question=normalized,
        question_sections=_question_sections(question),
        stated_facts=stated,
        supported_inferences=supported_inferences,
        unknown_legal_elements=unknown,
    )


_MISSING_ALIASES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "money_transfer_status",
        (
            r"(?:tien|tai san).{0,30}(?:da\s+)?(?:duoc\s+)?chuyen.{0,20}(?:hay chua|chua)",
            r"(?:da\s+)?chuyen\s+tien\s+(?:hay|chua)",
        ),
        ("transfer_executed",),
    ),
    (
        "transferred_amount",
        (
            r"(?:so tien|gia tri).{0,30}(?:da\s+)?chuyen",
            r"(?:chuyen|thanh toan).{0,20}bao nhieu",
        ),
        ("transferred_amount_vnd",),
    ),
    (
        "preparation_or_execution",
        (
            r"(?:giai doan\s+)?chuan bi.{0,25}(?:thuc hien|thuc te)",
            r"(?:da\s+)?duoc\s+thuc hien.{0,20}(?:hay|chua)",
            r"(?:thuc hien|thuc te).{0,25}chuan bi",
        ),
        ("act_was_executed",),
    ),
    (
        "ai_usage",
        (r"(?:co|da).{0,15}(?:su dung|dung).{0,12}(?:ai|tri tue nhan tao)",),
        ("used_ai",),
    ),
    (
        "voice_impersonation",
        (r"(?:co|da).{0,15}(?:gia giong|gia mao giong noi)",),
        ("voice_impersonation",),
    ),
)


def missing_fact_key(value: str) -> str | None:
    folded = _fold(value)
    for key, patterns, _ in _MISSING_ALIASES:
        if any(re.search(pattern, folded) for pattern in patterns):
            return key
    return None


def fact_is_answered_by(value: str, fact_state: Mapping[str, Any]) -> bool:
    key = missing_fact_key(value)
    if not key:
        return False
    stated = fact_state.get("stated_facts") or {}
    required_fields = next(
        fields for alias, _, fields in _MISSING_ALIASES if alias == key
    )
    return any(field in stated and stated[field] is not None for field in required_fields)


def filter_answered_missing_facts(
    values: Iterable[str],
    fact_state: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    """Remove missing questions already answered by explicit stated facts."""

    kept: list[str] = []
    keys: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = " ".join(str(value or "").split()).strip()
        if not text:
            continue
        key = missing_fact_key(text)
        if fact_is_answered_by(text, fact_state):
            continue
        identity = _fold(text)
        if identity in seen:
            continue
        seen.add(identity)
        kept.append(text)
        if key and key not in keys:
            keys.append(key)
    return kept, keys
