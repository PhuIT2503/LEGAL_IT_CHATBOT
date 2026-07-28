"""Deterministic retrieval coverage contracts for high-value legal events."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Mapping

from src.retrieval.legal_event import CanonicalLegalEvent, fold_legal_text


_PERSONAL_DATA_SALE_PROHIBITION = re.compile(
    r"(?:nghiem cam.{0,100})?(?:mua\s*,?\s*ban|mua ban|ban)"
    r".{0,45}du lieu ca nhan|du lieu ca nhan.{0,45}(?:mua\s*,?\s*ban|mua ban)",
)
_PERSONAL_DATA_CONSEQUENCE = re.compile(
    r"(?:xu ly ky luat.{0,180})?xu phat hanh chinh.{0,180}"
    r"truy cuu trach nhiem hinh su.{0,180}boi thuong",
)


@dataclass(frozen=True)
class RetrievalContractAudit:
    required_roles: tuple[str, ...]
    satisfied_roles: tuple[str, ...]
    missing_roles: tuple[str, ...]

    @property
    def requires_rescue(self) -> bool:
        return bool(self.missing_roles)

    def as_dict(self) -> dict[str, object]:
        return {
            "required_roles": list(self.required_roles),
            "satisfied_roles": list(self.satisfied_roles),
            "missing_roles": list(self.missing_roles),
            "requires_rescue": self.requires_rescue,
        }


def retrieval_contract_roles(
    event: CanonicalLegalEvent,
    text: str,
) -> tuple[str, ...]:
    """Label only provisions whose own text directly satisfies a contract."""

    if "sell_personal_data" not in event.actions:
        return ()
    folded = fold_legal_text(text)
    roles: list[str] = []
    if (
        _PERSONAL_DATA_SALE_PROHIBITION.search(folded)
        and not (
            "muc phat tien toi da" in folded
            and "nghiem cam" not in folded
        )
    ):
        roles.append("personal_data_sale_prohibition")
    if event.liability_intent and _PERSONAL_DATA_CONSEQUENCE.search(folded):
        roles.append("personal_data_consequence")
    return tuple(roles)


def assess_retrieval_contract(
    event: CanonicalLegalEvent,
    records: Iterable[Mapping[str, object]],
) -> RetrievalContractAudit:
    required: list[str] = []
    if "sell_personal_data" in event.actions:
        required.append("personal_data_sale_prohibition")
        if event.liability_intent:
            required.append("personal_data_consequence")

    satisfied: list[str] = []
    for record in records:
        text = "\n".join(
            (
                str(record.get("source") or ""),
                str(record.get("text") or ""),
            )
        )
        for role in retrieval_contract_roles(event, text):
            if role not in satisfied:
                satisfied.append(role)
    return RetrievalContractAudit(
        required_roles=tuple(required),
        satisfied_roles=tuple(role for role in required if role in satisfied),
        missing_roles=tuple(role for role in required if role not in satisfied),
    )


def build_retrieval_rescue_query(
    event: CanonicalLegalEvent,
    missing_roles: Iterable[str],
) -> str:
    """Build one bounded query; caller performs at most one extra search."""

    roles = set(missing_roles)
    terms: list[str] = []
    if "personal_data_sale_prohibition" in roles:
        terms.extend(
            (
                "nghiêm cấm mua bán dữ liệu cá nhân",
                "mua, bán dữ liệu cá nhân",
            )
        )
    if "personal_data_consequence" in roles and event.liability_intent:
        terms.extend(
            (
                "vi phạm pháp luật về bảo vệ dữ liệu cá nhân",
                "tùy theo tính chất mức độ hậu quả của hành vi vi phạm",
                "xử phạt hành chính truy cứu trách nhiệm hình sự nếu gây thiệt hại phải bồi thường thiệt hại",
            )
        )
    return "; ".join(dict.fromkeys(terms))


def annotate_retrieval_contract_records(
    event: CanonicalLegalEvent,
    records: Iterable[dict],
) -> list[dict]:
    annotated: list[dict] = []
    for record in records:
        item = dict(record)
        roles = retrieval_contract_roles(
            event,
            f"{item.get('source', '')}\n{item.get('text', '')}",
        )
        if roles:
            item["retrieval_contract_roles"] = list(roles)
        annotated.append(item)
    return annotated
