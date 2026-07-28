"""Deterministic provision-element coverage used after LLM applicability.

The rules are derived from the retrieved provision text, never from a document
or article identifier.  Their purpose is to prevent broad keyword overlap from
overriding a required factual element that the scenario does not contain.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Any, Mapping


MATCH = "MATCH"
PARTIAL_MATCH = "PARTIAL_MATCH"
NOT_APPLICABLE = "NOT_APPLICABLE"


def _fold(value: Any) -> str:
    text = unicodedata.normalize("NFD", str(value or "").casefold())
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return re.sub(r"\s+", " ", text.replace("đ", "d")).strip()


@dataclass(frozen=True)
class ElementCoverage:
    required_elements: tuple[str, ...]
    matched_elements: tuple[str, ...]
    missing_required_elements: tuple[str, ...]
    applicability: str
    reason: str


def evaluate_provision_element_coverage(
    provision_text: str,
    scenario_fact_state: Mapping[str, Any] | None,
) -> ElementCoverage:
    """Compare provision-derived elements with explicit stated facts."""

    source = _fold(provision_text)
    stated = dict((scenario_fact_state or {}).get("stated_facts") or {})
    required: list[str] = []
    matched: list[str] = []
    critical: set[str] = set()

    def add(key: str, is_matched: bool, *, is_critical: bool = False) -> None:
        if key in required:
            return
        required.append(key)
        if is_matched:
            matched.append(key)
        if is_critical:
            critical.add(key)

    if re.search(r"\b(?:tri tue nhan tao|cong nghe moi)\b", source):
        add("ai_or_new_technology", bool(stated.get("used_ai")))
    if (
        re.search(r"\bgia mao\b", source)
        and re.search(r"\b(?:video|hinh anh|giong noi)\b", source)
    ):
        add(
            "impersonated_video_image_or_voice",
            bool(stated.get("voice_impersonation")),
        )
    if re.search(r"\b(?:phan mem|ung dung cong nghe thong tin)\b", source):
        add(
            "software_or_it_application",
            bool(stated.get("used_ai")),
        )
    if (
        re.search(r"\bgia mao\b", source)
        and re.search(r"\b(?:thong tin|hinh anh)\b", source)
    ):
        add(
            "falsified_information_or_image",
            bool(stated.get("voice_impersonation")),
        )

    contract_cluster = bool(
        re.search(r"\bgiao ket hop dong\b", source)
        and re.search(
            r"\b(?:hop dong theo mau|dieu kien giao dich chung)\b",
            source,
        )
    )
    query_folded = _fold(
        (scenario_fact_state or {}).get("normalized_question") or ""
    )
    has_contract_fact = bool(
        re.search(
            r"\b(?:giao ket hop dong|ky hop dong|hop dong theo mau|"
            r"dieu kien giao dich chung)\b",
            query_folded,
        )
    )
    if re.search(r"\bgiao ket hop dong\b", source):
        add(
            "contract_conclusion",
            has_contract_fact,
            is_critical=contract_cluster,
        )
    if re.search(
        r"\b(?:hop dong theo mau|dieu kien giao dich chung)\b",
        source,
    ):
        add(
            "standard_form_contract_or_general_terms",
            has_contract_fact,
            is_critical=contract_cluster,
        )

    missing = tuple(key for key in required if key not in matched)
    missing_critical = [key for key in missing if key in critical]
    if missing_critical:
        applicability = NOT_APPLICABLE
        reason = (
            "Provision yêu cầu yếu tố giao kết/hợp đồng nhưng tình huống không "
            "nêu dữ kiện tương ứng: " + ", ".join(missing_critical) + "."
        )
    elif required and not missing:
        applicability = MATCH
        reason = "Các yếu tố fact có thể kiểm tra của provision đều được tình huống đáp ứng."
    elif matched:
        applicability = PARTIAL_MATCH
        reason = "Tình huống chỉ đáp ứng một phần yếu tố fact của provision."
    elif required:
        applicability = PARTIAL_MATCH
        reason = (
            "Chưa đủ rule deterministic để phủ nhận provision; các yếu tố fact "
            "đã nhận diện hiện chưa được tình huống đáp ứng."
        )
    else:
        # Không suy đoán cấu thành cho provision mà rule chưa nhận diện.
        applicability = PARTIAL_MATCH
        reason = "Chưa có rule deterministic để xác định đầy đủ cấu thành từ đoạn nguồn."

    return ElementCoverage(
        required_elements=tuple(required),
        matched_elements=tuple(matched),
        missing_required_elements=missing,
        applicability=applicability,
        reason=reason,
    )
