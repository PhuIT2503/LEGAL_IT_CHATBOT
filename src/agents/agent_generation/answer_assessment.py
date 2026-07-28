"""Single deterministic assessment used by every user-visible answer section.

This module is part of Generation formatting.  It reuses the Behavior Card,
Applicability decisions and final grounded sources; it does not retrieve,
rerank, call an LLM or create legal citations.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from typing import Any, Iterable, Mapping

from src.agents.common.grounded_validation import build_grounded_sources
from src.agents.common.legal_scenario_facts import (
    extract_legal_scenario_facts,
    filter_answered_missing_facts,
    missing_fact_key,
)
from src.retrieval.legal_behaviors import (
    ACTION_DEFINITIONS,
    CONDITION_DEFINITIONS,
    OBJECT_DEFINITIONS,
    PURPOSE_DEFINITIONS,
)
from src.retrieval.legal_event import extract_canonical_legal_event


logger = logging.getLogger(__name__)

LIKELY_VIOLATION = "LIKELY_VIOLATION"
PARTIAL_MATCH = "PARTIAL_MATCH"
INSUFFICIENT_FACTS = "INSUFFICIENT_FACTS"
NO_MATCH = "NO_MATCH"

_KEPT = {"KEEP", "WEAK_KEEP"}
_POSITIVE = {"MATCH", "PARTIAL_MATCH"}
_DEFINITIONS = {
    definition.key: definition.description
    for definition in (
        *ACTION_DEFINITIONS,
        *OBJECT_DEFINITIONS,
        *PURPOSE_DEFINITIONS,
        *CONDITION_DEFINITIONS,
    )
}
_FACT_BY_BEHAVIOR = {
    "create_ai_deepfake": "Có sử dụng AI hoặc công nghệ để tạo nội dung giả mạo.",
    "use_person_likeness": "Có sử dụng hình ảnh, khuôn mặt hoặc giọng nói của người khác.",
    "unauthorized_access": "Có hành vi truy cập hoặc xâm nhập hệ thống khi chưa được phép.",
    "exploit_vulnerability": "Có hành vi khai thác điểm yếu hoặc lỗ hổng bảo mật.",
    "extract_or_download_data": "Có hành vi lấy, trích xuất hoặc tải dữ liệu.",
    "collect_personal_data": "Có hành vi thu thập dữ liệu cá nhân.",
    "share_personal_data": "Có hành vi chia sẻ hoặc cung cấp dữ liệu cá nhân cho bên khác.",
    "sell_personal_data": "Có hành vi bán dữ liệu cá nhân để nhận tiền hoặc lợi ích.",
    "retain_personal_data": "Có hành vi tiếp tục lưu giữ dữ liệu cá nhân sau khi hết thẩm quyền.",
    "process_personal_data_without_consent": "Có hành vi xử lý dữ liệu cá nhân khi chưa có sự đồng ý.",
    "publish_false_content": "Có hành vi đăng hoặc phát tán thông tin giả, sai sự thật.",
    "misleading_advertising": "Có nội dung quảng cáo sai sự thật hoặc gây nhầm lẫn.",
    "use_copyrighted_work": "Có hành vi sao chép hoặc sử dụng tác phẩm, bản ghi được bảo hộ.",
    "without_consent": "Tình huống thể hiện chưa có sự đồng ý của chủ thể liên quan.",
    "without_authorization": "Tình huống thể hiện chủ thể chưa được cho phép hoặc không có quyền.",
    "public_distribution": "Nội dung đã được hoặc dự kiến được đăng tải, phát tán công khai.",
    "advertising": "Hành vi được thực hiện cho mục đích quảng cáo hoặc tiếp thị.",
    "commercial_gain": "Hành vi có mục đích thương mại hoặc thu lợi.",
    "data_exfiltration": "Mục đích của hành vi là lấy hoặc chiếm đoạt dữ liệu.",
}


def _fold(value: Any) -> str:
    text = unicodedata.normalize("NFD", str(value or "").casefold())
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return " ".join(text.replace("đ", "d").split())


def _clean(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFC", str(value or "")).split()).strip()


def _unique(values: Iterable[str], *, limit: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = _clean(value)
        key = _fold(item)
        if not item or key in seen:
            continue
        seen.add(key)
        result.append(item)
        if len(result) >= limit:
            break
    return result


def _profile_keys(profile: Mapping[str, Any] | None) -> tuple[set[str], set[str]]:
    profile = profile or {}
    actions = {str(key) for key in (profile.get("actions") or ()) if str(key)}
    all_keys = {
        str(key)
        for group in ("actions", "objects", "purposes", "conditions")
        for key in (profile.get(group) or ())
        if str(key)
    }
    return actions, all_keys


def _decision_matches(decision: Mapping[str, Any]) -> list[tuple[str, str]]:
    matches: list[tuple[str, str]] = []
    for item in decision.get("behavior_matches") or ():
        if isinstance(item, Mapping):
            key = _clean(item.get("behavior_key"))
            match = _clean(item.get("match")).upper()
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            key = _clean(item[0])
            match = _clean(item[1]).upper()
        else:
            continue
        if key:
            matches.append((key, match))
    return matches


def _meaningful_missing(value: Any) -> bool:
    folded = _fold(value)
    if not folded:
        return False
    if any(
        re.search(pattern, folded)
        for pattern in (
            r"^khong con dieu kien thieu\b",
            r"^khong con thieu (?:dieu kien|du kien|tinh tiet)\b",
            r"^khong co (?:dieu kien|du kien|tinh tiet).{0,30}(?:con thieu|can lam ro)\b",
            r"^chua xac dinh duoc dieu kien ap dung\b",
            r"^hanh vi khong lien quan\b",
        )
    ):
        return False
    # Applicability đôi khi dùng field này để giải thích vì sao một candidate
    # phụ không phù hợp. Chỉ câu thực sự biểu thị khoảng trống thông tin mới
    # được phép đi vào phần "Còn cần làm rõ".
    return bool(
        re.search(
            r"\b(?:can|chua|thieu|khong ro|xac minh|lam ro)\b|"
            r"\bkhong co (?:thong tin|du kien|tinh tiet)\b",
            folded,
        )
    )


def _query_facts(
    query: str,
    positive_keys: set[str],
) -> list[str]:
    folded = _fold(query)
    facts: list[str] = []
    if "create_ai_deepfake" in positive_keys:
        if re.search(r"\b(?:ai|tri tue nhan tao)\b", folded):
            facts.append("Có sử dụng trí tuệ nhân tạo.")
        if re.search(r"(?:gia giong|gia mao giong noi|giong noi gia)", folded):
            facts.append("Có giả mạo giọng nói của người khác.")
        elif re.search(r"(?:gia mao (?:video|hinh anh)|deepfake)", folded):
            facts.append("Có tạo hoặc sử dụng video, hình ảnh giả mạo.")
    if (
        "create_ai_deepfake" in positive_keys
        and re.search(r"(?:giam doc|lanh dao|nguoi quan ly)", folded)
        and re.search(r"(?:ke toan|chuyen tien|thanh toan)", folded)
    ):
        facts.append(
            "Việc giả mạo được dùng để mạo danh người có thẩm quyền và yêu cầu chuyển tiền."
        )
    return facts


def _query_supported_behavior_keys(query: str, source_bodies: Iterable[str]) -> set[str]:
    """Map explicit question facts only when a retained SOURCE confirms them."""

    query_folded = _fold(query)
    source_folded = _fold(" ".join(source_bodies))
    supported: set[str] = set()
    query_has_ai = bool(re.search(r"\b(?:ai|tri tue nhan tao)\b", query_folded))
    query_has_fake_likeness = bool(
        re.search(
            r"(?:deepfake|gia giong|gia mao (?:giong noi|video|hinh anh)|"
            r"(?:video|hinh anh|giong noi) gia mao)",
            query_folded,
        )
    )
    source_covers_ai_impersonation = bool(
        re.search(r"(?:tri tue nhan tao|cong nghe moi)", source_folded)
        and re.search(
            r"gia mao.{0,40}(?:video|hinh anh|giong noi)|"
            r"(?:video|hinh anh|giong noi).{0,40}gia mao",
            source_folded,
        )
    )
    if query_has_ai and query_has_fake_likeness and source_covers_ai_impersonation:
        supported.add("create_ai_deepfake")
    event = extract_canonical_legal_event(query)
    if (
        "sell_personal_data" in event.actions
        and re.search(
            r"(?:mua\s*,?\s*ban|mua ban|ban).{0,45}du lieu ca nhan|"
            r"du lieu ca nhan.{0,45}(?:mua\s*,?\s*ban|mua ban)",
            source_folded,
        )
    ):
        supported.add("sell_personal_data")
    if (
        "share_personal_data" in event.actions
        and re.search(
            r"(?:chia se|chuyen giao|cung cap|tiet lo).{0,45}"
            r"(?:du lieu ca nhan|thong tin cua nguoi tieu dung)",
            source_folded,
        )
    ):
        supported.add("share_personal_data")
    return supported


def _scenario_missing_facts(fact_state: Mapping[str, Any]) -> list[str]:
    stated = fact_state.get("stated_facts") or {}
    missing = [
        _clean(description)
        for description in (
            fact_state.get("unknown_legal_elements") or {}
        ).values()
        if _clean(description)
    ]
    if stated.get("requested_transfer") and not stated.get("transfer_executed"):
        missing.extend(
            (
                "Tiền hoặc tài sản đã được chuyển hay chưa.",
                "Người thực hiện có mục đích chiếm đoạt tài sản hay không.",
                "Đã phát sinh thiệt hại thực tế hay chưa.",
                "Hành vi mới ở giai đoạn chuẩn bị hay đã được thực hiện.",
            )
        )
    return missing


def _vnd(value: Any) -> str:
    try:
        return f"{int(value):,}".replace(",", ".") + " đồng"
    except (TypeError, ValueError):
        return ""


def _stated_fact_summaries(fact_state: Mapping[str, Any]) -> list[str]:
    stated = fact_state.get("stated_facts") or {}
    summaries: list[str] = []
    if stated.get("called_accountant") and stated.get("requested_transfer"):
        requested = _vnd(stated.get("requested_amount_vnd"))
        summaries.append(
            "A đã gọi kế toán và yêu cầu chuyển"
            + (f" {requested}" if requested else " tiền")
            + "."
        )
    if stated.get("transfer_executed"):
        transferred = _vnd(stated.get("transferred_amount_vnd"))
        summaries.append(
            "Theo tình huống, kế toán đã thực hiện chuyển"
            + (f" {transferred}" if transferred else " tiền")
            + "."
        )
    if stated.get("frozen_amount_vnd"):
        summaries.append(
            f"Ngân hàng đã phong tỏa {_vnd(stated['frozen_amount_vnd'])}."
        )
    if stated.get("onward_transferred_amount_vnd"):
        summaries.append(
            f"{_vnd(stated['onward_transferred_amount_vnd'])} đã được chuyển tiếp sang tài khoản khác."
        )
    if stated.get("actor_claimed_experiment"):
        summaries.append("A khai rằng hành vi chỉ nhằm thử nghiệm công nghệ.")
    if stated.get("actor_claimed_money_not_used"):
        summaries.append("A khai rằng chưa sử dụng số tiền.")
    if stated.get("former_employee") and stated.get("retained_customer_data"):
        summaries.append(
            "Người thực hiện là nhân viên cũ và vẫn giữ dữ liệu khách hàng sau khi nghỉ việc."
        )
    affected = stated.get("affected_subject_count")
    fields = stated.get("personal_data_fields") or []
    field_labels = {
        "full_name": "họ tên",
        "phone_number": "số điện thoại",
        "email": "email",
        "purchase_history": "lịch sử mua hàng",
        "address": "địa chỉ",
        "identifier": "số định danh",
    }
    if affected:
        detail = ", ".join(field_labels.get(field, str(field)) for field in fields)
        summaries.append(
            f"Dữ liệu liên quan đến {int(affected):,} khách hàng".replace(",", ".")
            + (f", gồm {detail}." if detail else ".")
        )
    elif fields:
        summaries.append(
            "File khách hàng có " + ", ".join(
                field_labels.get(field, str(field)) for field in fields
            ) + "."
        )
    if stated.get("sold_personal_data"):
        amount = _vnd(stated.get("data_sale_proceeds_vnd"))
        recipient = {
            "advertising_company": "một công ty quảng cáo",
            "business_partner": "một đối tác kinh doanh",
            "third_party": "một bên thứ ba",
        }.get(stated.get("recipient_type"), "một bên khác")
        summaries.append(
            f"Dữ liệu đã được bán/chuyển cho {recipient}"
            + (f" để nhận {amount}." if amount else ".")
        )
    elif stated.get("shared_personal_data"):
        summaries.append("File dữ liệu khách hàng đã được chuyển cho một bên khác.")
    if stated.get("received_payment") and not stated.get("data_sale_proceeds_vnd"):
        summaries.append(
            "Người thực hiện được mô tả là đã nhận một khoản tiền, nhưng chưa rõ số tiền và căn cứ của khoản nhận."
        )
    if stated.get("company_authorization") is False:
        summaries.append("Tình huống nêu rõ doanh nghiệp không cho phép việc chuyển dữ liệu.")
    if stated.get("data_subject_consent") is False:
        summaries.append("Tình huống nêu rõ khách hàng không cho phép việc chuyển dữ liệu.")
    return summaries


def _next_steps(
    query: str,
    positive_keys: set[str],
    fact_state: Mapping[str, Any],
) -> list[str]:
    folded = _fold(query)
    steps: list[str] = []
    if "create_ai_deepfake" in positive_keys:
        steps.append("Dừng ngay việc tạo hoặc sử dụng nội dung giả mạo.")
    if re.search(r"(?:chuyen tien|thanh toan|ngan hang|ke toan)", folded):
        steps.extend(
            (
                "Không chuyển tiền trước khi xác minh yêu cầu qua kênh liên lạc chính thức.",
                "Lưu giữ bản ghi cuộc gọi, số điện thoại, tin nhắn và chứng từ giao dịch.",
                "Nếu đã phát sinh giao dịch, thông báo ngay cho doanh nghiệp và ngân hàng.",
            )
        )
    if {"unauthorized_access", "exploit_vulnerability"}.intersection(positive_keys):
        steps.append("Ngừng truy cập hoặc khai thác hệ thống và bảo toàn log, thiết bị liên quan.")
    data_event = {
        "collect_personal_data",
        "share_personal_data",
        "sell_personal_data",
        "retain_personal_data",
        "process_personal_data_without_consent",
    }.intersection(positive_keys) or bool(
        (fact_state.get("stated_facts") or {}).get("shared_personal_data")
    )
    if data_event:
        steps.extend(
            (
                "Yêu cầu các bên dừng sử dụng/chuyển tiếp và bảo toàn nguyên trạng chứng cứ về file dữ liệu.",
                "Lưu email, log truy cập hoặc tải file, thiết bị liên quan và bằng chứng thanh toán.",
                "Xác định phạm vi chủ thể bị ảnh hưởng, trường dữ liệu và các bên đã nhận dữ liệu.",
                "Đánh giá quy trình xử lý sự cố và nghĩa vụ thông báo trước khi liên hệ khách hàng hoặc cơ quan có thẩm quyền.",
            )
        )
    steps.append(
        "Nếu có dấu hiệu chiếm đoạt, thiệt hại hoặc hành vi tiếp diễn, liên hệ cơ quan có thẩm quyền."
    )
    return _unique(steps, limit=6)


def _liability_categories(
    fact_state: Mapping[str, Any],
    sources: Iterable[Any],
) -> dict[str, str]:
    corpus = _fold(
        " ".join(
            f"{getattr(source, 'document', '')} {getattr(source, 'body', '')}"
            for source in sources
        )
    )
    stated = fact_state.get("stated_facts") or {}
    categories: dict[str, str] = {}
    if stated.get("former_employee"):
        categories["employment_or_internal"] = (
            "Cần đối chiếu hợp đồng, cam kết bảo mật và quy chế nội bộ."
        )
    if re.search(r"\b(?:boi thuong|trach nhiem dan su)\b", corpus):
        categories["civil"] = (
            "Có căn cứ xem xét bồi thường; vẫn cần chứng minh thiệt hại và quan hệ nhân quả."
        )
    if re.search(r"\b(?:xu phat hanh chinh|phat tien)\b", corpus):
        categories["administrative"] = (
            "Có căn cứ xem xét xử lý hành chính; chưa tự tính mức phạt nếu nguồn chưa đủ."
        )
    if re.search(r"\b(?:truy cuu trach nhiem hinh su|xu ly hinh su)\b", corpus):
        categories["criminal"] = (
            "Chỉ đặt ra khi đủ dấu hiệu cấu thành và chứng cứ của tội danh cụ thể."
        )
    return categories


def _sanction_source(document: str, body: str) -> bool:
    folded = _fold(f"{document} {body}")
    return bool(
        re.search(
            r"(?:xu phat|phat tien|hinh phat|truy cuu|muc phat|nghi dinh.{0,30}xu phat)",
            folded,
        )
    )


def build_answer_assessment(
    *,
    query: str,
    behavior_profile: Mapping[str, Any] | None,
    retrieval_decisions: Iterable[Mapping[str, Any]],
    context_texts: Iterable[str],
    final_context_records: Iterable[Mapping[str, Any]] = (),
    retrieval_is_complete: bool,
    scenario_fact_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the one assessment object used by the final answer renderer."""

    sources = build_grounded_sources(context_texts, query=query)
    fact_state = dict(
        scenario_fact_state
        or extract_legal_scenario_facts(query).as_dict()
    )
    applicability = [
        dict(decision)
        for decision in retrieval_decisions
        if str(decision.get("decision_stage") or "") == "applicability"
    ]
    kept = [
        decision
        for decision in applicability
        if _clean(decision.get("decision")).upper() in _KEPT
    ]
    actions, allowed_keys = _profile_keys(behavior_profile)

    positive_by_key: dict[str, set[str]] = {}
    direct_action_match = False
    any_partial = False
    for decision in kept:
        is_direct_decision = (
            _clean(decision.get("decision")).upper() == "KEEP"
            and _clean(decision.get("level")).upper() == "HIGH"
        )
        for key, match in _decision_matches(decision):
            if key not in allowed_keys or match not in _POSITIVE:
                continue
            positive_by_key.setdefault(key, set()).add(match)
            any_partial = any_partial or match == "PARTIAL_MATCH"
            if is_direct_decision and key in actions and match == "MATCH":
                direct_action_match = True

    question_supported_keys = _query_supported_behavior_keys(
        query,
        (source.body for source in sources),
    )
    positive_keys = set(positive_by_key).union(question_supported_keys)
    if kept and question_supported_keys:
        # The fact comes from the question and is confirmed by a SOURCE that
        # already survived Applicability; this does not infer a new legal rule.
        direct_action_match = True
    if "sell_personal_data" in question_supported_keys:
        # The original question states the sale and a retained SOURCE states
        # the same prohibited act verbatim. This exact deterministic match
        # must not fluctuate with the Applicability model's JSON wording.
        direct_action_match = True
    if not sources:
        status = NO_MATCH
    elif direct_action_match:
        status = LIKELY_VIOLATION
    elif positive_keys:
        status = PARTIAL_MATCH
    else:
        status = INSUFFICIENT_FACTS

    explicit_query_facts = _query_facts(query, positive_keys)
    matched_facts = _unique(
        (
            *explicit_query_facts,
            *_stated_fact_summaries(fact_state),
            *(
                _FACT_BY_BEHAVIOR.get(
                    key,
                    f"Có dấu hiệu liên quan đến {_DEFINITIONS.get(key, key.replace('_', ' '))}.",
                )
                for key in sorted(positive_keys)
                if not (key == "create_ai_deepfake" and explicit_query_facts)
            ),
        ),
        limit=14,
    )
    proposed_missing = _unique(
        (
            *(
                _clean(decision.get("missing_conditions"))
                for decision in kept
                if _meaningful_missing(decision.get("missing_conditions"))
            ),
            *_scenario_missing_facts(fact_state),
        ),
        limit=12,
    )
    stated_facts = fact_state.get("stated_facts") or {}
    long_fact_guard = bool(
        stated_facts.get("transfer_executed")
        or stated_facts.get("transferred_amount_vnd")
    )
    if long_fact_guard:
        missing_facts, missing_fact_keys = filter_answered_missing_facts(
            proposed_missing,
            fact_state,
        )
    else:
        missing_facts = proposed_missing
        missing_fact_keys = [
            key
            for key in (missing_fact_key(value) for value in proposed_missing)
            if key
        ]
    missing_fact_keys.extend(
        key
        for key in (fact_state.get("unknown_legal_elements") or {})
        if key not in missing_fact_keys
    )
    if status in {PARTIAL_MATCH, INSUFFICIENT_FACTS} and not missing_facts:
        missing_facts.append(
            "Cần bổ sung dữ kiện để đối chiếu đầy đủ các điều kiện áp dụng của quy định."
        )
    if status == NO_MATCH and not missing_facts:
        missing_facts.append(
            "Cần truy xuất được nguồn pháp luật điều chỉnh trực tiếp hành vi trong tình huống."
        )
    if not retrieval_is_complete and status != NO_MATCH:
        missing_facts = _unique(
            (
                *missing_facts,
                "Việc truy xuất căn cứ hiện chưa đầy đủ; có thể còn quy định liên quan chưa được đưa vào ngữ cảnh.",
            ),
            limit=8,
        )

    decision_by_article: dict[tuple[str, str], Mapping[str, Any]] = {}
    for decision in kept:
        key = (
            _fold(decision.get("document")),
            _fold(decision.get("article")),
        )
        previous = decision_by_article.get(key)
        current_priority = (
            _clean(decision.get("decision")).upper() == "KEEP",
            _clean(decision.get("level")).upper() == "HIGH",
        )
        previous_priority = (
            _clean(previous.get("decision")).upper() == "KEEP",
            _clean(previous.get("level")).upper() == "HIGH",
        ) if previous else (False, False)
        if previous is None or current_priority > previous_priority:
            decision_by_article[key] = decision

    records = list(final_context_records)
    record_by_coordinates: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    for record in records:
        coordinates = (
            _fold(record.get("document")),
            _fold(record.get("article")),
            _fold(record.get("clause")),
            _fold(record.get("point")),
        )
        record_by_coordinates.setdefault(coordinates, record)

    def source_record(source: Any) -> Mapping[str, Any]:
        exact = record_by_coordinates.get(
            (
                _fold(source.document),
                _fold(source.article),
                _fold(source.clause),
                _fold(source.point),
            )
        )
        if exact is not None:
            return exact
        return next(
            (
                record
                for coordinates, record in record_by_coordinates.items()
                if coordinates[:2]
                == (_fold(source.document), _fold(source.article))
            ),
            {},
        )

    applicable_sources = []
    for source in sources:
        record = source_record(source)
        applicable_sources.append(
            {
                "source_id": source.source_id,
                "document": source.document,
                "article": source.article,
                "clause": source.clause,
                "point": source.point,
                "body": source.body,
                "applicability_decision": _clean(
                    decision_by_article.get(
                        (_fold(source.document), _fold(source.article)),
                        {},
                    ).get("decision")
                ).upper(),
                "applicability_level": _clean(
                    decision_by_article.get(
                        (_fold(source.document), _fold(source.article)),
                        {},
                    ).get("level")
                ).upper(),
                "cross_encoder_score": record.get("cross_encoder_score"),
                "retrieval_score": record.get(
                    "retrieval_score",
                    record.get("score"),
                ),
                "behavior_score": record.get("behavior_score"),
            }
        )
    sanction_available = any(
        _sanction_source(source.document, source.body) for source in sources
    )
    if status in {LIKELY_VIOLATION, PARTIAL_MATCH} and not sanction_available:
        missing_facts = _unique(
            (
                *missing_facts,
                "Loại trách nhiệm cụ thể và mức xử lý chưa có căn cứ trực tiếp trong các nguồn hiện tại.",
            ),
            limit=8,
        )

    assessment = {
        "status": status,
        "matched_facts": matched_facts,
        "missing_facts": missing_facts,
        "applicable_sources": applicable_sources,
        "sanction_available": sanction_available,
        "next_steps": _next_steps(query, positive_keys, fact_state),
        "liability_categories": _liability_categories(fact_state, sources),
        "retrieval_is_complete": bool(retrieval_is_complete),
        "context_record_count": len(records),
        "has_partial_match": bool(any_partial),
        "stated_facts": dict(fact_state.get("stated_facts") or {}),
        "supported_inferences": dict(
            fact_state.get("supported_inferences") or {}
        ),
        "unknown_legal_elements": dict(
            fact_state.get("unknown_legal_elements") or {}
        ),
        "missing_fact_keys": missing_fact_keys,
    }
    logger.info(
        "[answer_assessment] %s",
        json.dumps(
            {
                "status": status,
                "matched_facts": matched_facts,
                "missing_facts": missing_facts,
                "applicable_source_ids": [
                    source["source_id"] for source in applicable_sources
                ],
                "sanction_available": sanction_available,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )
    return assessment
