"""Đánh giá khả năng áp dụng pháp luật trước generation.

Semantic relevance trả lời "đoạn này nói gần chủ đề nào"; applicability phải
trả lời câu chặt hơn: "quy tắc trong Điều này có thực sự điều chỉnh hành vi
được mô tả không". Kết quả chỉ dùng để lọc context và ghi debug log.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
import re
from typing import Any, Iterable, Sequence

from src.agents.common.grounded_validation import GroundedSource, build_grounded_sources
from src.agents.common.legal_element_coverage import (
    NOT_APPLICABLE as ELEMENT_NOT_APPLICABLE,
    evaluate_provision_element_coverage,
)
from src.agents.common.legal_scenario_facts import (
    extract_legal_scenario_facts,
    filter_answered_missing_facts,
)
from src.agents.common.retrieval_provenance import article_key
from src.retrieval.legal_behaviors import (
    BehaviorProfile,
    extract_legal_behavior,
    score_behavior_relevance,
)

logger = logging.getLogger(__name__)

HIGH = "HIGH"
MEDIUM = "MEDIUM"
LOW = "LOW"

KEEP = "KEEP"
WEAK_KEEP = "WEAK_KEEP"
REMOVE = "REMOVE"

_GENERIC_REASON_RE = re.compile(
    r"(?:tình tiết(?: nêu trên)? thuộc đúng nhóm (?:hoạt động|đối tượng)|"
    r"có liên hệ trực tiếp|(?:là|đây là) căn cứ trực tiếp|"
    r"có dấu hiệu(?: vi phạm)?\.?$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ApplicabilityDecision:
    candidate_id: str
    document: str
    article: str
    scope: str
    situation_behavior: str
    level: str
    explanation: str
    missing_conditions: str
    behavior_matches: tuple[tuple[str, str], ...] = ()
    behavior_score: float = 0.0
    validation_status: str = "VALID"
    reason_rejected: str = ""
    decision: str = REMOVE
    is_seed: bool = False
    seed_survived: bool = False
    seed_removed: bool = False
    behavior_preserved: bool = False
    applicability_removed: bool = False
    reason_removed: str = ""
    decision_stage: str = "applicability"
    required_elements: tuple[str, ...] = ()
    matched_elements: tuple[str, ...] = ()
    missing_required_elements: tuple[str, ...] = ()
    element_applicability: str = "PARTIAL_MATCH"
    element_reason: str = ""


@dataclass(frozen=True)
class ApplicabilityResult:
    contexts: tuple[str, ...]
    decisions: tuple[ApplicabilityDecision, ...]
    retrieval_gap: bool
    gap_reason: str


def _article_key(source: GroundedSource) -> tuple[str, str]:
    return (
        " ".join(source.document.casefold().split()),
        " ".join(source.article.casefold().split()),
    )


def _group_sources(
    contexts: Iterable[str],
) -> list[tuple[str, list[GroundedSource]]]:
    grouped: dict[tuple[str, str], list[GroundedSource]] = {}
    seen: set[tuple[str, str, str, str, str]] = set()
    # Parse từng context để giữ đúng thứ tự đã qua semantic relevance; hàm
    # build_grounded_sources khi nhận cả pool sẽ ưu tiên loại văn bản và làm
    # mất ánh xạ A1/A2 với candidate order.
    for context in contexts:
        for source in build_grounded_sources([context], limit=256):
            exact_key = (
                source.document,
                source.article,
                source.clause or "",
                source.point or "",
                source.body,
            )
            if exact_key in seen:
                continue
            seen.add(exact_key)
            grouped.setdefault(_article_key(source), []).append(source)
    return [
        (f"A{index}", article_sources)
        for index, article_sources in enumerate(grouped.values(), start=1)
    ]


def _article_text(sources: Sequence[GroundedSource], max_chars: int = 4200) -> str:
    text = "\n".join(source.text for source in sources)
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + "…"


def _build_prompt(
    query: str,
    grouped: Sequence[tuple[str, list[GroundedSource]]],
    behavior_profile: BehaviorProfile,
    scenario_fact_state: dict[str, Any],
) -> str:
    candidates = "\n\n".join(
        f'<CANDIDATE id="{candidate_id}">\n{_article_text(sources)}\n</CANDIDATE>'
        for candidate_id, sources in grouped
    )
    expected_ids = ", ".join(candidate_id for candidate_id, _ in grouped)
    behavior_card = json.dumps(behavior_profile.as_dict(), ensure_ascii=False)
    behavior_keys = list(
        dict.fromkeys(
            behavior_profile.actions
            + behavior_profile.objects
            + behavior_profile.purposes
            + behavior_profile.conditions
        )
    )
    expected_behaviors = ", ".join(behavior_keys) or "(không có)"
    fact_card = json.dumps(scenario_fact_state, ensure_ascii=False)
    return f"""
Bạn là bộ kiểm tra khả năng áp dụng pháp luật. Đây là bước LỌC, không phải bước
tư vấn và không được dùng kiến thức ngoài các CANDIDATE.

TÌNH HUỐNG:
{query}

BEHAVIOR_CARD BẤT BIẾN DO RETRIEVAL CUNG CẤP:
{behavior_card}

FACT_STATE BẤT BIẾN DO HỆ THỐNG TRÍCH TỪ TÌNH HUỐNG:
{fact_card}

CÁC ĐIỀU LUẬT ĐÃ RETRIEVE:
{candidates}

Có {len(grouped)} candidate. Bắt buộc trả đúng {len(grouped)} decision với đủ
các ID: {expected_ids}. Không được bỏ qua ID nào.

Với TỪNG CANDIDATE, thực hiện độc lập:
1. Nêu phạm vi/quyền/nghĩa vụ/hành vi mà chính nội dung Điều điều chỉnh.
2. Với TỪNG key trong BEHAVIOR_CARD, chỉ chấm MATCH, PARTIAL_MATCH hoặc NOT_MATCH.
   Bắt buộc đủ các key sau: {expected_behaviors}.
3. So sánh cấu thành hành vi, đối tượng, chủ thể và mục đích/điều kiện nếu nguồn có.
4. Chấm:
   - HIGH: quy tắc trực tiếp điều chỉnh hành vi cốt lõi.
   - MEDIUM: quy tắc chỉ điều chỉnh một phần độc lập hoặc hậu quả của hành vi.
   - LOW: chỉ trùng từ khóa/bối cảnh, cần một hành vi khác, hoặc không thể giải
     thích cụ thể vì sao quy tắc áp dụng.

QUY TẮC NGHIÊM NGẶT:
- Việc cùng có từ "video", "hình ảnh", "thương mại", "dữ liệu" không đủ để
  kết luận áp dụng.
- Ví dụ, quy định về khai thác bản ghi âm/ghi hình thương mại không tự động
  điều chỉnh hành vi tạo deepfake; chỉ giữ nếu tình huống thực sự có hành vi
  sử dụng đối tượng/quyền mà chính quy định đó mô tả.
- Nếu explanation không chỉ ra được sự tương ứng cụ thể giữa hành vi và nội
  dung Điều, bắt buộc LOW.
- Không nêu tên hoặc số hiệu văn bản không có trong CANDIDATE.
- Không được viết lại, diễn giải thành một hành vi mới hoặc thêm behavior key
  ngoài BEHAVIOR_CARD. Không trả field situation_behavior.
- Chỉ ``stated_facts`` là dữ kiện người dùng đã nêu. ``supported_inferences``
  phải giữ ở mức suy luận; ``unknown_legal_elements`` không được viết thành
  dữ kiện chắc chắn.
- Không đưa một nội dung đã có trong ``stated_facts`` vào missing_conditions.
- Nếu các nguồn chỉ bao phủ ngoại vi và có vẻ thiếu căn cứ điều chỉnh hành vi
  cốt lõi, đặt retrieval_gap=true nhưng không tự đoán tên điều luật còn thiếu.
- Viết ngắn: scope/hành vi tối đa 25 từ, explanation tối đa 60 từ,
  missing_conditions tối đa 35 từ. Nếu HIGH và không thiếu điều kiện, ghi rõ
  "Không còn điều kiện thiếu vì tình huống đã thể hiện ...".

Chỉ trả JSON hợp lệ, không Markdown:
{{
  "retrieval_gap": false,
  "gap_reason": "",
  "decisions": [
    {{
      "id": "ID chính xác của candidate",
      "scope": "phạm vi cụ thể rút từ nội dung Điều",
      "behavior_matches": [
        {{"behavior_key": "key có nguyên văn trong BEHAVIOR_CARD", "match": "MATCH|PARTIAL_MATCH|NOT_MATCH"}}
      ],
      "applicability": "HIGH|MEDIUM|LOW",
      "explanation": "so sánh cụ thể, không dùng câu chung chung",
      "missing_conditions": "điều kiện còn thiếu; hoặc Không còn điều kiện thiếu vì ..."
    }}
  ]
}}
""".strip()


def _json_payload(content: Any) -> dict[str, Any] | None:
    text = str(content or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _normalise_level(value: Any) -> str:
    level = str(value or "").strip().upper().replace(" ", "_")
    aliases = {
        "CAO": HIGH,
        "TRUNG_BÌNH": MEDIUM,
        "TRUNG_BINH": MEDIUM,
        "THẤP": LOW,
        "THAP": LOW,
    }
    return aliases.get(level, level if level in {HIGH, MEDIUM, LOW} else LOW)


def _safe_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _validated_decision(
    candidate_id: str,
    sources: Sequence[GroundedSource],
    raw: dict[str, Any] | None,
    *,
    behavior_profile: BehaviorProfile,
    behavior_score: float,
    is_seed: bool,
    is_recursive: bool,
    candidate_score: float,
    max_seed_score: float,
    candidate_actions: set[str],
    seed_actions: set[str],
    scenario_fact_state: dict[str, Any],
) -> ApplicabilityDecision:
    first = sources[0]
    raw = raw or {}
    scope = _safe_text(raw.get("scope"))
    explanation = _safe_text(raw.get("explanation"))
    missing = _safe_text(raw.get("missing_conditions"))
    stated_facts = scenario_fact_state.get("stated_facts") or {}
    if stated_facts.get("transfer_executed"):
        filtered_missing, _ = filter_answered_missing_facts(
            re.split(r"(?<=[.;?])\s+|\s*;\s*", missing),
            scenario_fact_state,
        )
        missing = "; ".join(filtered_missing)
    raw_present = bool(raw)
    requested_level = _normalise_level(raw.get("applicability"))
    level = requested_level
    if level == HIGH and not missing:
        missing = (
            "Không còn điều kiện thiếu được xác định từ tình huống và nội dung Điều."
        )

    allowed_keys = list(
        dict.fromkeys(
            behavior_profile.actions
            + behavior_profile.objects
            + behavior_profile.purposes
            + behavior_profile.conditions
        )
    )
    allowed_set = set(allowed_keys)
    matches: list[tuple[str, str]] = []
    behavior_errors: list[str] = []
    raw_matches = raw.get("behavior_matches")
    if raw.get("situation_behavior"):
        behavior_errors.append("output chứa situation_behavior tự do")
    if not isinstance(raw_matches, list):
        behavior_errors.append("thiếu behavior_matches")
        raw_matches = []
    seen_behavior_keys: set[str] = set()
    for item in raw_matches:
        if not isinstance(item, dict):
            behavior_errors.append("behavior assessment không phải object")
            continue
        key = _safe_text(item.get("behavior_key"))
        match = _safe_text(item.get("match")).upper()
        if key not in allowed_set:
            behavior_errors.append(f"behavior mới/không hợp lệ: {key or '(rỗng)'}")
            continue
        if key in seen_behavior_keys:
            behavior_errors.append(f"behavior lặp: {key}")
            continue
        if match not in {"MATCH", "PARTIAL_MATCH", "NOT_MATCH"}:
            behavior_errors.append(f"match không hợp lệ cho {key}: {match or '(rỗng)'}")
            continue
        seen_behavior_keys.add(key)
        matches.append((key, match))
    missing_keys = [key for key in allowed_keys if key not in seen_behavior_keys]
    if missing_keys:
        behavior_errors.append("thiếu behavior key: " + ", ".join(missing_keys))

    primary_matches = {
        key: match for key, match in matches if key in set(behavior_profile.actions)
    }
    # A primary-action match is mandatory only when the Behavior Card actually
    # contains a primary action.  The benchmark exposed object-only,
    # purpose-only and condition-only cards that were previously rejected by
    # checking an empty action set.
    if level == HIGH and behavior_profile.has_primary_action and not any(
        match in {"MATCH", "PARTIAL_MATCH"} for match in primary_matches.values()
    ):
        behavior_errors.append("HIGH nhưng không match primary action")

    near_zero = float(os.getenv("APPLICABILITY_MIN_BEHAVIOR_SCORE", "0.18"))
    if level == HIGH and behavior_profile.has_primary_action and behavior_score < near_zero:
        behavior_errors.append(
            f"HIGH nhưng behavior_score={behavior_score:.4f} < {near_zero:.4f}"
        )
    if (
        is_recursive
        and candidate_score > max_seed_score
        and seed_actions
        and not candidate_actions.intersection(seed_actions)
    ):
        behavior_errors.append(
            "recursive candidate vượt seed score nhưng không cùng matched behavior action"
        )

    behavior = "; ".join(f"{key}:{match}" for key, match in matches)

    # HIGH/MEDIUM chỉ hợp lệ khi model giải thích được bằng nội dung cụ thể.
    # Thiếu field, câu rỗng hoặc câu mẫu chung đều bị hạ xuống LOW.
    invalid_reasoning = (
        len(scope) < 12
        or not matches
        or len(explanation) < 45
        or len(missing) < 12
        or bool(_GENERIC_REASON_RE.search(explanation))
    )
    if level in {HIGH, MEDIUM} and invalid_reasoning:
        logger.debug(
            "[legal_applicability] %s bị hạ LOW vì không giải thích đủ cụ thể.",
            candidate_id,
        )
        level = LOW
        explanation = explanation or "Không giải thích được sự tương ứng cụ thể."
        behavior_errors.append("giải thích applicability không đủ cụ thể")

    validation_status = "INVALID" if behavior_errors else "VALID"
    reason_rejected = "; ".join(dict.fromkeys(behavior_errors))
    if behavior_errors:
        level = LOW
        logger.debug(
            "[legal_applicability] applicability_validation=%s",
            {
                "candidate_id": candidate_id,
                "status": validation_status,
                "reason_rejected": reason_rejected,
                "behavior_score": behavior_score,
            },
        )

    positive_matches = {
        key: match
        for key, match in matches
        if match in {"MATCH", "PARTIAL_MATCH"}
    }
    if behavior_profile.has_primary_action:
        applicable_match = any(
            key in set(behavior_profile.actions)
            for key in positive_matches
        )
    else:
        applicable_match = bool(positive_matches)

    behavior_medium = float(
        os.getenv("BEHAVIOR_GATE_MIN_SCORE", "0.18")
    )
    medium_behavior = behavior_score >= behavior_medium
    recursive_mismatch = any(
        "recursive candidate vượt seed score" in error
        for error in behavior_errors
    )
    invented_behavior = any(
        "behavior mới/không hợp lệ" in error
        or "situation_behavior tự do" in error
        for error in behavior_errors
    )

    # Decision logic is deliberately separate from validation level.  Invalid
    # behavior output is never trusted, but a Phase 2 seed may still be weakly
    # preserved from retrieval provenance instead of disappearing solely due
    # to a validator formatting/reasoning defect.
    decision = REMOVE
    behavior_preserved = False
    if raw_present and not recursive_mismatch:
        if validation_status == "VALID" and requested_level == HIGH:
            decision = KEEP
        elif validation_status == "VALID" and requested_level == MEDIUM:
            decision = WEAK_KEEP
        elif (
            requested_level in {HIGH, MEDIUM}
            and applicable_match
            and (is_seed or medium_behavior)
            and not invented_behavior
        ):
            decision = WEAK_KEEP
            behavior_preserved = medium_behavior
        elif (
            behavior_profile.is_empty
            and is_seed
            and requested_level in {HIGH, MEDIUM}
        ):
            # No behavior key exists to validate.  Reject any invented key,
            # but preserve the retrieval seed as weak context; Generation sees
            # only the retrieved law text, never the invented behavior output.
            decision = WEAK_KEEP

    seed_survived = is_seed and decision != REMOVE
    seed_removed = is_seed and decision == REMOVE
    reason_removed = ""
    if decision == REMOVE:
        if not raw_present:
            reason_removed = "Applicability không trả decision hợp lệ cho candidate."
        elif recursive_mismatch:
            reason_removed = "Recursive candidate không cùng behavior với seed."
        elif requested_level == LOW:
            reason_removed = "Applicability đánh giá LOW; candidate không trực tiếp điều chỉnh hành vi."
        elif invented_behavior:
            reason_removed = "Applicability sinh behavior ngoài Behavior Card."
        elif behavior_score < behavior_medium and not is_seed:
            reason_removed = (
                f"behavior_score={behavior_score:.4f} < {behavior_medium:.4f} "
                "và candidate không phải seed."
            )
        else:
            reason_removed = reason_rejected or "Applicability không đủ điều kiện giữ."

    element_coverage = evaluate_provision_element_coverage(
        "\n".join(source.text for source in sources),
        scenario_fact_state,
    )
    if element_coverage.applicability == ELEMENT_NOT_APPLICABLE:
        decision = REMOVE
        level = LOW
        seed_survived = False
        seed_removed = is_seed
        reason_removed = element_coverage.reason

    return ApplicabilityDecision(
        candidate_id=candidate_id,
        document=first.document,
        article=first.article,
        scope=scope or "Không xác định được phạm vi điều chỉnh.",
        situation_behavior=behavior or "Không trích xuất được hành vi tương ứng.",
        level=level,
        explanation=explanation or "Không đủ cơ sở để áp dụng Điều này.",
        missing_conditions=missing or "Chưa xác định được điều kiện áp dụng.",
        behavior_matches=tuple(matches),
        behavior_score=float(behavior_score),
        validation_status=validation_status,
        reason_rejected=reason_rejected,
        decision=decision,
        is_seed=is_seed,
        seed_survived=seed_survived,
        seed_removed=seed_removed,
        behavior_preserved=behavior_preserved,
        applicability_removed=decision == REMOVE,
        reason_removed=reason_removed,
        required_elements=element_coverage.required_elements,
        matched_elements=element_coverage.matched_elements,
        missing_required_elements=element_coverage.missing_required_elements,
        element_applicability=element_coverage.applicability,
        element_reason=element_coverage.reason,
    )


def check_legal_applicability(
    query: str,
    context_texts: Iterable[str],
    *,
    llm_client,
    behavior_profile: BehaviorProfile | None = None,
    candidate_records: Iterable[dict[str, Any]] | None = None,
    scenario_fact_state: dict[str, Any] | None = None,
) -> ApplicabilityResult:
    """Phân loại candidate thành KEEP/WEAK_KEEP/REMOVE trước Generation."""

    behavior_profile = behavior_profile or extract_legal_behavior(query)
    scenario_fact_state = scenario_fact_state or extract_legal_scenario_facts(
        query
    ).as_dict()
    grouped = _group_sources(context_texts)
    if not grouped:
        return ApplicabilityResult((), (), True, "Không có Điều luật để đánh giá.")

    raw_by_id: dict[str, dict[str, Any]] = {}
    payloads: list[dict[str, Any]] = []
    pending = list(grouped)
    # Mặc định chỉ gọi đúng một lần. Đây là một cổng chọn lọc trước generation,
    # không đáng để giữ Ollama bận thêm nhiều phút chỉ nhằm sửa JSON thiếu ID.
    # Có thể bật lại repair rõ ràng qua env khi dùng provider nhanh hơn.
    max_repairs = max(0, int(os.getenv("LEGAL_APPLICABILITY_REPAIR_ATTEMPTS", "0")))
    max_tokens = max(128, int(os.getenv("LEGAL_APPLICABILITY_MAX_TOKENS", "800")))
    for attempt in range(max_repairs + 1):
        if not pending:
            break
        try:
            response = llm_client.invoke(
                _build_prompt(
                    query,
                    pending,
                    behavior_profile,
                    scenario_fact_state,
                ),
                tag="legal_applicability",
                max_completion_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            payload = _json_payload(getattr(response, "content", response))
        except Exception:
            logger.warning(
                "Legal Applicability Check thất bại ở lượt %s; candidate thiếu sẽ bị loại.",
                attempt + 1,
                exc_info=True,
            )
            payload = None
        if payload:
            payloads.append(payload)
            pending_ids = {candidate_id for candidate_id, _ in pending}
            for item in payload.get("decisions", []):
                if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                    continue
                candidate_id = item["id"].strip()
                if candidate_id in pending_ids:
                    raw_by_id[candidate_id] = item
        pending = [
            group for group in pending if group[0] not in raw_by_id
        ]
        if pending:
            logger.debug(
                "[legal_applicability] Lượt %s còn thiếu decision cho %s; yêu cầu lại.",
                attempt + 1,
                [candidate_id for candidate_id, _ in pending],
            )

    records_by_article: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in candidate_records or ():
        records_by_article.setdefault(article_key(record), []).append(record)
    seed_records = [
        record
        for records in records_by_article.values()
        for record in records
        if record.get("is_seed")
    ]
    max_seed_score = max(
        (
            max(
                float(record.get("retrieval_score") or record.get("score") or 0.0),
                float(record.get("behavior_score") or 0.0),
            )
            for record in seed_records
        ),
        default=0.0,
    )
    seed_actions = {
        str(action)
        for record in seed_records
        for action in (record.get("matched_behavior_actions") or ())
    }

    decisions: list[ApplicabilityDecision] = []
    kept_contexts: list[str] = []
    for candidate_id, sources in grouped:
        key = _article_key(sources[0])
        records = records_by_article.get(key, [])
        fallback_match = score_behavior_relevance(
            behavior_profile,
            "\n".join(source.text for source in sources),
        )
        behavior_score = max(
            (float(record.get("behavior_score") or 0.0) for record in records),
            default=float(fallback_match.score),
        )
        candidate_score = max(
            (
                max(
                    float(record.get("retrieval_score") or record.get("score") or 0.0),
                    float(record.get("behavior_score") or 0.0),
                )
                for record in records
            ),
            default=0.0,
        )
        candidate_actions = {
            str(action)
            for record in records
            for action in (record.get("matched_behavior_actions") or ())
        } or set(fallback_match.matched_actions)
        is_seed = any(bool(record.get("is_seed")) for record in records)
        decision = _validated_decision(
            candidate_id,
            sources,
            raw_by_id.get(candidate_id),
            behavior_profile=behavior_profile,
            behavior_score=behavior_score,
            is_seed=is_seed,
            is_recursive=any(not record.get("is_seed") for record in records),
            candidate_score=candidate_score,
            max_seed_score=max_seed_score,
            candidate_actions=candidate_actions,
            seed_actions=seed_actions,
            scenario_fact_state=scenario_fact_state,
        )
        decisions.append(decision)
        action = "drop" if decision.decision == REMOVE else "keep"
        logger.debug(
            "[legal_applicability] applicability_behavior=%s",
            {
                "id": candidate_id,
                "behavior_card": behavior_profile.as_dict(),
                "matches": list(decision.behavior_matches),
                "behavior_score": decision.behavior_score,
            },
        )
        logger.debug(
            "[legal_applicability] id=%s document=%r article=%r scope=%r "
            "behavior=%r applicability=%s decision=%s explanation=%r "
            "missing=%r action=%s seed_survived=%s seed_removed=%s "
            "behavior_preserved=%s applicability_removed=%s reason_removed=%r "
            "decision_stage=%s",
            candidate_id,
            decision.document,
            decision.article,
            decision.scope,
            decision.situation_behavior,
            decision.level,
            decision.decision,
            decision.explanation,
            decision.missing_conditions,
            action,
            decision.seed_survived,
            decision.seed_removed,
            decision.behavior_preserved,
            decision.applicability_removed,
            decision.reason_removed,
            decision.decision_stage,
        )
        if decision.reason_rejected:
            logger.debug(
                "[legal_applicability] reason_rejected=%s",
                {
                    "id": candidate_id,
                    "document": decision.document,
                    "article": decision.article,
                    "reason": decision.reason_rejected,
                },
            )
        if decision.decision != REMOVE:
            kept_contexts.append("\n".join(source.text for source in sources))

    explicit_gap = any(bool(payload.get("retrieval_gap")) for payload in payloads)
    gap_reason = next(
        (
            _safe_text(payload.get("gap_reason"))
            for payload in payloads
            if _safe_text(payload.get("gap_reason"))
        ),
        "",
    )
    retrieval_gap = explicit_gap or bool(pending) or not kept_contexts
    if retrieval_gap and not gap_reason:
        gap_reason = "Không có căn cứ trực tiếp cho toàn bộ hành vi cốt lõi."
    logger.debug(
        "[legal_applicability] Giữ %s/%s Điều; retrieval_gap=%s reason=%r",
        len(kept_contexts),
        len(grouped),
        retrieval_gap,
        gap_reason,
    )
    return ApplicabilityResult(
        contexts=tuple(kept_contexts),
        decisions=tuple(decisions),
        retrieval_gap=retrieval_gap,
        gap_reason=gap_reason,
    )
