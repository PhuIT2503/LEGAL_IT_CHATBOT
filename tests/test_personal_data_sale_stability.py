from src.agents.agent_generation.answer_assessment import build_answer_assessment
from src.agents.common.grounded_validation import (
    _is_query_fact_number,
    build_grounded_sources,
    render_grounded_answer,
)
from src.agents.common.legal_scenario_facts import extract_legal_scenario_facts
from src.agents.common.query_expansion import expand_legal_query
from src.agents.common.retrieval_contract import (
    annotate_retrieval_contract_records,
    assess_retrieval_contract,
    build_retrieval_rescue_query,
)
from src.agents.common.retrieval_ranking import select_balanced_top_k
from src.retrieval.legal_behaviors import extract_legal_behavior
from src.retrieval.legal_domains import select_legal_domains
from src.retrieval.legal_event import extract_canonical_legal_event


CONCRETE_SCENARIO = (
    "B là nhân viên cũ. Sau khi nghỉ việc, B giữ lại file dữ liệu của "
    "20.000 khách hàng gồm họ tên, số điện thoại, email và lịch sử mua hàng. "
    "B bán file này cho một công ty quảng cáo với giá 30 triệu đồng mà không "
    "được công ty hoặc khách hàng cho phép. B có thể chịu những trách nhiệm nào?"
)

AMBIGUOUS_SCENARIO = (
    "Một nhân viên cũ chuyển một file khách hàng cho đối tác và nhận một "
    "khoản tiền. Việc này có vấn đề pháp lý gì?"
)


def test_concrete_personal_data_sale_event_preserves_material_facts():
    event = extract_canonical_legal_event(CONCRETE_SCENARIO)

    assert event.actions == (
        "sell_personal_data",
        "share_personal_data",
        "retain_personal_data",
    )
    assert event.objects == ("personal_data",)
    assert {"advertising", "commercial_gain"}.issubset(event.purposes)
    assert {"without_consent", "without_authorization"}.issubset(event.conditions)
    assert event.actor_roles == ("former_employee",)
    assert event.data_fields == (
        "full_name",
        "phone_number",
        "email",
        "purchase_history",
    )
    assert event.affected_subject_count == 20_000
    assert event.sale_amount_vnd == 30_000_000
    assert event.recipient_type == "advertising_company"
    assert event.required_domains == ("personal_data",)
    assert event.liability_intent is True


def test_ambiguous_event_routes_concept_but_does_not_invent_unknown_facts():
    event = extract_canonical_legal_event(AMBIGUOUS_SCENARIO)

    assert event.objects == ("personal_data",)
    assert "share_personal_data" in event.actions
    assert event.actor_roles == ("former_employee",)
    assert event.required_domains == ("personal_data",)
    assert event.affected_subject_count is None
    assert event.sale_amount_vnd is None
    assert event.received_payment is True
    assert "commercial_gain" in event.purposes
    assert event.conditions == ()


def test_concrete_event_is_shared_by_domain_behavior_expansion_and_facts():
    event = extract_canonical_legal_event(CONCRETE_SCENARIO)
    domains = select_legal_domains(CONCRETE_SCENARIO, event=event)
    behavior = extract_legal_behavior(CONCRETE_SCENARIO, event=event)
    _, terms = expand_legal_query(CONCRETE_SCENARIO, event=event)
    facts = extract_legal_scenario_facts(CONCRETE_SCENARIO, event=event).as_dict()

    assert domains.selected[0] == "personal_data"
    assert "advertising" in domains.selected
    assert "telecommunications" not in domains.selected
    assert behavior.actions[:3] == (
        "sell_personal_data",
        "share_personal_data",
        "retain_personal_data",
    )
    assert behavior.objects == ("personal_data",)
    assert "mua bán dữ liệu cá nhân" in terms
    assert "xử lý vi phạm dữ liệu cá nhân" in terms
    assert facts["stated_facts"]["affected_subject_count"] == 20_000
    assert facts["stated_facts"]["data_sale_proceeds_vnd"] == 30_000_000
    assert facts["stated_facts"]["company_authorization"] is False
    assert facts["stated_facts"]["data_subject_consent"] is False


def test_ambiguous_fact_state_keeps_material_elements_unknown():
    event = extract_canonical_legal_event(AMBIGUOUS_SCENARIO)
    domains = select_legal_domains(AMBIGUOUS_SCENARIO, event=event)
    facts = extract_legal_scenario_facts(AMBIGUOUS_SCENARIO, event=event).as_dict()

    assert domains.selected[0] == "personal_data"
    assert "affected_subject_count" not in facts["stated_facts"]
    assert "data_sale_proceeds_vnd" not in facts["stated_facts"]
    assert facts["stated_facts"]["received_payment"] is True
    assert "company_authorization" in facts["unknown_legal_elements"]
    assert "data_subject_consent" in facts["unknown_legal_elements"]


def test_retrieval_contract_requires_primary_rule_and_consequence():
    event = extract_canonical_legal_event(CONCRETE_SCENARIO)
    supporting_only = [
        {
            "source": "Luật Bảo vệ quyền lợi người tiêu dùng 2023",
            "text": "Chia sẻ thông tin của người tiêu dùng cho bên thứ ba.",
        }
    ]

    audit = assess_retrieval_contract(event, supporting_only)

    assert audit.missing_roles == (
        "personal_data_sale_prohibition",
        "personal_data_consequence",
    )
    rescue_query = build_retrieval_rescue_query(event, audit.missing_roles)
    assert "nghiêm cấm mua bán dữ liệu cá nhân" in rescue_query
    assert "bồi thường thiệt hại" in rescue_query


def test_retrieval_contract_is_satisfied_by_direct_articles_and_reserves_them():
    event = extract_canonical_legal_event(CONCRETE_SCENARIO)
    records = annotate_retrieval_contract_records(
        event,
        [
            {
                "chunk_id": "noise",
                "source": "Văn bản khác",
                "text": "Quảng cáo hàng hóa.",
                "score": 0.99,
            },
            {
                "chunk_id": "article-7",
                "source": "Luật Bảo vệ dữ liệu cá nhân",
                "text": "Nghiêm cấm mua, bán dữ liệu cá nhân.",
                "score": 0.20,
            },
            {
                "chunk_id": "article-8",
                "source": "Luật Bảo vệ dữ liệu cá nhân",
                "text": (
                    "Tùy theo tính chất, mức độ vi phạm có thể bị xử phạt hành "
                    "chính, truy cứu trách nhiệm hình sự và bồi thường thiệt hại."
                ),
                "score": 0.10,
            },
        ],
    )

    assert assess_retrieval_contract(event, records).missing_roles == ()
    selected = select_balanced_top_k(CONCRETE_SCENARIO, records, top_k=2)
    assert {item["chunk_id"] for item in selected} == {"article-7", "article-8"}


def test_answer_assessment_preserves_facts_liabilities_and_incident_steps():
    event = extract_canonical_legal_event(CONCRETE_SCENARIO)
    profile = extract_legal_behavior(CONCRETE_SCENARIO, event=event)
    fact_state = extract_legal_scenario_facts(
        CONCRETE_SCENARIO, event=event
    ).as_dict()
    context = [
        "[Khoản 6, Điều 7, Luật Bảo vệ dữ liệu cá nhân 2025] "
        "Nghiêm cấm mua, bán dữ liệu cá nhân.",
        "[Khoản 1, Điều 8, Luật Bảo vệ dữ liệu cá nhân 2025] "
        "Tùy theo tính chất, mức độ vi phạm có thể bị xử phạt hành chính, "
        "truy cứu trách nhiệm hình sự và bồi thường thiệt hại.",
    ]
    decisions = [
        {
            "decision_stage": "applicability",
            "decision": "KEEP",
            "level": "HIGH",
            "document": "Luật Bảo vệ dữ liệu cá nhân 2025",
            "article": "7",
            "behavior_matches": [
                ("sell_personal_data", "MATCH"),
                ("personal_data", "MATCH"),
            ],
            "missing_conditions": "Không còn điều kiện thiếu về hành vi bán.",
        },
        {
            "decision_stage": "applicability",
            "decision": "KEEP",
            "level": "HIGH",
            "document": "Luật Bảo vệ dữ liệu cá nhân 2025",
            "article": "8",
            "behavior_matches": [("sell_personal_data", "PARTIAL_MATCH")],
            "missing_conditions": "Cần xác định mức độ và hậu quả cụ thể.",
        },
    ]

    assessment = build_answer_assessment(
        query=CONCRETE_SCENARIO,
        behavior_profile=profile.as_dict(),
        retrieval_decisions=decisions,
        context_texts=context,
        retrieval_is_complete=True,
        scenario_fact_state=fact_state,
    )

    joined_facts = " ".join(assessment["matched_facts"])
    joined_steps = " ".join(assessment["next_steps"])
    assert "20.000 khách hàng" in joined_facts
    assert "30.000.000 đồng" in joined_facts
    assert "nhân viên cũ" in joined_facts
    assert "log truy cập" in joined_steps
    assert "phạm vi chủ thể bị ảnh hưởng" in joined_steps
    assert set(assessment["liability_categories"]) == {
        "employment_or_internal",
        "civil",
        "administrative",
        "criminal",
    }
    sources = build_grounded_sources(context, query=CONCRETE_SCENARIO)
    rendered = render_grounded_answer(
        "[[CITE:S1,S2]]",
        sources,
        is_complete=True,
        answer_assessment=assessment,
    )
    assert "Trách nhiệm nội bộ/lao động" in rendered
    assert "Trách nhiệm dân sự" in rendered
    assert "Trách nhiệm hành chính" in rendered
    assert "Trách nhiệm hình sự" in rendered
    assert "Điều 7, Khoản 6" in rendered
    assert "Điều 8, Khoản 1" in rendered


def test_numeric_provenance_distinguishes_user_fact_from_invented_fine():
    assert _is_query_fact_number(
        "30",
        "B nhận 30 triệu đồng từ việc bán file dữ liệu.",
        CONCRETE_SCENARIO,
    )
    assert not _is_query_fact_number(
        "30",
        "B bị phạt 30 triệu đồng.",
        CONCRETE_SCENARIO,
    )
    assert not _is_query_fact_number(
        "300",
        "B nhận 300 triệu đồng từ việc bán file dữ liệu.",
        CONCRETE_SCENARIO,
    )
