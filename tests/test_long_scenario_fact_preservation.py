import hashlib
import json

from langchain_core.messages import AIMessage

from evaluation.trace_long_scenario import LONG_SCENARIO
from src.agents.agent_generation.prompts import (
    build_generation_payload,
    canonical_generation_payload_hash,
)
from src.agents.common.legal_applicability import check_legal_applicability
from src.agents.common.legal_scenario_facts import (
    extract_legal_scenario_facts,
    filter_answered_missing_facts,
)
from src.retrieval.legal_behaviors import extract_legal_behavior


class FakeApplicabilityLLM:
    def invoke(self, prompt, **kwargs):
        del prompt, kwargs
        return AIMessage(
            content=json.dumps(
                {
                    "retrieval_gap": False,
                    "gap_reason": "",
                    "decisions": [
                        {
                            "id": "A1",
                            "scope": "Ứng dụng công nghệ thông tin giả mạo để phục vụ giao kết hợp đồng.",
                            "behavior_matches": [
                                {
                                    "behavior_key": "create_ai_deepfake",
                                    "match": "MATCH",
                                },
                                {
                                    "behavior_key": "synthetic_media",
                                    "match": "MATCH",
                                },
                            ],
                            "applicability": "HIGH",
                            "explanation": (
                                "Tình huống có AI và giả giọng, trùng yếu tố công nghệ "
                                "và giả mạo được quy định trong candidate."
                            ),
                            "missing_conditions": (
                                "Cần làm rõ tiền đã chuyển chưa và có giao kết hợp đồng hay không."
                            ),
                        }
                    ],
                },
                ensure_ascii=False,
            )
        )


CONTRACT_CONTEXT = """
[Khoản 12, Điều 33, Nghị định 15/2020/NĐ-CP] Phạt tiền đối với hành vi sử dụng
phần mềm, ứng dụng công nghệ thông tin để giả mạo thông tin, hình ảnh của cá
nhân, tổ chức phục vụ việc giao kết hợp đồng, hợp đồng theo mẫu, điều kiện giao
dịch chung.
""".strip()


def test_a_stated_fact_preservation():
    facts = extract_legal_scenario_facts(LONG_SCENARIO).as_dict()
    stated = facts["stated_facts"]

    assert stated["used_ai"] is True
    assert stated["voice_impersonation"] is True
    assert stated["impersonated_person_role"] == "giám đốc"
    assert stated["called_accountant"] is True
    assert stated["requested_transfer"] is True
    assert stated["requested_amount_vnd"] == 300_000_000
    assert stated["transfer_executed"] is True
    assert stated["transferred_amount_vnd"] == 300_000_000
    assert stated["frozen_amount_vnd"] == 180_000_000
    assert stated["onward_transferred_amount_vnd"] == 120_000_000
    assert stated["act_was_executed"] is True
    assert stated["actor_claimed_experiment"] is True
    assert stated["actor_claimed_money_not_used"] is True


def test_b_no_answered_fact_in_missing_list():
    fact_state = extract_legal_scenario_facts(LONG_SCENARIO).as_dict()
    filtered, missing_keys = filter_answered_missing_facts(
        [
            "Tiền hoặc tài sản đã được chuyển hay chưa.",
            "Hành vi mới ở giai đoạn chuẩn bị hay đã được thực hiện.",
            "Số tiền chuyển là bao nhiêu.",
            "Cần làm rõ đầy đủ chứng cứ về ý định chiếm đoạt.",
        ],
        fact_state,
    )

    assert "money_transfer_status" not in missing_keys
    assert "preparation_or_execution" not in missing_keys
    assert "transferred_amount" not in missing_keys
    assert not any("đã được chuyển hay chưa" in value for value in filtered)
    assert any("ý định chiếm đoạt" in value for value in filtered)


def test_c_inference_separation():
    facts = extract_legal_scenario_facts(LONG_SCENARIO).as_dict()

    assert "final_unrecoverable_loss" not in facts["stated_facts"]
    assert "appropriation_intent" not in facts["stated_facts"]
    assert facts["supported_inferences"]["possible_appropriation_intent"] is True
    assert "final_unrecoverable_loss" in facts["unknown_legal_elements"]


def test_d_contract_provision_is_not_applicable_without_contract_fact():
    fact_state = extract_legal_scenario_facts(LONG_SCENARIO).as_dict()
    result = check_legal_applicability(
        LONG_SCENARIO,
        [CONTRACT_CONTEXT],
        llm_client=FakeApplicabilityLLM(),
        behavior_profile=extract_legal_behavior(LONG_SCENARIO),
        scenario_fact_state=fact_state,
    )

    decision = result.decisions[0]
    assert decision.element_applicability == "NOT_APPLICABLE"
    assert "contract_conclusion" in decision.missing_required_elements
    assert decision.decision == "REMOVE"
    assert result.contexts == ()


def test_e_frozen_payload_hash_ignores_only_model_configuration():
    fact_state = extract_legal_scenario_facts(LONG_SCENARIO).as_dict()
    common = dict(
        query=LONG_SCENARIO,
        context_text=CONTRACT_CONTEXT,
        is_complete=False,
        scenario_fact_state=fact_state,
        behavior_profile=extract_legal_behavior(LONG_SCENARIO).as_dict(),
        applicability_results=[],
        answer_assessment={"status": "PARTIAL_MATCH"},
    )
    qwen = build_generation_payload(
        **common,
        model_config={"model": "qwen2.5:7b"},
    )
    gpt = build_generation_payload(
        **common,
        model_config={"model": "gpt-4o-mini"},
    )

    assert canonical_generation_payload_hash(qwen) == canonical_generation_payload_hash(gpt)
    raw_qwen = hashlib.sha256(
        json.dumps(qwen, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    raw_gpt = hashlib.sha256(
        json.dumps(gpt, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    assert raw_qwen != raw_gpt


def test_long_scenario_behavior_alias_is_preserved():
    profile = extract_legal_behavior(LONG_SCENARIO)

    assert "create_ai_deepfake" in profile.actions
    assert "synthetic_media" in profile.objects
