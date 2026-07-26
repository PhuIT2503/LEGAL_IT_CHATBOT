import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.agents.common.legal_applicability import (
    HIGH,
    KEEP,
    LOW,
    MEDIUM,
    REMOVE,
    WEAK_KEEP,
    check_legal_applicability,
)
from src.agents.common.legal_relevance_filter import prepare_generation_context
from src.agents.common.retrieval_provenance import normalise_provenance_record
from src.retrieval.legal_behaviors import BehaviorProfile


IMAGE_RIGHT = (
    "[Khoản 1, Điều 32, Bộ luật Dân sự 2015] Việc sử dụng hình ảnh của cá nhân "
    "phải được người đó đồng ý; sử dụng vì mục đích thương mại phải trả thù lao, "
    "trừ trường hợp các bên có thỏa thuận khác."
)
RECORDING_RIGHT = (
    "[Khoản 1, Điều 33, Luật Sở hữu trí tuệ] Tổ chức, cá nhân sử dụng bản ghi âm, "
    "ghi hình đã công bố nhằm mục đích thương mại phải trả tiền nhuận bút, thù lao."
)
ADVERTISING = (
    "[Khoản 1, Điều 19, Luật Quảng cáo] Nội dung quảng cáo phải bảo đảm trung thực, "
    "chính xác, rõ ràng và không gây thiệt hại cho người sản xuất, kinh doanh."
)


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.prompt = ""
        self.tags = []
        self.kwargs = []

    def invoke(self, prompt, tag=None, **kwargs):
        self.prompt = prompt
        self.tags.append(tag)
        self.kwargs.append(kwargs)
        content = self.payload if isinstance(self.payload, str) else json.dumps(
            self.payload, ensure_ascii=False
        )
        return SimpleNamespace(content=content)


class SequenceLLM(FakeLLM):
    def __init__(self, payloads):
        super().__init__(payloads[0])
        self.payloads = list(payloads)

    def invoke(self, prompt, tag=None, **kwargs):
        self.prompt = prompt
        self.tags.append(tag)
        self.kwargs.append(kwargs)
        payload = self.payloads.pop(0)
        return SimpleNamespace(content=json.dumps(payload, ensure_ascii=False))


class StaticReranker:
    def __init__(self, scores):
        self.scores = scores

    def score(self, query, passages):
        return list(self.scores)


def deepfake_payload(*, retrieval_gap=False):
    def matches(**overrides):
        defaults = {
            "create_ai_deepfake": "NOT_MATCH",
            "use_person_likeness": "NOT_MATCH",
            "synthetic_media": "NOT_MATCH",
            "person_likeness": "NOT_MATCH",
            "advertising": "NOT_MATCH",
        }
        defaults.update(overrides)
        return [
            {"behavior_key": key, "match": value}
            for key, value in defaults.items()
        ]

    return {
        "retrieval_gap": retrieval_gap,
        "gap_reason": "Thiếu căn cứ điều chỉnh kỹ thuật giả mạo" if retrieval_gap else "",
        "decisions": [
            {
                "id": "A1",
                "scope": "Quyền của cá nhân đối với việc người khác sử dụng hình ảnh của mình.",
                "behavior_matches": matches(
                    create_ai_deepfake="PARTIAL_MATCH",
                    use_person_likeness="MATCH",
                    synthetic_media="PARTIAL_MATCH",
                    person_likeness="MATCH",
                    advertising="MATCH",
                ),
                "applicability": "HIGH",
                "explanation": "Tình huống sử dụng diện mạo nhận diện được của một cá nhân cho quảng cáo, đúng đối tượng là hình ảnh cá nhân và mục đích thương mại mà nguồn mô tả.",
                "missing_conditions": "Cần xác minh người nổi tiếng đã đồng ý hoặc có thỏa thuận về thù lao hay chưa.",
            },
            {
                "id": "A2",
                "scope": "Khai thác bản ghi âm hoặc bản ghi hình đã công bố nhằm mục đích thương mại.",
                "behavior_matches": matches(synthetic_media="PARTIAL_MATCH", advertising="MATCH"),
                "applicability": "LOW",
                "explanation": "Nguồn giả định có hành vi sử dụng một bản ghi đã công bố, còn tình huống chỉ mô tả tạo video giả mới và không nói đã khai thác bản ghi thuộc quyền liên quan.",
                "missing_conditions": "Thiếu dữ kiện về việc sử dụng một bản ghi âm hoặc ghi hình đã công bố.",
            },
            {
                "id": "A3",
                "scope": "Yêu cầu về tính trung thực và chính xác của nội dung quảng cáo.",
                "behavior_matches": matches(advertising="MATCH"),
                "applicability": "MEDIUM",
                "explanation": "Video được dùng làm nội dung quảng cáo nên yêu cầu về tính trung thực có thể điều chỉnh phần thông tin quảng cáo, nhưng nguồn không điều chỉnh riêng kỹ thuật tạo deepfake.",
                "missing_conditions": "Cần xác định nội dung giả mạo đã làm thông tin quảng cáo sai lệch hoặc gây thiệt hại hay chưa.",
            },
        ],
    }


class LegalApplicabilityTests(unittest.TestCase):
    def test_deepfake_drops_commercial_recording_article_but_keeps_direct_and_partial_rules(self):
        llm = FakeLLM(deepfake_payload())
        result = check_legal_applicability(
            "AI tạo video deepfake của người nổi tiếng để quảng cáo.",
            [IMAGE_RIGHT, RECORDING_RIGHT, ADVERTISING],
            llm_client=llm,
        )

        levels = {decision.article: decision.level for decision in result.decisions}
        self.assertEqual(HIGH, levels["32"])
        self.assertEqual(LOW, levels["33"])
        self.assertEqual(MEDIUM, levels["19"])
        kept = "\n".join(result.contexts)
        self.assertIn("Điều 32", kept)
        self.assertIn("Điều 19", kept)
        self.assertNotIn("Điều 33", kept)
        self.assertEqual(["legal_applicability"], llm.tags)
        self.assertEqual(800, llm.kwargs[0]["max_completion_tokens"])
        self.assertEqual(
            {"type": "json_object"},
            llm.kwargs[0]["response_format"],
        )

    def test_strong_behavior_recovers_candidate_downgraded_for_weak_explanation(self):
        payload = deepfake_payload()
        payload["decisions"] = [
            {
                "id": "A1",
                "scope": "Quyền hình ảnh của cá nhân trong giao dịch thương mại.",
                "behavior_matches": [
                    {"behavior_key": "use_person_likeness", "match": "MATCH"},
                    {"behavior_key": "person_likeness", "match": "MATCH"},
                    {"behavior_key": "advertising", "match": "MATCH"},
                ],
                "applicability": "HIGH",
                "explanation": "Tình tiết thuộc đúng nhóm hoạt động.",
                "missing_conditions": "Chưa biết cá nhân đã đồng ý hay chưa.",
            }
        ]
        result = check_legal_applicability(
            "Dùng hình ảnh người nổi tiếng để quảng cáo.",
            [IMAGE_RIGHT],
            llm_client=FakeLLM(payload),
        )

        self.assertEqual(LOW, result.decisions[0].level)
        self.assertEqual(WEAK_KEEP, result.decisions[0].decision)
        self.assertTrue(result.decisions[0].behavior_preserved)
        self.assertIn("Điều 32", result.contexts[0])

    def test_rejects_applicability_that_invents_behavior_not_in_card(self):
        payload = deepfake_payload()
        payload["decisions"] = [payload["decisions"][0]]
        payload["decisions"][0]["behavior_matches"].append(
            {"behavior_key": "delete_personal_data", "match": "MATCH"}
        )
        result = check_legal_applicability(
            "AI tạo video deepfake của người nổi tiếng để quảng cáo.",
            [IMAGE_RIGHT],
            llm_client=FakeLLM(payload),
        )

        self.assertEqual(LOW, result.decisions[0].level)
        self.assertEqual("INVALID", result.decisions[0].validation_status)
        self.assertIn("delete_personal_data", result.decisions[0].reason_rejected)

    def test_rejects_high_when_behavior_score_is_near_zero(self):
        payload = deepfake_payload()
        payload["decisions"] = [{**payload["decisions"][0], "id": "A1"}]
        result = check_legal_applicability(
            "AI tạo video deepfake của người nổi tiếng để quảng cáo.",
            ["[Khoản 1, Điều 99, Luật Thử nghiệm] Quy định về lưu trữ hồ sơ kế toán."],
            llm_client=FakeLLM(payload),
        )

        self.assertEqual(LOW, result.decisions[0].level)
        self.assertIn("behavior_score", result.decisions[0].reason_rejected)

    def test_rejects_recursive_candidate_that_outranks_seed_with_different_action(self):
        payload = deepfake_payload()
        payload["decisions"] = [{**payload["decisions"][2], "id": "A1", "applicability": "HIGH"}]
        seed = normalise_provenance_record(
            {
                "chunk_id": "law_D7_K2_Pg",
                "text": "[Điểm g, Khoản 2, Điều 7, Luật Thử nghiệm] Quy định về deepfake.",
                "score": 0.5,
                "behavior_score": 0.2,
                "matched_behavior_actions": ["create_ai_deepfake"],
            },
            is_seed=True,
            recursive_depth=0,
            expansion_reason="phase2_final_candidate",
        )
        recursive = normalise_provenance_record(
            {
                "chunk_id": "law_D19_PARENT",
                "text": ADVERTISING,
                "score": 0.9,
                "behavior_score": 0.9,
                "matched_behavior_actions": ["delete_personal_data"],
            },
            is_seed=False,
            recursive_depth=1,
            expansion_reason="incoming reference",
        )
        result = check_legal_applicability(
            "AI tạo video deepfake của người nổi tiếng để quảng cáo.",
            [ADVERTISING],
            llm_client=FakeLLM(payload),
            candidate_records=[seed, recursive],
        )

        self.assertEqual(LOW, result.decisions[0].level)
        self.assertEqual(REMOVE, result.decisions[0].decision)
        self.assertIn("không cùng matched behavior action", result.decisions[0].reason_rejected)

    def test_high_is_valid_for_object_only_behavior_card(self):
        payload = {
            "retrieval_gap": False,
            "gap_reason": "",
            "decisions": [
                {
                    "id": "A1",
                    "scope": "Biện pháp phát hiện và loại bỏ mã độc trong hệ thống thông tin.",
                    "behavior_matches": [
                        {"behavior_key": "website_or_information_system", "match": "MATCH"}
                    ],
                    "applicability": "HIGH",
                    "explanation": "Nguồn quy định biện pháp bảo vệ chính hệ thống thông tin là đối tượng được nêu trong câu hỏi và áp dụng trực tiếp cho chủ quản hệ thống.",
                    "missing_conditions": "Không còn điều kiện thiếu vì tình huống đã xác định hệ thống thông tin.",
                }
            ],
        }
        record = normalise_provenance_record(
            {
                "chunk_id": "law_D10_K1",
                "text": "[Khoản 1, Điều 10, Nghị định 53] Chủ quản phải kiểm tra hệ thống thông tin để phát hiện và loại bỏ mã độc.",
                "score": 0.9,
                "behavior_score": 0.65,
            },
            is_seed=True,
            recursive_depth=0,
            expansion_reason="phase2_final_candidate",
        )
        result = check_legal_applicability(
            "Hệ thống thông tin phải thực hiện biện pháp nào để loại bỏ mã độc?",
            [record["text"]],
            llm_client=FakeLLM(payload),
            behavior_profile=BehaviorProfile(
                actions=(),
                objects=("website_or_information_system",),
                purposes=(),
                conditions=(),
            ),
            candidate_records=[record],
        )

        decision = result.decisions[0]
        self.assertEqual(HIGH, decision.level)
        self.assertEqual(KEEP, decision.decision)
        self.assertEqual("VALID", decision.validation_status)
        self.assertTrue(decision.seed_survived)
        self.assertFalse(decision.seed_removed)
        self.assertTrue(result.contexts)

    def test_high_is_valid_for_purpose_only_and_condition_only_cards(self):
        scenarios = (
            (
                BehaviorProfile((), (), ("advertising",), ()),
                "advertising",
                "Mục đích quảng cáo thương mại của nội dung được công bố.",
            ),
            (
                BehaviorProfile((), (), (), ("without_consent",)),
                "without_consent",
                "Điều kiện xử lý dữ liệu khi chưa có sự đồng ý hợp lệ.",
            ),
        )
        for profile, key, scope in scenarios:
            with self.subTest(key=key):
                payload = {
                    "retrieval_gap": False,
                    "gap_reason": "",
                    "decisions": [
                        {
                            "id": "A1",
                            "scope": scope,
                            "behavior_matches": [
                                {"behavior_key": key, "match": "MATCH"}
                            ],
                            "applicability": "HIGH",
                            "explanation": "Nội dung điều luật điều chỉnh trực tiếp đúng mục đích hoặc điều kiện đã được xác định trong câu hỏi của người dùng.",
                            "missing_conditions": "Không còn điều kiện thiếu vì tình huống đã thể hiện dữ kiện cần kiểm tra.",
                        }
                    ],
                }
                result = check_legal_applicability(
                    "Tình huống kiểm thử Applicability.",
                    ["[Khoản 1, Điều 10, Luật Thử nghiệm] Quy định áp dụng cho tình huống kiểm thử này."],
                    llm_client=FakeLLM(payload),
                    behavior_profile=profile,
                )

                self.assertEqual(HIGH, result.decisions[0].level)
                self.assertEqual(KEEP, result.decisions[0].decision)
                self.assertEqual("VALID", result.decisions[0].validation_status)

    def test_empty_behavior_card_rejects_invented_key_but_weakly_preserves_seed(self):
        payload = {
            "retrieval_gap": False,
            "gap_reason": "",
            "decisions": [
                {
                    "id": "A1",
                    "scope": "Giá trị pháp lý của thông báo điện tử trong giao kết hợp đồng.",
                    "behavior_matches": [
                        {"behavior_key": "use_electronic_notice", "match": "MATCH"}
                    ],
                    "applicability": "HIGH",
                    "explanation": "Điều luật trực tiếp quy định giá trị của thông báo điện tử được sử dụng trong quá trình giao kết và thực hiện hợp đồng giữa các bên.",
                    "missing_conditions": "Không còn điều kiện thiếu vì câu hỏi đã xác định thông báo điện tử trong hợp đồng.",
                }
            ],
        }
        text = "[Điều 38, Luật Giao dịch điện tử 2023] Thông báo dưới dạng thông điệp dữ liệu có giá trị pháp lý trong giao kết hợp đồng."
        seed = normalise_provenance_record(
            {"chunk_id": "law_D38_PARENT", "text": text, "score": 1.0},
            is_seed=True,
            recursive_depth=0,
            expansion_reason="phase2_final_candidate",
        )
        result = check_legal_applicability(
            "Thông báo điện tử trong hợp đồng có giá trị pháp lý thế nào?",
            [text],
            llm_client=FakeLLM(payload),
            behavior_profile=BehaviorProfile((), (), (), ()),
            candidate_records=[seed],
        )

        decision = result.decisions[0]
        self.assertEqual("INVALID", decision.validation_status)
        self.assertIn("use_electronic_notice", decision.reason_rejected)
        self.assertEqual(WEAK_KEEP, decision.decision)
        self.assertTrue(decision.seed_survived)
        self.assertTrue(result.contexts)

    def test_malformed_checker_output_fails_closed(self):
        result = check_legal_applicability(
            "AI tạo video deepfake của người nổi tiếng để quảng cáo.",
            [RECORDING_RIGHT],
            llm_client=FakeLLM("not-json"),
        )

        self.assertEqual(LOW, result.decisions[0].level)
        self.assertTrue(result.retrieval_gap)
        self.assertEqual((), result.contexts)

    def test_missing_candidate_decision_is_requested_again(self):
        payload = deepfake_payload()
        first = {
            "retrieval_gap": False,
            "gap_reason": "",
            "decisions": [payload["decisions"][0]],
        }
        recording = {**payload["decisions"][1], "id": "A2"}
        second = {
            "retrieval_gap": False,
            "gap_reason": "",
            "decisions": [recording],
        }
        llm = SequenceLLM([first, second])

        with patch.dict(os.environ, {"LEGAL_APPLICABILITY_REPAIR_ATTEMPTS": "1"}):
            result = check_legal_applicability(
                "AI tạo video deepfake của người nổi tiếng để quảng cáo.",
                [IMAGE_RIGHT, RECORDING_RIGHT],
                llm_client=llm,
            )

        self.assertEqual([HIGH, LOW], [decision.level for decision in result.decisions])
        self.assertEqual(2, len(llm.tags))
        self.assertFalse(result.retrieval_gap)

    def test_missing_candidate_is_not_retried_by_default(self):
        payload = deepfake_payload()
        first = {
            "retrieval_gap": False,
            "gap_reason": "",
            "decisions": [payload["decisions"][0]],
        }
        llm = SequenceLLM([first])

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LEGAL_APPLICABILITY_REPAIR_ATTEMPTS", None)
            result = check_legal_applicability(
                "AI tạo video deepfake của người nổi tiếng để quảng cáo.",
                [IMAGE_RIGHT, RECORDING_RIGHT],
                llm_client=llm,
            )

        self.assertEqual(1, len(llm.tags))
        self.assertTrue(result.retrieval_gap)

    def test_generation_preparation_runs_relevance_then_applicability(self):
        payload = deepfake_payload()
        # Relevance filter ưu tiên Luật trước Bộ luật, nên candidate order sau
        # cổng đầu là Điều 33, Điều 19, rồi Điều 32. Applicability phải bám ID
        # của chính nội dung được gửi, không bám thứ tự đầu vào ban đầu.
        image, recording, advertising = payload["decisions"]
        payload["decisions"] = [
            {**recording, "id": "A1"},
            {**advertising, "id": "A2"},
            {**image, "id": "A3"},
        ]
        llm = FakeLLM(payload)
        raw_records = []
        for article, text, score in (
            ("32", IMAGE_RIGHT, 0.9),
            ("33", RECORDING_RIGHT, 0.8),
            ("19", ADVERTISING, 0.7),
        ):
            raw_records.append(
                normalise_provenance_record(
                    {
                        "chunk_id": f"test_D{article}_K1",
                        "parent_id": f"test_D{article}_PARENT",
                        "text": text,
                        "score": score,
                        "behavior_score": 0.5,
                        "cross_encoder_score": score / 2,
                    },
                    is_seed=True,
                    recursive_depth=0,
                    expansion_reason="phase2_final_candidate",
                )
            )
        filtered, update = prepare_generation_context(
            {
                "query": "AI tạo video deepfake của người nổi tiếng để quảng cáo.",
                "context_texts": [IMAGE_RIGHT, RECORDING_RIGHT, ADVERTISING],
                "context_records": raw_records,
                "behavior_profile": {
                    "actions": ["create_ai_deepfake", "use_person_likeness"],
                    "objects": ["synthetic_media", "person_likeness"],
                    "purposes": ["advertising"],
                    "conditions": [],
                },
                "graph_context": "",
                "retrieval_is_complete": True,
            },
            reranker=StaticReranker([0.9, 0.8, 0.7]),
            llm_client=llm,
        )

        joined = "\n".join(filtered)
        self.assertNotIn("Điều 33", joined)
        self.assertTrue(update["retrieval_is_relevant"])
        self.assertTrue(update["retrieval_is_complete"])
        self.assertEqual({"32", "19"}, {record["article"] for record in update["context_records"]})
        self.assertTrue(all(record["parent_id"] for record in update["context_records"]))
        self.assertTrue(all(record["is_seed"] for record in update["context_records"]))


if __name__ == "__main__":
    unittest.main()
