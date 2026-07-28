import unittest

from src.agents.agent_generation.answer_assessment import (
    LIKELY_VIOLATION,
    NO_MATCH,
    PARTIAL_MATCH,
    build_answer_assessment,
)
from src.agents.common.grounded_validation import (
    INSUFFICIENT_GROUNDS,
    build_grounded_sources,
    render_grounded_answer,
)


FAKE_VOICE_QUERY = (
    "Một nhân viên dùng AI giả giọng giám đốc gọi cho kế toán yêu cầu chuyển "
    "tiền gấp. Nếu sự việc xảy ra thì người đó có thể vi phạm những quy định "
    "pháp luật nào?"
)
FAKE_VOICE_CONTEXT = [
    "[Điểm g, Khoản 2, Điều 7, Luật An ninh mạng 2025] "
    "g) Sử dụng trí tuệ nhân tạo hoặc công nghệ mới để giả mạo video, hình "
    "ảnh, giọng nói của người khác trái quy định của pháp luật; tạo lập, đăng "
    "tải, phát tán thông tin quy định tại khoản 1 Điều này;"
]
PROFILE = {
    "actions": ["create_ai_deepfake"],
    "objects": [],
    "purposes": [],
    "conditions": [],
}


def applicability_decision(
    *,
    decision: str = "KEEP",
    level: str = "HIGH",
    match: str = "MATCH",
    missing: str = "Không còn điều kiện thiếu vì hành vi đã được mô tả.",
):
    return {
        "decision_stage": "applicability",
        "decision": decision,
        "level": level,
        "behavior_matches": [("create_ai_deepfake", match)],
        "missing_conditions": missing,
    }


def assessment_for(
    *,
    context=FAKE_VOICE_CONTEXT,
    decisions=None,
    profile=PROFILE,
):
    return build_answer_assessment(
        query=FAKE_VOICE_QUERY,
        behavior_profile=profile,
        retrieval_decisions=decisions or [applicability_decision()],
        context_texts=context,
        retrieval_is_complete=True,
    )


def render_fake_voice(assessment):
    sources = build_grounded_sources(FAKE_VOICE_CONTEXT, query=FAKE_VOICE_QUERY)
    return render_grounded_answer(
        "[[QUOTE:S1]]\n\n[[CITE:S1]]",
        sources,
        is_complete=True,
        answer_assessment=assessment,
    )


class AnswerAssessmentTests(unittest.TestCase):
    def test_match_has_one_positive_conclusion_and_nonempty_facts(self):
        assessment = assessment_for()
        answer = render_fake_voice(assessment)

        self.assertEqual(LIKELY_VIOLATION, assessment["status"])
        self.assertTrue(assessment["matched_facts"])
        self.assertIn("có dấu hiệu thuộc phạm vi điều chỉnh", answer.casefold())
        self.assertNotIn("chưa đủ căn cứ để kết luận dứt khoát", answer.casefold())

    def test_partial_match_keeps_positive_signal_and_states_missing_facts(self):
        missing = "Cần xác minh nội dung có thật sự giả mạo giọng nói hay không."
        assessment = assessment_for(
            decisions=[
                applicability_decision(
                    decision="WEAK_KEEP",
                    level="MEDIUM",
                    match="PARTIAL_MATCH",
                    missing=missing,
                )
            ],
            context=[
                "[Điều 9, Luật Công nghệ] Việc sử dụng nội dung số của người "
                "khác phải đáp ứng điều kiện do pháp luật quy định."
            ],
        )
        sources = build_grounded_sources(
            [
                "[Điều 9, Luật Công nghệ] Việc sử dụng nội dung số của người "
                "khác phải đáp ứng điều kiện do pháp luật quy định."
            ]
        )
        answer = render_grounded_answer(
            "[[QUOTE:S1]]\n\n[[CITE:S1]]",
            sources,
            is_complete=True,
            answer_assessment=assessment,
        )

        self.assertEqual(PARTIAL_MATCH, assessment["status"])
        self.assertIn("có dấu hiệu liên quan", answer.casefold())
        self.assertIn(missing, answer)
        self.assertNotIn("không có dấu hiệu", answer.casefold())

    def test_candidate_comment_is_not_rendered_as_a_missing_fact(self):
        assessment = assessment_for(
            decisions=[
                applicability_decision(
                    missing=(
                        "Hành vi cốt lõi là chuyển dữ liệu sang máy chủ nước "
                        "ngoài, không phải cho tổ chức quảng cáo."
                    )
                )
            ],
        )

        self.assertNotIn(
            "Hành vi cốt lõi",
            " ".join(assessment["missing_facts"]),
        )

    def test_copyright_property_right_does_not_trigger_transfer_fraud_checklist(self):
        assessment = build_answer_assessment(
            query=(
                "Dùng tác phẩm có bản quyền làm dữ liệu AI có liên quan đến "
                "quyền tài sản nào của tác giả?"
            ),
            behavior_profile={
                "actions": ["use_copyrighted_work"],
                "objects": [],
                "purposes": [],
                "conditions": [],
            },
            retrieval_decisions=[
                {
                    "decision_stage": "applicability",
                    "decision": "WEAK_KEEP",
                    "level": "MEDIUM",
                    "behavior_matches": [
                        ("use_copyrighted_work", "PARTIAL_MATCH")
                    ],
                    "missing_conditions": (
                        "Cần xác minh loại quyền tài sản được sử dụng."
                    ),
                }
            ],
            context_texts=[
                "[Điều 20, Luật Sở hữu trí tuệ] Quyền tài sản của tác giả."
            ],
            retrieval_is_complete=True,
        )

        missing = " ".join(assessment["missing_facts"])
        self.assertNotIn("Tiền hoặc tài sản đã được chuyển", missing)
        self.assertNotIn("mục đích chiếm đoạt", missing)

    def test_no_source_uses_no_match_without_inventing_a_law(self):
        assessment = assessment_for(context=[])
        answer = render_grounded_answer(
            INSUFFICIENT_GROUNDS,
            [],
            is_complete=True,
            answer_assessment=assessment,
        )

        self.assertEqual(NO_MATCH, assessment["status"])
        self.assertIn("Chưa tìm thấy căn cứ phù hợp", answer)
        self.assertNotIn("Điều 7", answer)
        self.assertNotIn("Luật An ninh mạng 2025", answer)

    def test_no_sanction_does_not_block_behavior_conclusion_or_invent_amount(self):
        assessment = assessment_for()
        answer = render_fake_voice(assessment)

        self.assertFalse(assessment["sanction_available"])
        self.assertEqual(LIKELY_VIOLATION, assessment["status"])
        self.assertIn("có dấu hiệu thuộc phạm vi điều chỉnh", answer.casefold())
        self.assertNotRegex(answer, r"\b\d{1,3}(?:[.,]\d{3})+\s*đồng\b")
        self.assertIn("chưa đủ để xác định chính xác trách nhiệm", answer)

    def test_fake_voice_snapshot_has_required_facts_structure_and_citation_limit(self):
        answer = render_fake_voice(assessment_for())

        expected_headings = (
            "## Kết luận sơ bộ",
            "## Vì sao",
            "## Quy định pháp luật liên quan",
            "## Còn cần làm rõ",
            "## Nên làm gì tiếp theo",
            "## Căn cứ pháp lý",
        )
        positions = [answer.index(heading) for heading in expected_headings]
        self.assertEqual(sorted(positions), positions)
        self.assertIn("Có sử dụng trí tuệ nhân tạo.", answer)
        self.assertIn("Có giả mạo giọng nói của người khác.", answer)
        self.assertIn("mạo danh người có thẩm quyền và yêu cầu chuyển tiền", answer)
        self.assertIn("Tiền hoặc tài sản đã được chuyển hay chưa.", answer)
        self.assertIn("Lưu giữ bản ghi cuộc gọi", answer)
        self.assertNotIn("Chưa xác định được dấu hiệu phù hợp", answer)
        self.assertNotIn("Khi đặt hai nội dung cạnh nhau", answer)
        self.assertNotIn("Các yếu tố cần đối chiếu là giọng", answer)
        self.assertNotIn("Chưa đủ căn cứ để kết luận dứt khoát", answer)
        self.assertEqual(2, answer.count("Điều 7, Khoản 2, Điểm g"))

    def test_positive_conclusion_cannot_coexist_with_global_denial(self):
        answer = render_fake_voice(assessment_for()).casefold()

        self.assertIn("có dấu hiệu thuộc phạm vi điều chỉnh", answer)
        self.assertNotIn("chưa đủ căn cứ kết luận hành vi", answer)
        self.assertNotIn("chưa đủ căn cứ để kết luận dứt khoát", answer)

    def test_visible_safe_fallback_uses_the_new_template(self):
        assessment = assessment_for()
        sources = build_grounded_sources(
            FAKE_VOICE_CONTEXT,
            query=FAKE_VOICE_QUERY,
        )
        answer = render_grounded_answer(
            INSUFFICIENT_GROUNDS,
            sources,
            is_complete=True,
            answer_assessment=assessment,
        )

        self.assertTrue(answer.startswith("## Kết luận sơ bộ"))
        self.assertIn("## Nên làm gì tiếp theo", answer)
        self.assertNotIn("## Tóm tắt tình huống", answer)
        self.assertNotIn("## Trả lời câu hỏi của người dùng", answer)

    def test_citation_budget_prefers_first_direct_keep_high_source(self):
        query = "Người bán cung cấp thông tin sai lệch có bị cấm không?"
        contexts = [
            "[Điều 14, Luật Bảo vệ người tiêu dùng] Nghĩa vụ cung cấp thông tin.",
            "[Điểm a, Khoản 1, Điều 10, Luật Bảo vệ người tiêu dùng] "
            "Cấm cung cấp thông tin sai lệch cho người tiêu dùng.",
        ]
        assessment = build_answer_assessment(
            query=query,
            behavior_profile={
                "actions": ["misleading_advertising"],
                "objects": [],
                "purposes": [],
                "conditions": [],
            },
            retrieval_decisions=[
                {
                    "decision_stage": "applicability",
                    "document": "Luật Bảo vệ người tiêu dùng",
                    "article": "14",
                    "decision": "WEAK_KEEP",
                    "level": "MEDIUM",
                    "behavior_matches": [
                        ("misleading_advertising", "PARTIAL_MATCH")
                    ],
                    "missing_conditions": "Cần làm rõ nghĩa vụ cung cấp thông tin.",
                },
                {
                    "decision_stage": "applicability",
                    "document": "Luật Bảo vệ người tiêu dùng",
                    "article": "10",
                    "decision": "KEEP",
                    "level": "HIGH",
                    "behavior_matches": [("misleading_advertising", "MATCH")],
                    "missing_conditions": "Không còn điều kiện thiếu.",
                },
            ],
            context_texts=contexts,
            retrieval_is_complete=True,
        )
        sources = build_grounded_sources(contexts, query=query)
        answer = render_grounded_answer(
            "[[QUOTE:S1]] [[CITE:S1]]\n[[QUOTE:S2]] [[CITE:S2]]",
            sources,
            is_complete=True,
            answer_assessment=assessment,
        )

        self.assertNotIn("Điều 14", answer)
        self.assertEqual(2, answer.count("Điều 10, Khoản 1, Điểm a"))

    def test_explicit_law_cross_reference_recovers_kept_target_source(self):
        query = "Sao chép một phần tác phẩm để nghiên cứu có ngoại lệ nào?"
        contexts = [
            "[Khoản 2, Điều 29, Nghị Định 17/2023/NĐ-CP] Sao chép hợp lý "
            "một phần tác phẩm phục vụ nghiên cứu quy định tại Điều 25 của "
            "Luật Sở hữu trí tuệ.",
            "[Điểm e, Khoản 1, Điều 25, Luật Sở hữu trí tuệ] Sao chép hợp lý "
            "một phần tác phẩm phục vụ nghiên cứu, học tập.",
        ]
        assessment = build_answer_assessment(
            query=query,
            behavior_profile={
                "actions": ["use_copyrighted_work"],
                "objects": [],
                "purposes": [],
                "conditions": [],
            },
            retrieval_decisions=[
                {
                    "decision_stage": "applicability",
                    "document": document,
                    "article": article,
                    "decision": "KEEP",
                    "level": "HIGH",
                    "behavior_matches": [("use_copyrighted_work", "MATCH")],
                    "missing_conditions": "Không còn điều kiện thiếu.",
                }
                for document, article in (
                    ("Nghị Định 17/2023/NĐ-CP", "29"),
                    ("Luật Sở hữu trí tuệ", "25"),
                )
            ],
            context_texts=contexts,
            retrieval_is_complete=True,
        )
        sources = build_grounded_sources(contexts, query=query)
        decree_source_id = next(
            source.source_id for source in sources if source.article == "29"
        )
        answer = render_grounded_answer(
            f"[[QUOTE:{decree_source_id}]] [[CITE:{decree_source_id}]]",
            sources,
            is_complete=True,
            answer_assessment=assessment,
        )

        self.assertEqual(2, answer.count("Điều 29, Khoản 2"))
        self.assertEqual(2, answer.count("Điều 25, Khoản 1, Điểm e"))

    def test_citation_budget_uses_existing_cross_encoder_score(self):
        query = "Quảng cáo sai sự thật về chất lượng sản phẩm."
        contexts = [
            "[Khoản 1, Điều 14, Luật Bảo vệ người tiêu dùng] Hàng hóa phải "
            "đúng nội dung đã quảng cáo.",
            "[Điểm a, Khoản 1, Điều 10, Luật Bảo vệ người tiêu dùng] Không "
            "được cung cấp thông tin sai lệch gây nhầm lẫn về sản phẩm.",
        ]
        sources = build_grounded_sources(contexts, query=query)
        score_by_article = {"14": 0.2, "10": 0.9}
        assessment = build_answer_assessment(
            query=query,
            behavior_profile={
                "actions": ["misleading_advertising"],
                "objects": [],
                "purposes": [],
                "conditions": [],
            },
            retrieval_decisions=[
                {
                    "decision_stage": "applicability",
                    "document": "Luật Bảo vệ người tiêu dùng",
                    "article": article,
                    "decision": "KEEP",
                    "level": "HIGH",
                    "behavior_matches": [("misleading_advertising", "MATCH")],
                    "missing_conditions": "Không còn điều kiện thiếu.",
                }
                for article in ("14", "10")
            ],
            context_texts=contexts,
            final_context_records=[
                {
                    "document": source.document,
                    "article": source.article,
                    "clause": source.clause,
                    "point": source.point,
                    "cross_encoder_score": score_by_article[source.article],
                }
                for source in sources
            ],
            retrieval_is_complete=True,
        )
        answer = render_grounded_answer(
            " ".join(
                f"[[QUOTE:{source.source_id}]] [[CITE:{source.source_id}]]"
                for source in sources
            ),
            sources,
            is_complete=True,
            answer_assessment=assessment,
        )

        self.assertNotIn("Điều 14", answer)
        self.assertEqual(2, answer.count("Điều 10, Khoản 1, Điểm a"))

    def test_user_citation_line_canonicalises_amendment_suffix(self):
        contexts = [
            "[Điểm a, Khoản 2, Điều 81, Nghị Định 15 2020 NĐ-CP "
            "(sửa đổi, bổ sung Nghị Định 14 2022)] Truy cập bất hợp pháp "
            "vào tài khoản của tổ chức, cá nhân."
        ]
        assessment = build_answer_assessment(
            query="Truy cập trái phép vào tài khoản.",
            behavior_profile={
                "actions": ["unauthorized_access"],
                "objects": [],
                "purposes": [],
                "conditions": [],
            },
            retrieval_decisions=[
                {
                    "decision_stage": "applicability",
                    "document": (
                        "Nghị Định 15 2020 NĐ-CP "
                        "(sửa đổi, bổ sung Nghị Định 14 2022)"
                    ),
                    "article": "81",
                    "decision": "KEEP",
                    "level": "HIGH",
                    "behavior_matches": [("unauthorized_access", "MATCH")],
                    "missing_conditions": "Không còn điều kiện thiếu.",
                }
            ],
            context_texts=contexts,
            retrieval_is_complete=True,
        )
        sources = build_grounded_sources(contexts, query="Truy cập trái phép.")
        answer = render_grounded_answer(
            f"[[CITE:{sources[0].source_id}]]",
            sources,
            is_complete=True,
            answer_assessment=assessment,
        )

        citation_section = answer.split("## Căn cứ pháp lý", 1)[1]
        self.assertIn("- Nghị Định 15 2020 NĐ-CP, Điều 81", citation_section)
        self.assertNotIn("(sửa đổi", citation_section)


if __name__ == "__main__":
    unittest.main()
