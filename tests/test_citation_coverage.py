import unittest

from src.agents.agent_generation.citation_coverage import (
    citation_label,
    deduplicate_required_citations,
    find_missing_required_citations,
    format_required_citations,
    make_required_citation,
)
from src.agents.agent_generation.prompts import build_answer_prompt
from src.workflow.node_validate_answer_coverage import validate_answer_coverage_node


class CitationCoverageTests(unittest.TestCase):
    def setUp(self):
        self.required = [
            make_required_citation(
                "lu_t_giao_d_ch_i_n_t_2023_D37",
                "Điều 37 dẫn chiếu việc nhận thông điệp dữ liệu sang Điều 16.",
            ),
            make_required_citation(
                "lu_t_giao_d_ch_i_n_t_2023_D16",
                "Điều 16 quy định trực tiếp việc nhận thông điệp dữ liệu.",
            ),
        ]

    def test_citation_label_from_canonical_id(self):
        self.assertEqual("Điều 37", citation_label("lu_t_giao_d_ch_i_n_t_2023_D37"))

    def test_missing_citation_does_not_match_prefix_number(self):
        required = [make_required_citation("van_ban_D1", "required")]
        self.assertEqual(required, find_missing_required_citations("Theo Điều 10...", required))

    def test_cat2_07_requires_both_articles(self):
        answer = "Theo Điều 16 Luật Giao dịch điện tử, thông điệp được xem là đã nhận..."
        missing = find_missing_required_citations(answer, self.required)
        self.assertEqual(["Điều 37"], [item["label"] for item in missing])

        complete = "Theo Điều 37, vấn đề này được thực hiện theo Điều 16 của Luật."
        self.assertEqual([], find_missing_required_citations(complete, self.required))

    def test_prompt_places_coverage_contract_near_question(self):
        prompt = build_answer_prompt("Câu hỏi", "Ngữ cảnh", self.required)
        self.assertIn("CÁC CĂN CỨ BẮT BUỘC", prompt)
        self.assertIn("Điều 37", prompt)
        self.assertLess(prompt.index("YÊU CẦU BAO PHỦ"), prompt.index("Câu hỏi: Câu hỏi"))

    def test_validator_reports_missing_articles(self):
        result = validate_answer_coverage_node(
            {"final_response": "Chỉ viện dẫn Điều 16.", "required_citations": self.required}
        )
        self.assertEqual(["Điều 37"], [item["label"] for item in result["missing_required_citations"]])

    def test_deduplicate_preserves_first_requirement(self):
        duplicate = self.required + [make_required_citation("lu_t_giao_d_ch_i_n_t_2023_D37", "khác")]
        result = deduplicate_required_citations(duplicate)
        self.assertEqual(["Điều 37", "Điều 16"], [item["label"] for item in result])
        self.assertIn("dẫn chiếu", format_required_citations(result))


if __name__ == "__main__":
    unittest.main()
