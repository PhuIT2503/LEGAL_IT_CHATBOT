import unittest

from src.agents.common.legal_response import (
    INCOMPLETE_WARNING,
    document_priority,
    extract_legal_sources,
    finalize_legal_answer,
    sort_context_records,
)


class LegalResponseTests(unittest.TestCase):
    def test_document_priority_and_context_order(self):
        records = [
            {"text": "[Khoản 1, Điều 5, Nghị Định 15 2020 NĐ-CP] Phạt tiền", "score": 0.9},
            {"text": "[Điều 8, Luật An ninh mạng 2025] Hành vi bị nghiêm cấm", "score": 0.7},
        ]
        self.assertEqual(1, document_priority("Luật An ninh mạng 2025"))
        self.assertEqual(4, document_priority("Nghị Định 15 2020 NĐ-CP", "Phạt tiền"))
        ordered = sort_context_records(records)
        self.assertIn("Luật An ninh", ordered[0]["text"])

    def test_sources_and_incomplete_warning_are_deterministic(self):
        context = ["[Điểm a, Khoản 1, Điều 8, Luật An ninh mạng 2025] Nội dung"]
        sources = extract_legal_sources(context)
        self.assertEqual("Luật An ninh mạng 2025", sources[0]["document"])
        answer = finalize_legal_answer(
            "## Kết luận\nNội dung\n[retrieve] không được lộ",
            is_complete=False,
            context_texts=context,
        )
        self.assertIn(INCOMPLETE_WARNING, answer)
        self.assertIn("## Căn cứ pháp lý", answer)
        self.assertNotIn("[retrieve]", answer)

    def test_only_lists_articles_actually_cited_in_answer(self):
        context = [
            "[Điều 10, Luật Dữ liệu 2024] A",
            "[Điều 20, Luật Dữ liệu 2024] B",
        ]
        answer = finalize_legal_answer(
            "Áp dụng **Điều 20 Luật Dữ liệu 2024**.",
            is_complete=True,
            context_texts=context,
        )
        self.assertIn("Điều 20", answer)
        self.assertNotIn("Điều 10", answer)


if __name__ == "__main__":
    unittest.main()
