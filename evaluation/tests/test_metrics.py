import unittest

from evaluation.metrics import (
    applicability_accuracy,
    citation_accuracy,
    citations_from_answer,
    expected_legal_units,
    hallucinated_citations,
    legal_recall_at,
    reciprocal_rank,
    recursive_metrics,
    set_precision_recall,
)


class MetricsTest(unittest.TestCase):
    def setUp(self):
        self.case = {
            "expected_documents": ["Luật An ninh mạng 2025"],
            "expected_articles": ["13"],
            "expected_clauses": ["3"],
            "expected_points": ["h"],
        }
        self.expected = expected_legal_units(self.case)
        self.correct = {
            "document": "Luật An ninh mạng 2025.docx",
            "article": "13",
            "clause": "3",
            "point": "h",
        }
        self.wrong = {
            "document": "Luật Bảo vệ dữ liệu cá nhân 2025",
            "article": "25",
        }

    def test_parallel_ground_truth_arrays_reuse_single_document(self):
        case = {
            "expected_documents": ["Luật Giao dịch điện tử 2023"],
            "expected_articles": ["15", "16", "17"],
            "expected_clauses": ["", "", ""],
            "expected_points": ["", "", ""],
        }
        units = expected_legal_units(case)
        self.assertEqual([item.article for item in units], ["15", "16", "17"])
        self.assertTrue(all(item.document == case["expected_documents"][0] for item in units))

    def test_candidate_coordinates_fall_back_to_chunk_id(self):
        from evaluation.metrics import legal_unit_from_record

        unit = legal_unit_from_record(
            {
                "chunk_id": "Lu_t_An_ninh_m_ng_2025_D13_K3_Ph",
                "source": "Luật An ninh mạng 2025.docx",
                "metadata": {},
            }
        )
        self.assertEqual(
            (unit.document, unit.article, unit.clause, unit.point),
            ("Luật An ninh mạng 2025", "13", "3", "h"),
        )

    def test_recall_at_k_and_mrr(self):
        ranked = [self.wrong, self.correct]
        self.assertEqual(legal_recall_at(ranked, self.expected, 1), 0.0)
        self.assertEqual(legal_recall_at(ranked, self.expected, 5), 1.0)
        self.assertEqual(reciprocal_rank(ranked, self.expected), 0.5)

    def test_precision_recall_is_macro_safe_for_empty_sets(self):
        self.assertEqual(set_precision_recall([], []), (1.0, 1.0))
        self.assertEqual(set_precision_recall([], ["cybersecurity"]), (0.0, 0.0))
        self.assertEqual(set_precision_recall(["cybersecurity"], []), (0.0, 1.0))

    def test_recursive_precision_and_noise(self):
        records = [
            {**self.correct, "is_seed": True},
            {**self.correct, "is_seed": False, "chunk_id": "recursive-good"},
            {**self.wrong, "is_seed": False, "chunk_id": "recursive-noise"},
        ]
        precision, noise = recursive_metrics(records, self.expected)
        self.assertEqual(precision, 0.5)
        self.assertEqual(noise, 0.5)

    def test_applicability_accuracy_scores_keep_and_drop(self):
        candidates = [self.correct, self.wrong]
        self.assertEqual(applicability_accuracy(candidates, [self.correct], self.expected), 1.0)
        self.assertEqual(applicability_accuracy(candidates, [self.wrong], self.expected), 0.0)

    def test_rendered_citation_parser_and_hallucination(self):
        answer = (
            "(Căn cứ: **Luật An ninh mạng 2025**, Điều 13, Khoản 3, Điểm h)\n"
            "- **Nghị định 15/2020/NĐ-CP**, Điều 80"
        )
        citations = citations_from_answer(answer)
        self.assertEqual(len(citations), 2)
        hallucinated = hallucinated_citations(citations, [self.correct])
        self.assertEqual(len(hallucinated), 1)
        self.assertEqual(hallucinated[0]["article"], "80")

    def test_citation_accuracy_penalises_missing_and_wrong_citations(self):
        second_expected = expected_legal_units(
            {
                "expected_documents": ["Luật An ninh mạng 2025", "Nghị Định 15 2020 NĐ-CP"],
                "expected_articles": ["13", "81"],
                "expected_clauses": ["3", ""],
                "expected_points": ["h", ""],
            }
        )
        self.assertAlmostEqual(citation_accuracy([self.correct], second_expected), 2 / 3)
        self.assertEqual(citation_accuracy([self.wrong], self.expected), 0.0)

    def test_citation_parser_supports_filename_style_documents(self):
        citations = citations_from_answer(
            "- **2023_361 + 362_11-VBHN-VPQH**, Điều 25"
        )
        self.assertEqual(citations[0]["document"], "2023_361 + 362_11-VBHN-VPQH")
        self.assertEqual(citations[0]["article"], "25")

    def test_citation_parser_does_not_treat_analysis_as_document(self):
        citations = citations_from_answer(
            "- Phân tích: Hành vi vi phạm điều kiện theo Khoản 1, Điều 10, Nghị định 53/2022/NĐ-CP."
        )
        self.assertEqual(citations, [])


if __name__ == "__main__":
    unittest.main()
