import unittest
from unittest.mock import patch

from src.agents.common.legal_relevance_filter import (
    HIGH,
    LOW,
    MEDIUM,
    REMOVE,
    WEAK_KEEP,
    filter_legal_contexts,
    prepare_generation_context,
    select_applicability_candidate_budget,
)
from src.agents.common.retrieval_provenance import normalise_provenance_record


class StaticReranker:
    def __init__(self, scores):
        self.scores = scores

    def score(self, query, passages):
        return list(self.scores)


CYBER = (
    "[Điểm d, Khoản 1, Điều 18, Luật An ninh mạng 2025] "
    "Khai thác điểm yếu, lỗ hổng bảo mật để xâm nhập trái phép hệ thống "
    "thông tin và chiếm đoạt dữ liệu."
)
PRIVACY = (
    "[Khoản 1, Điều 28, Luật Bảo vệ dữ liệu cá nhân 2025] "
    "Dữ liệu sinh trắc học của chủ thể dữ liệu phải được bảo vệ."
)
SANCTION = (
    "[Điểm a, Khoản 2, Điều 80, Nghị định 15/2020/NĐ-CP] "
    "Phạt tiền đối với hành vi truy nhập trái phép vào mạng của người khác."
)


class LegalRelevanceFilterTests(unittest.TestCase):
    def test_drops_low_article_and_keeps_high_and_medium(self):
        result = filter_legal_contexts(
            "SQL Injection khai thác lỗ hổng website tải cơ sở dữ liệu",
            [CYBER, PRIVACY, SANCTION],
            reranker=StaticReranker([0.04, 0.0003, 0.01]),
        )

        levels = {
            (decision.document, decision.article): decision.level
            for decision in result.decisions
        }
        self.assertEqual(HIGH, levels[("Luật An ninh mạng 2025", "18")])
        self.assertEqual(LOW, levels[("Luật Bảo vệ dữ liệu cá nhân 2025", "28")])
        self.assertEqual(MEDIUM, levels[("Nghị định 15/2020/NĐ-CP", "80")])
        joined = "\n".join(result.contexts)
        self.assertIn("Điều 18", joined)
        self.assertIn("Điều 80", joined)
        self.assertNotIn("Điều 28", joined)
        self.assertNotIn("sinh trắc học", joined)

    def test_scores_article_by_its_most_relevant_clause_and_keeps_whole_article(self):
        article = "\n".join(
            [
                "[Khoản 1, Điều 18, Luật An ninh mạng 2025] Quy định chung.",
                "[Điểm d, Khoản 2, Điều 18, Luật An ninh mạng 2025] "
                "Khai thác lỗ hổng để xâm nhập hệ thống.",
            ]
        )
        result = filter_legal_contexts(
            "khai thác lỗ hổng website",
            [article, PRIVACY],
            reranker=StaticReranker([0.0002, 0.03, 0.0001]),
        )

        self.assertEqual(1, result.kept_count)
        self.assertEqual(HIGH, result.decisions[0].level)
        self.assertIn("Quy định chung", result.contexts[0])
        self.assertIn("Khai thác lỗ hổng", result.contexts[0])

    def test_prepare_context_filters_graph_context_and_clears_unfiltered_copy(self):
        filtered, update = prepare_generation_context(
            {
                "query": "khai thác lỗ hổng website",
                "context_texts": [CYBER],
                "graph_context": PRIVACY,
            },
            reranker=StaticReranker([0.03, 0.0001]),
        )

        self.assertEqual("", update["graph_context"])
        self.assertTrue(update["retrieval_is_relevant"])
        self.assertNotIn("sinh trắc học", "\n".join(filtered))

    def test_all_low_produces_no_generation_context(self):
        result = filter_legal_contexts(
            "khai thác lỗ hổng website",
            [PRIVACY],
            reranker=StaticReranker([0.0001]),
        )

        self.assertEqual(LOW, result.decisions[0].level)
        self.assertEqual((), result.contexts)

    def test_low_relevance_seed_is_preserved_for_applicability(self):
        seed = normalise_provenance_record(
            {
                "chunk_id": "privacy_D28_K1",
                "text": PRIVACY,
                "score": 0.8,
                "behavior_score": 0.0,
            },
            is_seed=True,
            recursive_depth=0,
            expansion_reason="phase2_final_candidate",
        )
        result = filter_legal_contexts(
            "khai thác lỗ hổng website",
            [PRIVACY],
            reranker=StaticReranker([0.0001]),
            candidate_records=[seed],
        )

        decision = result.decisions[0]
        self.assertEqual(LOW, decision.level)
        self.assertEqual(WEAK_KEEP, decision.decision)
        self.assertTrue(decision.seed_preserved)
        self.assertFalse(decision.relevance_removed)
        self.assertIn("Điều 28", result.contexts[0])

    def test_low_relevance_non_seed_with_weak_behavior_is_removed(self):
        recursive = normalise_provenance_record(
            {
                "chunk_id": "privacy_D28_K1",
                "text": PRIVACY,
                "score": 0.1,
                "behavior_score": 0.1,
            },
            is_seed=False,
            recursive_depth=1,
            expansion_reason="incoming_reference",
        )
        result = filter_legal_contexts(
            "khai thác lỗ hổng website",
            [PRIVACY],
            reranker=StaticReranker([0.0001]),
            candidate_records=[recursive],
        )

        decision = result.decisions[0]
        self.assertEqual(REMOVE, decision.decision)
        self.assertTrue(decision.relevance_removed)
        self.assertTrue(decision.reason_removed)
        self.assertEqual((), result.contexts)

    def test_candidate_budget_selects_top_scores_and_keeps_stable_context_order(self):
        contexts = [
            f"[Điều {article}, Luật Thử nghiệm] Nội dung Điều {article}."
            for article in range(1, 8)
        ]
        records = [
            normalise_provenance_record(
                {
                    "chunk_id": f"test_D{article}_PARENT",
                    "text": context,
                    "score": score,
                },
                is_seed=True,
                recursive_depth=0,
                expansion_reason="phase2_final_candidate",
            )
            for article, context, score in zip(
                range(1, 8),
                contexts,
                (0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1),
            )
        ]
        relevance = filter_legal_contexts(
            "Nội dung thử nghiệm",
            contexts,
            reranker=StaticReranker([0.7, 0.1, 0.6, 0.5, 0.4, 0.3, 0.2]),
            candidate_records=records,
        )

        result = select_applicability_candidate_budget(
            relevance,
            candidate_records=records,
            budget=6,
        )

        joined = "\n".join(result.contexts)
        self.assertNotIn("Điều 2.", joined)
        self.assertEqual(6, len(result.contexts))
        self.assertEqual(6, len({record["article"] for record in result.records}))
        pruned = [decision for decision in result.decisions if decision.candidate_budget_pruned]
        self.assertEqual(["2"], [decision.article for decision in pruned])
        self.assertTrue(pruned[0].is_seed)
        self.assertEqual(7, pruned[0].priority_rank)
        self.assertIn("budget=6", pruned[0].reason_removed)

    def test_candidate_budget_can_be_disabled(self):
        relevance = filter_legal_contexts(
            "Nội dung thử nghiệm",
            [
                "[Điều 1, Luật Thử nghiệm] Nội dung một.",
                "[Điều 2, Luật Thử nghiệm] Nội dung hai.",
            ],
            reranker=StaticReranker([0.8, 0.4]),
        )

        result = select_applicability_candidate_budget(relevance, budget=0)

        self.assertEqual(2, len(result.contexts))
        self.assertTrue(all(decision.candidate_budget_selected for decision in result.decisions))

    def test_candidate_budget_is_unlimited_by_default(self):
        relevance = filter_legal_contexts(
            "Nội dung thử nghiệm",
            [
                "[Điều 1, Luật Thử nghiệm] Nội dung một.",
                "[Điều 2, Luật Thử nghiệm] Nội dung hai.",
            ],
            reranker=StaticReranker([0.8, 0.4]),
        )

        with patch.dict("os.environ", {}, clear=True):
            result = select_applicability_candidate_budget(relevance)

        self.assertEqual(2, len(result.contexts))
        self.assertTrue(all(decision.candidate_budget_selected for decision in result.decisions))


if __name__ == "__main__":
    unittest.main()
