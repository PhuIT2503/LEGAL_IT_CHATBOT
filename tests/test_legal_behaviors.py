import unittest

from src.agents.common.retrieval_ranking import (
    filter_behaviorally_relevant,
    rerank_context_records_with_behavior,
)
from src.retrieval.legal_behaviors import (
    extract_legal_behavior,
    score_behavior_relevance,
)


class CapturingReranker:
    def __init__(self, scores):
        self.scores = list(scores)
        self.query = ""
        self.passages = []

    def score(self, query, passages):
        self.query = query
        self.passages = list(passages)
        return self.scores


class LegalBehaviorTests(unittest.TestCase):
    def test_extracts_deepfake_behavior_card(self):
        profile = extract_legal_behavior(
            "Deepfake AI dùng hình ảnh người nổi tiếng để quảng cáo"
        )

        self.assertIn("create_ai_deepfake", profile.actions)
        self.assertIn("use_person_likeness", profile.actions)
        self.assertIn("synthetic_media", profile.objects)
        self.assertIn("person_likeness", profile.objects)
        self.assertIn("advertising", profile.purposes)

    def test_deepfake_provision_scores_above_commercial_recording(self):
        profile = extract_legal_behavior(
            "Deepfake AI dùng hình ảnh người nổi tiếng để quảng cáo"
        )
        direct = score_behavior_relevance(
            profile,
            "Sử dụng trí tuệ nhân tạo hoặc công nghệ mới để giả mạo video, "
            "hình ảnh, giọng nói của người khác.",
        )
        keyword_only = score_behavior_relevance(
            profile,
            "Sử dụng bản ghi âm, ghi hình đã công bố nhằm mục đích thương mại "
            "trong chương trình quảng cáo.",
        )

        self.assertGreaterEqual(direct.score, 0.35)
        self.assertLess(keyword_only.score, 0.18)
        self.assertGreater(direct.score, keyword_only.score)

    def test_behavior_gate_drops_keyword_match_even_when_ce_prefers_it(self):
        query = "Deepfake AI dùng hình ảnh người nổi tiếng để quảng cáo"
        profile = extract_legal_behavior(query)
        reranker = CapturingReranker([0.2, 0.9])
        records = [
            {
                "chunk_id": "cyber_D7_K1",
                "source": "Luật An ninh mạng",
                "text": "Dùng công nghệ mới để giả mạo video, hình ảnh, giọng nói.",
                "original_rank": 2,
            },
            {
                "chunk_id": "ip_D33_K1",
                "source": "Luật Sở hữu trí tuệ",
                "text": "Dùng bản ghi âm, ghi hình trong chương trình quảng cáo.",
                "original_rank": 1,
            },
        ]

        ranked = rerank_context_records_with_behavior(
            query,
            records,
            behavior_profile=profile,
            reranker=reranker,
        )
        filtered, removed = filter_behaviorally_relevant(
            ranked,
            behavior_profile=profile,
            minimum=0.18,
            activation=0.35,
        )

        self.assertEqual(1, removed)
        self.assertEqual(["cyber_D7_K1"], [item["chunk_id"] for item in filtered])
        self.assertIn("Hành vi:", reranker.query)

    def test_sql_injection_matches_exploit_and_data_theft(self):
        profile = extract_legal_behavior(
            "SQL Injection, khai thác lỗ hổng website, tải xuống cơ sở dữ liệu"
        )
        direct = score_behavior_relevance(
            profile,
            "Khai thác điểm yếu, lỗ hổng bảo mật để chiếm đoạt thông tin.",
        )
        unrelated = score_behavior_relevance(
            profile,
            "Quy định quyền của chủ thể đối với dữ liệu sinh trắc học.",
        )

        self.assertIn("unauthorized_access", profile.actions)
        self.assertIn("exploit_vulnerability", profile.actions)
        self.assertIn("extract_or_download_data", profile.actions)
        self.assertGreaterEqual(direct.score, 0.35)
        self.assertLess(unrelated.score, 0.18)

    def test_gate_is_conservative_when_no_candidate_matches_taxonomy(self):
        profile = extract_legal_behavior("SQL Injection vào website")
        records = [
            {"chunk_id": "unknown", "behavior_score": 0.1},
            {"chunk_id": "another", "behavior_score": 0.0},
        ]

        filtered, removed = filter_behaviorally_relevant(
            records,
            behavior_profile=profile,
            minimum=0.18,
            activation=0.35,
        )

        self.assertEqual(records, filtered)
        self.assertEqual(0, removed)


if __name__ == "__main__":
    unittest.main()
