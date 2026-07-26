import unittest
from unittest.mock import patch

try:
    import qdrant_client  # noqa: F401
    from src.agents.agent_retrieval.node_hybrid_search import hybrid_search_node
except ModuleNotFoundError:
    hybrid_search_node = None


class FakePoint:
    def __init__(self, chunk_id, dieu_id, source, text, score):
        self.score = score
        self.payload = {
            "id": chunk_id,
            "dieu_id": dieu_id,
            "van_ban_id": source.replace(" ", "_"),
            "content": text,
            "metadata": {"source": source},
        }


class CyberReranker:
    def score(self, query, passages):
        scores = []
        for passage in passages:
            folded = passage.casefold()
            if "khai thác điểm yếu, lỗ hổng" in folded:
                scores.append(0.95)
            elif "tấn công mạng" in folded:
                scores.append(0.80)
            else:
                scores.append(0.05)
        return scores


@unittest.skipIf(hybrid_search_node is None, "qdrant-client chỉ có trong container app")
class RetrievalP0IntegrationTests(unittest.TestCase):
    def test_expansion_increases_recall_but_original_query_controls_ranking_and_seeds(self):
        original_hits = [
            FakePoint(
                "cyber_D18_Pd",
                "cyber_D18",
                "Luật An ninh mạng 2025.docx",
                "Khai thác điểm yếu, lỗ hổng bảo mật để chiếm đoạt thông tin.",
                1.0,
            ),
            FakePoint(
                "privacy_D9_K1",
                "privacy_D9",
                "Luật Bảo vệ dữ liệu cá nhân 2025.docx",
                "Sự đồng ý của chủ thể dữ liệu cá nhân.",
                0.4,
            ),
        ]
        expanded_hits = [
            FakePoint(
                "cyber_D2_K13",
                "cyber_D2",
                "Luật An ninh mạng 2025.docx",
                "Tấn công mạng là hành vi chiếm đoạt thông tin và phá hoại hệ thống.",
                1.0,
            ),
            original_hits[0],
        ]

        with patch(
            "src.agents.agent_retrieval.node_hybrid_search.hybrid_search",
            side_effect=[
                {"children": original_hits},
                {"children": expanded_hits},
            ],
        ):
            result = hybrid_search_node(
                {
                    "query": "SQL Injection khai thác lỗ hổng website tải cơ sở dữ liệu",
                    "mode": "critic",
                },
                qdrant_client=object(),
                embedding_model=object(),
                bm25=object(),
                qdrant_child_col="children",
                qdrant_parent_col="parents",
                top_k=3,
                prefetch_limit=20,
                article_expand_score_ratio=0.4,
                cross_encoder_reranker=CyberReranker(),
            )

        self.assertEqual("cyber_D18_Pd", result["retrieved_chunks"][0]["chunk_id"])
        self.assertTrue(
            any(item.casefold().endswith("_d2") for item in result["retrieved_dieu_ids"]),
            result["retrieved_dieu_ids"],
        )
        self.assertFalse(
            any("privacy" in item.casefold() for item in result["retrieved_dieu_ids"])
        )
        self.assertNotIn("sự đồng ý của chủ thể dữ liệu", result["expanded_query"])


if __name__ == "__main__":
    unittest.main()
