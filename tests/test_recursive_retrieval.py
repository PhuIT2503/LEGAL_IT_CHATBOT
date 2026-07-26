import unittest

from src.agents.agent_retrieval.recursive_retrieval import recursive_retrieve


class FakeStore:
    def __init__(self, missing=None):
        self.missing = set(missing or [])

    def child_chunk_count(self, dieu_id):
        return 3 if dieu_id.endswith("D1") else 1

    def fetch_parent_record(self, dieu_id):
        if dieu_id in self.missing:
            return None
        if dieu_id.endswith("D1"):
            title = "Luật Thử nghiệm 2025"
        else:
            title = "Nghị Định chuyên ngành 2025"
        article = dieu_id.rsplit("D", 1)[-1]
        return {
            "dieu_id": dieu_id,
            "text": f"[Điều {article}, {title}] Toàn văn {dieu_id}",
            "source": title,
        }


class FakeQueryEngine:
    def check_retrieval_completeness(self, focus, all_ids, counts, totals, focus_node_ids=None):
        return {
            "is_complete": False,
            "missing_dieu_ids": ["luat_test_D2"],
            "missing_references": [{"missing_dieu_id": "luat_test_D2", "reason": "D1 dẫn D2"}],
            "compound_penalty_behaviors": [],
            "structurally_incomplete_dieu": [],
            "suggestions": [],
        }

    def find_missing_references(self, frontier, known, focus_node_ids=None):
        if "luat_test_D2" in frontier and "luat_test_D3" not in known:
            return [{"missing_dieu_id": "luat_test_D3", "reason": "D2 dẫn D3"}]
        return []


def state():
    return {
        "query": "Câu hỏi pháp lý",
        "retrieved_chunks": [
            {
                "chunk_id": "luat_test_D1_K1",
                "text": "[Khoản 1, Điều 1, Luật Thử nghiệm 2025] Một phần",
                "dieu_id_raw": "luat_test_D1",
                "van_ban_id_raw": "luat_test",
                "score": 1.0,
            }
        ],
        "retrieved_dieu_ids": ["luat_test_D1"],
        "dieu_scores": {"luat_test_D1": 1.0},
        "context_texts": [],
    }


class RecursiveRetrievalTests(unittest.TestCase):
    def test_keeps_seed_and_fetches_behavior_eligible_multi_hop_references(self):
        result = recursive_retrieve(
            state(),
            dieu_content_store=FakeStore(),
            critic_query_engine=FakeQueryEngine(),
            critic_score_ratio=0.6,
            critic_max_dieu=4,
            max_depth=3,
            max_iterations=5,
        )
        self.assertTrue(result["retrieval_is_complete"])
        self.assertEqual(
            {"luat_test_D2", "luat_test_D3"},
            set(result["graph_fetched_dieu_ids"]),
        )
        self.assertIn("Một phần", "\n".join(result["context_texts"]))
        self.assertNotIn("Toàn văn luat_test_D1", "\n".join(result["context_texts"]))
        seed = next(record for record in result["context_records"] if record["is_seed"])
        self.assertEqual("luat_test_D1_K1", seed["chunk_id"])
        self.assertEqual(0, seed["recursive_depth"])
        self.assertEqual("1", seed["clause"])

    def test_marks_incomplete_when_referenced_article_cannot_be_fetched(self):
        result = recursive_retrieve(
            state(),
            dieu_content_store=FakeStore(missing={"luat_test_D3"}),
            critic_query_engine=FakeQueryEngine(),
            critic_score_ratio=0.6,
            critic_max_dieu=4,
        )
        self.assertFalse(result["retrieval_is_complete"])
        self.assertTrue(result["critic_report"]["unresolved"])

    def test_exact_seed_node_prevents_article_level_incoming_reference_chain(self):
        class ExactNodeEngine:
            def __init__(self):
                self.focus_node_ids = None

            def check_retrieval_completeness(
                self, focus, all_ids, counts, totals, focus_node_ids=None
            ):
                self.focus_node_ids = focus_node_ids
                return {
                    "is_complete": True,
                    "missing_dieu_ids": [],
                    "missing_references": [],
                    "compound_penalty_behaviors": [],
                    "structurally_incomplete_dieu": [],
                    "suggestions": [],
                }

            def find_missing_references(self, frontier, known, focus_node_ids=None):
                raise AssertionError("Không được mở rộng khi exact seed không có cạnh")

        engine = ExactNodeEngine()
        exact_state = state()
        exact_state["retrieved_chunks"][0].update(
            {
                "chunk_id": "Lu_t_An_ninh_m_ng_2025_D13_K3_Ph",
                "dieu_id_raw": "Lu_t_An_ninh_m_ng_2025_D13",
                "van_ban_id_raw": "Lu_t_An_ninh_m_ng_2025",
            }
        )
        exact_state["retrieved_dieu_ids"] = ["lu_t_an_ninh_m_ng_2025_D13"]
        exact_state["dieu_scores"] = {"lu_t_an_ninh_m_ng_2025_D13": 1.0}
        result = recursive_retrieve(
            exact_state,
            dieu_content_store=FakeStore(),
            critic_query_engine=engine,
            critic_score_ratio=0.6,
            critic_max_dieu=4,
        )

        self.assertEqual(
            ["lu_t_an_ninh_m_ng_2025_D13_K3_Ph"], engine.focus_node_ids
        )
        self.assertEqual([], result["graph_fetched_dieu_ids"])
        self.assertEqual(
            "Lu_t_An_ninh_m_ng_2025_D13_K3_Ph",
            result["context_records"][0]["chunk_id"],
        )

    def test_recursive_candidate_below_behavior_threshold_is_rejected(self):
        behavior_state = state()
        behavior_state.update(
            {
                "query": "Deepfake AI dùng hình ảnh người nổi tiếng để quảng cáo",
                "behavior_profile": {
                    "actions": ["create_ai_deepfake", "use_person_likeness"],
                    "objects": ["synthetic_media", "person_likeness"],
                    "purposes": ["advertising"],
                    "conditions": [],
                },
            }
        )
        behavior_state["retrieved_chunks"][0]["behavior_score"] = 0.63
        result = recursive_retrieve(
            behavior_state,
            dieu_content_store=FakeStore(),
            critic_query_engine=FakeQueryEngine(),
            critic_score_ratio=0.6,
            critic_max_dieu=4,
        )

        self.assertEqual([], result["graph_fetched_dieu_ids"])
        self.assertEqual(1, len(result["context_records"]))
        self.assertTrue(result["context_records"][0]["is_seed"])
        self.assertTrue(result["critic_report"]["recursive_rejected"])


if __name__ == "__main__":
    unittest.main()
