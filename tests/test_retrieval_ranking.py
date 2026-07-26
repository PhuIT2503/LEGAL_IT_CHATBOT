import unittest

from src.agents.common.retrieval_ranking import (
    deduplicate_context_records,
    filter_semantically_relevant,
    lexical_relevance,
    rerank_context_records,
    select_balanced_top_k,
)


class PrivacyReranker:
    def score(self, query, passages):
        return [0.95 if "Bảo vệ dữ liệu cá nhân" in passage else 0.05 for passage in passages]


class StaticReranker:
    def __init__(self, scores):
        self.scores = scores

    def score(self, query, passages):
        return list(self.scores)


class RetrievalRankingTests(unittest.TestCase):
    def test_privacy_law_outranks_generic_penalty_text(self):
        query = (
            "Ứng dụng thu thập vị trí, không thông báo mục đích và không xin "
            "sự đồng ý trước khi chia sẻ dữ liệu cá nhân cho đối tác quảng cáo"
        )
        records = [
            {
                "source": "Nghị Định xử phạt",
                "text": "Phạt tiền đối với việc cung cấp thông tin sai về chủ quyền.",
            },
            {
                "source": "Luật Bảo vệ dữ liệu cá nhân 2025",
                "text": (
                    "Ứng dụng di động phải thông báo việc sử dụng dữ liệu vị trí. "
                    "Thu thập dữ liệu cá nhân phải có sự đồng ý của chủ thể."
                ),
            },
        ]
        ranked = rerank_context_records(query, records, reranker=PrivacyReranker())
        self.assertIn("Luật Bảo vệ dữ liệu", ranked[0]["source"])

    def test_matching_phrases_score_above_isolated_generic_terms(self):
        query = "chia sẻ dữ liệu cá nhân khi chưa có sự đồng ý"
        relevant = "Dữ liệu cá nhân được chia sẻ sau khi có sự đồng ý."
        generic = "Chia sẻ thông tin và dữ liệu phục vụ quản lý nhà nước."
        self.assertGreater(
            lexical_relevance(query, relevant),
            lexical_relevance(query, generic),
        )

    def test_penalises_dotted_form_that_only_matches_contact_fields(self):
        query = "ứng dụng thu thập email vị trí và dữ liệu cá nhân"
        records = [
            {
                "source": "Nghị định về dịch vụ trực tuyến",
                "text": "Địa chỉ: ........ Email: ........ Vị trí: ........",
            },
            {
                "source": "Luật Bảo vệ dữ liệu cá nhân 2025",
                "text": "Bảo vệ dữ liệu cá nhân trên ứng dụng di động.",
            },
        ]
        self.assertIn(
            "Luật Bảo vệ dữ liệu",
            rerank_context_records(query, records, reranker=PrivacyReranker())[0]["source"],
        )

    def test_original_query_candidate_wins_semantic_tie(self):
        records = [
            {
                "source": "Nguồn expansion",
                "text": "Candidate chỉ có từ query mở rộng",
                "expanded_rank": 1,
            },
            {
                "source": "Nguồn query gốc",
                "text": "Candidate từ query gốc",
                "original_rank": 4,
            },
        ]
        ranked = rerank_context_records(
            "query gốc",
            records,
            reranker=StaticReranker([0.7, 0.7]),
        )

        self.assertEqual("Nguồn query gốc", ranked[0]["source"])

    def test_semantic_gate_keeps_only_strong_recursive_seeds(self):
        records = [
            {"semantic_score": 0.92, "chunk_id": "strong"},
            {"semantic_score": 0.61, "chunk_id": "medium"},
            {"semantic_score": 0.12, "chunk_id": "weak"},
        ]
        selected = filter_semantically_relevant(records, ratio=0.7, minimum=0.2)

        self.assertEqual(["strong"], [record["chunk_id"] for record in selected])

    def test_balanced_top_k_keeps_distinct_laws_and_one_penalty_article(self):
        chunks = [
            {"dieu_id_raw": "law_D28", "source": "Luật Bảo vệ dữ liệu 2025", "text": "Khoản 1"},
            {"dieu_id_raw": "law_D28", "source": "Luật Bảo vệ dữ liệu 2025", "text": "Khoản 3"},
            {"dieu_id_raw": "law_D31", "source": "Luật Bảo vệ dữ liệu 2025", "text": "Vị trí"},
            {"dieu_id_raw": "law_D9", "source": "Luật Bảo vệ dữ liệu 2025", "text": "Đồng ý"},
            {
                "dieu_id_raw": "decree_D84",
                "source": "Nghị Định xử phạt",
                "text": "Phạt tiền đối với chia sẻ thông tin cá nhân",
            },
        ]
        selected = select_balanced_top_k(
            "Hành vi này bị xử lý như thế nào?",
            chunks,
            top_k=4,
        )
        self.assertEqual(4, len({chunk["dieu_id_raw"] for chunk in selected}))
        self.assertTrue(any("Nghị Định" in chunk["source"] for chunk in selected))

    def test_duplicate_key_is_document_article_and_clause(self):
        chunks = [
            {
                "chunk_id": "law_D7_K1_Pa",
                "dieu_id_raw": "law_D7",
                "van_ban_id_raw": "law",
                "text": "Điểm a",
            },
            {
                "chunk_id": "law_D7_K1_Pb",
                "dieu_id_raw": "law_D7",
                "van_ban_id_raw": "law",
                "text": "Điểm b cùng Khoản 1",
            },
            {
                "chunk_id": "law_D7_K2",
                "dieu_id_raw": "law_D7",
                "van_ban_id_raw": "law",
                "text": "Khoản 2",
            },
            {
                "chunk_id": "other_D7_K1",
                "dieu_id_raw": "other_D7",
                "van_ban_id_raw": "other",
                "text": "Văn bản khác",
            },
        ]

        unique, duplicate_removed = deduplicate_context_records(chunks)
        selected = select_balanced_top_k("Quy định nào áp dụng?", chunks, top_k=4)

        self.assertEqual(3, len(unique))
        self.assertEqual(1, duplicate_removed)
        self.assertEqual(3, len(selected))
        self.assertEqual(
            ["law_D7_K1_Pa", "law_D7_K2", "other_D7_K1"],
            [chunk["chunk_id"] for chunk in selected],
        )


if __name__ == "__main__":
    unittest.main()
