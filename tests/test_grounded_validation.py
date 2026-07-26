import unittest
from unittest.mock import patch
import unicodedata

from src.agents.common.grounded_validation import (
    INCOMPLETE_GROUNDS_WARNING,
    INSUFFICIENT_GROUNDS,
    build_grounded_sources,
    build_extractive_grounded_draft,
    build_safe_grounded_fallback,
    render_grounded_answer,
    salvage_grounded_draft,
    validate_grounded_draft,
)


CONTEXT = [
    "[Khoản 1, Điều 10, Luật Dữ liệu 2024] Chủ thể phải bảo vệ dữ liệu.\n"
    "[Điểm a, Khoản 2, Điều 10, Luật Dữ liệu 2024] Không được chia sẻ trái phép.",
    "[Khoản 3, Điều 20, Nghị định 99/2025/NĐ-CP] Phạt cá nhân từ 10 đến 20 triệu đồng; "
    "tổ chức bị phạt gấp hai lần nếu thực hiện hành vi chia sẻ trái phép.",
]


def valid_draft() -> str:
    return """## Tóm tắt tình huống
Người dùng mô tả việc chia sẻ dữ liệu cho bên thứ ba.

## Các vấn đề pháp lý
Cần xác định việc chia sẻ có phù hợp quy định và có chế tài hay không.

## Phân tích
### Hành vi chia sẻ dữ liệu
[[QUOTE:S2]]

- **Nội dung điều luật:** Quy định này cấm việc chia sẻ dữ liệu trái phép cho chủ thể khác.
- **Hành vi phù hợp:** Dữ liệu đã được chia sẻ cho bên thứ ba.
- **Phân tích:** Tình tiết người dùng nêu là chia sẻ dữ liệu cho bên thứ ba, trong khi đoạn luật cấm việc chia sẻ trái phép; sự trùng khớp về hành vi làm phát sinh căn cứ áp dụng quy định này.
- **Điều kiện còn thiếu:** Không còn thiếu dữ kiện về việc chia sẻ; vẫn cần xác định việc chia sẻ có được pháp luật cho phép hay không.
- **Căn cứ pháp lý:** [[CITE:S2]]
- **Đánh giá:** 🟢 Đủ căn cứ: Theo tình huống được mô tả, hành vi vi phạm quy định retrieved. [[CITE:S2]]

## Chế tài
[[QUOTE:S3]]

- **Hành vi:** Chia sẻ trái phép.
- **Đối tượng:** Mức nêu trong SOURCE áp dụng cho cá nhân; tổ chức theo quy tắc trong SOURCE. [[CITE:S3]]
- **Điều kiện áp dụng:** Chỉ áp dụng khi xác định đúng hành vi nêu trên.
- **Cộng dồn:** SOURCE không quy định cộng dồn nên chưa đủ căn cứ để khẳng định.
- **Đánh giá:** 🟢 Đủ căn cứ: Có căn cứ xử phạt theo đoạn retrieved. [[CITE:S3]]

## Trả lời câu hỏi của người dùng
### 1. Việc chia sẻ dữ liệu có vi phạm không?
- **Trả lời trực tiếp:** Có vi phạm quy định cấm chia sẻ dữ liệu trái phép.
- **Vì sao:** Tình huống đã thể hiện hành vi chia sẻ cho bên thứ ba đúng với hành vi bị cấm trong nguồn. [[CITE:S2]]
- **Căn cứ:** [[CITE:S2]]
- **Còn thiếu:** Cần xác định việc chia sẻ có thuộc trường hợp được pháp luật cho phép hay không."""


class GroundedValidationTests(unittest.TestCase):
    def test_splits_parent_context_and_prioritises_law(self):
        sources = build_grounded_sources(CONTEXT)
        self.assertEqual(3, len(sources))
        self.assertEqual("S1", sources[0].source_id)
        self.assertEqual("Luật Dữ liệu 2024", sources[0].document)
        self.assertEqual("2", sources[1].clause)
        self.assertEqual("a", sources[1].point)

    def test_long_article_cannot_push_retrieved_penalty_context_out_of_source_limit(self):
        long_law = "\n".join(
            f"[Khoản {index}, Điều 10, Luật Dữ liệu 2024] Nghĩa vụ xử lý dữ liệu {index}."
            for index in range(1, 10)
        )
        penalty = (
            "[Khoản 3, Điều 20, Nghị định 99/2025/NĐ-CP] "
            "Phạt tiền hành vi chia sẻ dữ liệu trái phép."
        )
        sources = build_grounded_sources(
            [long_law, penalty],
            limit=3,
            query="chia sẻ dữ liệu trái phép bị xử phạt",
        )

        self.assertTrue(
            any(source.document.startswith("Nghị định 99") for source in sources)
        )

    def test_valid_answer_renders_exact_quotes_inline_citations_and_used_sources(self):
        sources = build_grounded_sources(CONTEXT)
        validation = validate_grounded_draft(valid_draft(), sources)
        self.assertTrue(validation.is_valid, validation.issues)

        answer = render_grounded_answer(valid_draft(), sources, is_complete=True)
        self.assertNotIn("[[", answer)
        self.assertIn("> “Không được chia sẻ trái phép.”", answer)
        self.assertIn("(Căn cứ: **Luật Dữ liệu 2024**, Điều 10, Khoản 2, Điểm a)", answer)
        self.assertIn("# Căn cứ pháp lý", answer)
        self.assertNotIn("Khoản 1", answer)  # S1 không được phần phân tích sử dụng.

    def test_rejects_unknown_source_article_and_document(self):
        sources = build_grounded_sources(CONTEXT)
        invalid = valid_draft().replace(
            "hành vi vi phạm quy định retrieved. [[CITE:S2]]",
            "hành vi vi phạm Điều 999 Nghị định 777/2099/NĐ-CP. [[CITE:S99]]",
            1,
        )
        validation = validate_grounded_draft(invalid, sources)
        self.assertFalse(validation.is_valid)
        joined = " ".join(validation.issues)
        self.assertIn("S99", joined)
        self.assertIn("Điều 999", joined)
        self.assertIn("Nghị định 777/2099/NĐ-CP", joined)

    def test_accepts_equivalent_document_separators(self):
        sources = build_grounded_sources(CONTEXT)
        equivalent = valid_draft().replace(
            "Có căn cứ xử phạt theo đoạn retrieved.",
            "Có căn cứ xử phạt theo Nghị định 99_2025_NĐ-CP.",
        )
        validation = validate_grounded_draft(equivalent, sources)
        self.assertTrue(validation.is_valid, validation.issues)

    def test_accepts_cross_reference_coordinates_present_verbatim_in_source(self):
        context = [
            "[Khoản 2, Điều 100, Nghị định 15/2020/NĐ-CP] Phạt hành vi được "
            "quy định tại Khoản 1 Điều 28 nếu chủ thể thực hiện trên mạng xã hội."
        ]
        query = (
            "Chủ thể thực hiện hành vi nêu tại Khoản 1 Điều 28 trên mạng xã hội "
            "và có bị xử phạt không?"
        )
        sources = build_grounded_sources(context, query=query)

        draft = build_extractive_grounded_draft(query=query, sources=sources)

        self.assertNotEqual(INSUFFICIENT_GROUNDS, draft)
        validation = validate_grounded_draft(draft, sources, query=query)
        self.assertTrue(validation.is_valid, validation.issues)

    def test_rejects_legal_conclusion_without_inline_citation(self):
        sources = build_grounded_sources(CONTEXT)
        invalid = valid_draft().replace(
            "Tình huống đã thể hiện hành vi chia sẻ cho bên thứ ba đúng với hành vi bị cấm trong nguồn. [[CITE:S2]]",
            "Tình huống đã thể hiện hành vi chia sẻ cho bên thứ ba đúng với hành vi bị cấm trong nguồn.",
        )
        validation = validate_grounded_draft(invalid, sources)
        self.assertFalse(validation.is_valid)
        self.assertTrue(any("citation" in issue.casefold() for issue in validation.issues))

    def test_synthesis_answers_each_original_question_instead_of_each_law(self):
        sources = build_grounded_sources(CONTEXT)
        query = """Công ty chia sẻ dữ liệu trái phép.
1. Công ty có vi phạm không?
2. Công ty có bị xử phạt không?
3. Công ty có phải xóa dữ liệu không?"""
        synthesis = valid_draft().replace(
            """### 1. Việc chia sẻ dữ liệu có vi phạm không?
- **Trả lời trực tiếp:** Có vi phạm quy định cấm chia sẻ dữ liệu trái phép.
- **Vì sao:** Tình huống đã thể hiện hành vi chia sẻ cho bên thứ ba đúng với hành vi bị cấm trong nguồn. [[CITE:S2]]
- **Căn cứ:** [[CITE:S2]]
- **Còn thiếu:** Cần xác định việc chia sẻ có thuộc trường hợp được pháp luật cho phép hay không.""",
            """### 1. Công ty có vi phạm không?
- **Trả lời trực tiếp:** Có vi phạm quy định cấm chia sẻ dữ liệu trái phép.
- **Vì sao:** Công ty đã chia sẻ dữ liệu, trùng với hành vi chia sẻ trái phép được nguồn cấm. [[CITE:S2]]
- **Căn cứ:** [[CITE:S2]]
- **Còn thiếu:** Cần xác định có trường hợp ngoại lệ cho phép việc chia sẻ hay không.

### 2. Công ty có bị xử phạt không?
- **Trả lời trực tiếp:** Chưa đủ căn cứ để kết luận mức phạt áp dụng cho công ty.
- **Vì sao:** Nguồn có quy định mức phạt nhưng cần xác định đúng chủ thể và đầy đủ điều kiện của hành vi. [[CITE:S3]]
- **Căn cứ:** [[CITE:S3]]
- **Còn thiếu:** Cần xác định tư cách chủ thể và điều kiện áp dụng mức xử phạt.

### 3. Công ty có phải xóa dữ liệu không?
- **Trả lời trực tiếp:** Không tìm thấy căn cứ trong các nguồn hiện có để khẳng định nghĩa vụ xóa dữ liệu.
- **Vì sao:** Không tìm thấy căn cứ nào trong nguồn hiện có quy định trực tiếp việc xóa dữ liệu trong tình huống này.
- **Căn cứ:** Không tìm thấy căn cứ trong các nguồn hiện có cho câu hỏi này.
- **Còn thiếu:** Cần truy xuất quy định trực tiếp về nghĩa vụ xóa dữ liệu và điều kiện phát sinh nghĩa vụ.""",
        )

        validation = validate_grounded_draft(synthesis, sources, query=query)

        self.assertTrue(validation.is_valid, validation.issues)

    def test_rejects_synthesis_heading_named_after_article(self):
        sources = build_grounded_sources(CONTEXT)
        invalid = valid_draft().replace(
            "### 1. Việc chia sẻ dữ liệu có vi phạm không?",
            "### Điều 10 Luật Dữ liệu",
        )

        validation = validate_grounded_draft(invalid, sources)

        self.assertFalse(validation.is_valid)
        self.assertTrue(any("theo điều luật" in issue for issue in validation.issues))

    def test_rejects_negative_answer_inferred_only_from_missing_source(self):
        sources = build_grounded_sources(CONTEXT)
        invalid = valid_draft().replace(
            """### 1. Việc chia sẻ dữ liệu có vi phạm không?
- **Trả lời trực tiếp:** Có vi phạm quy định cấm chia sẻ dữ liệu trái phép.
- **Vì sao:** Tình huống đã thể hiện hành vi chia sẻ cho bên thứ ba đúng với hành vi bị cấm trong nguồn. [[CITE:S2]]
- **Căn cứ:** [[CITE:S2]]
- **Còn thiếu:** Cần xác định việc chia sẻ có thuộc trường hợp được pháp luật cho phép hay không.""",
            """### 1. Công ty có phải xóa dữ liệu không?
- **Trả lời trực tiếp:** Không phải xóa dữ liệu.
- **Vì sao:** Không tìm thấy căn cứ nào trong nguồn hiện có quy định trực tiếp về nghĩa vụ xóa dữ liệu.
- **Căn cứ:** Không tìm thấy căn cứ trong các nguồn hiện có cho câu hỏi này.
- **Còn thiếu:** Cần truy xuất quy định trực tiếp về nghĩa vụ xóa dữ liệu.""",
        )

        validation = validate_grounded_draft(invalid, sources)

        self.assertFalse(validation.is_valid)
        self.assertTrue(any("chỉ từ việc không tìm thấy" in issue for issue in validation.issues))

    def test_rejects_sanction_number_in_synthesis_not_present_in_source(self):
        sources = build_grounded_sources(CONTEXT)
        invalid = valid_draft().replace(
            "Tình huống đã thể hiện hành vi chia sẻ cho bên thứ ba đúng với hành vi bị cấm trong nguồn. [[CITE:S2]]",
            "Công ty bị phạt 999 triệu đồng vì đã chia sẻ dữ liệu trái phép. [[CITE:S3]]",
        ).replace(
            "- **Căn cứ:** [[CITE:S2]]",
            "- **Căn cứ:** [[CITE:S3]]",
            1,
        )

        validation = validate_grounded_draft(invalid, sources)

        self.assertFalse(validation.is_valid)
        self.assertTrue(any("999" in issue for issue in validation.issues))

    def test_rejects_cross_citation_and_hallucinated_sanction_number(self):
        sources = build_grounded_sources(CONTEXT)
        invalid = valid_draft().replace(
            "hành vi vi phạm quy định retrieved. [[CITE:S2]]",
            "hành vi vi phạm Điều 10. [[CITE:S3]]",
            1,
        ).replace(
            "Có căn cứ xử phạt theo đoạn retrieved. [[CITE:S3]]",
            "Cá nhân bị phạt 999 triệu đồng. [[CITE:S3]]",
        )
        validation = validate_grounded_draft(invalid, sources)
        joined = " ".join(validation.issues)
        self.assertIn("999", joined)

    def test_rejects_unsupported_legal_effect(self):
        sources = build_grounded_sources(CONTEXT)
        invalid = valid_draft().replace(
            "Tình tiết người dùng nêu là chia sẻ dữ liệu cho bên thứ ba, trong khi đoạn luật cấm việc chia sẻ trái phép; sự trùng khớp về hành vi làm phát sinh căn cứ áp dụng quy định này.",
            "Tình tiết này khiến chủ thể có thể bị xử lý hình sự theo pháp luật, dù đoạn SOURCE không có hậu quả đó. [[CITE:S2]]",
        )
        validation = validate_grounded_draft(invalid, sources)
        self.assertFalse(validation.is_valid)
        self.assertTrue(
            any("xử lý hình sự" in issue for issue in validation.issues),
            validation.issues,
        )

    def test_safe_fallback_is_exact_when_no_source(self):
        validation = validate_grounded_draft(INSUFFICIENT_GROUNDS, [])
        self.assertTrue(validation.is_valid)
        answer = render_grounded_answer(
            INSUFFICIENT_GROUNDS,
            [],
            is_complete=False,
        )
        self.assertEqual(INSUFFICIENT_GROUNDS, answer)
        self.assertNotIn("Căn cứ pháp lý", answer)

    def test_pure_insufficient_answer_does_not_show_incomplete_warning(self):
        sources = build_grounded_sources(CONTEXT)

        answer = render_grounded_answer(
            INSUFFICIENT_GROUNDS,
            sources,
            is_complete=False,
        )

        self.assertEqual(INSUFFICIENT_GROUNDS, answer)
        self.assertNotIn(INCOMPLETE_GROUNDS_WARNING, answer)

    def test_incomplete_warning_when_some_cited_source_exists(self):
        sources = build_grounded_sources(CONTEXT)
        answer = render_grounded_answer(valid_draft(), sources, is_complete=False)
        self.assertIn(INCOMPLETE_GROUNDS_WARNING, answer)

    def test_salvages_valid_analysis_when_sanction_and_final_section_are_broken(self):
        sources = build_grounded_sources(CONTEXT)
        broken = valid_draft().replace(
            "[[QUOTE:S3]]",
            "",
        ).replace(
            "Có căn cứ xử phạt theo đoạn retrieved. [[CITE:S3]]",
            "Cá nhân bị phạt 999 triệu đồng.",
        ).replace(
            "Tình huống đã thể hiện hành vi chia sẻ cho bên thứ ba đúng với hành vi bị cấm trong nguồn. [[CITE:S2]]",
            "Tình huống đã thể hiện hành vi chia sẻ cho bên thứ ba đúng với hành vi bị cấm trong nguồn.",
        )
        salvaged = salvage_grounded_draft(
            query="Công ty chia sẻ dữ liệu cho bên thứ ba.",
            draft=broken,
            sources=sources,
        )
        self.assertNotEqual(INSUFFICIENT_GROUNDS, salvaged)
        self.assertTrue(validate_grounded_draft(salvaged, sources).is_valid)
        self.assertIn("## Phân tích", salvaged)
        self.assertIn("Chưa xác định được", salvaged)

    def test_salvage_fails_closed_without_any_grounded_analysis_block(self):
        sources = build_grounded_sources(CONTEXT)
        salvaged = salvage_grounded_draft(
            query="Câu hỏi",
            draft="# Phân tích\nKhông có marker nguồn.",
            sources=sources,
        )
        self.assertEqual(INSUFFICIENT_GROUNDS, salvaged)

    def test_salvages_quote_only_block_when_qwen_drops_conclusion_and_citation(self):
        sources = build_grounded_sources(CONTEXT)
        broken = """# Tóm tắt tình huống
Tình huống.
# Xác định vấn đề pháp lý
Vấn đề.
# Phân tích
## Hành vi chia sẻ
[[QUOTE:S2]]
Model phân tích nhưng quên hoàn toàn citation và dòng kết luận.
# Chế tài (nếu có)
Không rõ.
# Kết luận
Không có citation."""
        salvaged = salvage_grounded_draft(
            query="Công ty chia sẻ dữ liệu.",
            draft=broken,
            sources=sources,
        )
        self.assertNotEqual(INSUFFICIENT_GROUNDS, salvaged)
        self.assertIn("[[QUOTE:S2]]", salvaged)
        self.assertIn("[[CITE:S2]]", salvaged)
        self.assertTrue(validate_grounded_draft(salvaged, sources).is_valid)

    def test_builds_extractive_answer_when_model_has_no_valid_marker(self):
        sources = build_grounded_sources(
            CONTEXT,
            query="chia sẻ dữ liệu cá nhân chưa có sự đồng ý",
        )
        draft = build_extractive_grounded_draft(
            query="Công ty chia sẻ dữ liệu cá nhân chưa có sự đồng ý.",
            sources=sources,
        )
        self.assertNotEqual(INSUFFICIENT_GROUNDS, draft)
        self.assertIn("[[QUOTE:", draft)
        self.assertIn("[[CITE:", draft)
        self.assertTrue(validate_grounded_draft(draft, sources).is_valid)

    def test_extractive_synthesis_answers_every_original_question(self):
        sources = build_grounded_sources(
            CONTEXT,
            query="chia sẻ dữ liệu trái phép bị xử phạt và nghĩa vụ xóa dữ liệu",
        )
        query = """Công ty chia sẻ dữ liệu trái phép.
1. Công ty có vi phạm không?
2. Công ty có bị xử phạt không?
3. Công ty có phải xóa dữ liệu không?"""

        draft = build_extractive_grounded_draft(query=query, sources=sources)
        validation = validate_grounded_draft(draft, sources, query=query)

        self.assertTrue(validation.is_valid, validation.issues)
        self.assertIn("### 1. Công ty có vi phạm không?", draft)
        self.assertIn("### 2. Công ty có bị xử phạt không?", draft)
        self.assertIn("### 3. Công ty có phải xóa dữ liệu không?", draft)
        self.assertIn("Không tìm thấy căn cứ trong các nguồn hiện có", draft)

    def test_conservative_fallback_keeps_sources_when_strict_fallback_fails(self):
        sources = build_grounded_sources(CONTEXT)
        query = """Công ty chia sẻ dữ liệu trái phép.
1. Công ty có vi phạm không?
2. Công ty có bị xử phạt không?
3. Công ty có phải xóa dữ liệu không?"""

        with patch(
            "src.agents.common.grounded_validation.build_extractive_grounded_draft",
            return_value=INSUFFICIENT_GROUNDS,
        ):
            draft = build_safe_grounded_fallback(
                query=query,
                sources=sources,
                is_complete=False,
            )

        self.assertNotEqual(INSUFFICIENT_GROUNDS, draft)
        self.assertEqual(3, draft.count("- **Trả lời trực tiếp:**"))
        self.assertIn("[[QUOTE:S", draft)
        self.assertIn("[[CITE:S", draft)
        rendered = render_grounded_answer(draft, sources, is_complete=False)
        self.assertIn(INCOMPLETE_GROUNDS_WARNING, rendered)
        self.assertIn("## Căn cứ pháp lý", rendered)
        self.assertNotIn("\n\n" + INSUFFICIENT_GROUNDS, rendered)

    def test_decomposed_vietnamese_query_uses_factual_behavior_not_question(self):
        context = [
            "[Khoản 1, Điều 32, Bộ luật Dân sự 2015] Việc sử dụng hình ảnh "
            "của cá nhân phải được người đó đồng ý."
        ]
        query = unicodedata.normalize(
            "NFD",
            """Một cá nhân tạo video deepfake bằng hình ảnh người nổi tiếng mà không có sự đồng ý.
Những quyền nào của người bị làm deepfake có thể bị xâm phạm?""",
        )
        sources = build_grounded_sources(context, query=query)

        draft = build_extractive_grounded_draft(query=query, sources=sources)

        self.assertNotEqual(INSUFFICIENT_GROUNDS, draft)
        self.assertIn("**Hành vi phù hợp:** Một cá nhân tạo video deepfake", draft)
        self.assertNotIn("**Hành vi phù hợp:** Những quyền nào", draft)
        self.assertIn(
            "### 1. Những quyền nào của người bị làm deepfake có thể bị xâm phạm?",
            draft,
        )

    def test_synthesis_extracts_inline_numbered_questions(self):
        sources = build_grounded_sources(CONTEXT, query="chia sẻ dữ liệu trái phép")
        query = (
            "Công ty chia sẻ dữ liệu trái phép. "
            "1. Công ty có vi phạm không? "
            "2. Công ty có bị xử phạt không? "
            "3. Công ty có phải xóa dữ liệu không?"
        )

        draft = build_extractive_grounded_draft(query=query, sources=sources)
        validation = validate_grounded_draft(draft, sources, query=query)

        self.assertTrue(validation.is_valid, validation.issues)
        self.assertIn("### 1. Công ty có vi phạm không?", draft)
        self.assertIn("### 2. Công ty có bị xử phạt không?", draft)
        self.assertIn("### 3. Công ty có phải xóa dữ liệu không?", draft)

    def test_user_question_about_sanctions_is_not_treated_as_model_hallucination(self):
        sources = build_grounded_sources(
            CONTEXT,
            query="chia sẻ dữ liệu trái phép bị xử lý như thế nào",
        )
        query = (
            "Công ty chia sẻ dữ liệu cho đối tác quảng cáo khi chưa xin sự đồng ý. "
            "Công ty có vi phạm và có thể bị xử lý như thế nào?"
        )
        draft = build_extractive_grounded_draft(query=query, sources=sources)

        self.assertNotEqual(INSUFFICIENT_GROUNDS, draft)
        validation = validate_grounded_draft(draft, sources, query=query)
        self.assertTrue(validation.is_valid, validation.issues)

    def test_extractive_answer_reserves_a_slot_for_retrieved_penalty_source(self):
        sources = build_grounded_sources(
            CONTEXT,
            query="chia sẻ dữ liệu trái phép bị xử lý như thế nào",
        )
        draft = build_extractive_grounded_draft(
            query="Công ty chia sẻ dữ liệu trái phép và có thể bị xử lý như thế nào?",
            sources=sources,
            max_articles=2,
        )

        penalty_source = next(
            source for source in sources if source.document.startswith("Nghị định 99")
        )
        self.assertIn(f"[[QUOTE:{penalty_source.source_id}]]", draft)

    def test_rejects_copying_only_language(self):
        sources = build_grounded_sources(CONTEXT)
        copying = valid_draft().replace(
            "Tình tiết người dùng nêu là chia sẻ dữ liệu cho bên thứ ba, trong khi đoạn luật cấm việc chia sẻ trái phép; sự trùng khớp về hành vi làm phát sinh căn cứ áp dụng quy định này.",
            "Chỉ xác nhận nội dung SOURCE và không bổ sung phân tích nào khác.",
        )
        validation = validate_grounded_draft(copying, sources)
        self.assertFalse(validation.is_valid)
        self.assertTrue(any("lặp SOURCE" in issue for issue in validation.issues))

    def test_rejects_generic_reused_legal_analysis_language(self):
        sources = build_grounded_sources(CONTEXT)
        generic = valid_draft().replace(
            "Tình tiết người dùng nêu là chia sẻ dữ liệu cho bên thứ ba, trong khi đoạn luật cấm việc chia sẻ trái phép; sự trùng khớp về hành vi làm phát sinh căn cứ áp dụng quy định này.",
            "Tình tiết nêu trên thuộc đúng nhóm hoạt động mà điều luật điều chỉnh; đây là căn cứ trực tiếp và có liên hệ trực tiếp với hành vi của người dùng.",
        )

        validation = validate_grounded_draft(generic, sources)

        self.assertFalse(validation.is_valid)
        self.assertTrue(
            any("câu chung chung" in issue for issue in validation.issues),
            validation.issues,
        )

    def test_extractive_analysis_is_specific_to_each_source(self):
        sources = build_grounded_sources(
            CONTEXT,
            query="Công ty chia sẻ dữ liệu trái phép và bị xử lý như thế nào?",
        )
        draft = build_extractive_grounded_draft(
            query="Công ty chia sẻ dữ liệu trái phép và bị xử lý như thế nào?",
            sources=sources,
            max_articles=2,
        )

        self.assertNotEqual(INSUFFICIENT_GROUNDS, draft)
        self.assertIn("**Nội dung điều luật:**", draft)
        self.assertIn("**Hành vi phù hợp:**", draft)
        self.assertIn("**Điều kiện còn thiếu:**", draft)
        self.assertNotIn("thuộc đúng nhóm hoạt động", draft)
        self.assertNotIn("có liên hệ trực tiếp", draft)
        self.assertIn("mô tả chế tài", draft)

    def test_rejects_identical_reasoning_reused_for_multiple_laws(self):
        sources = build_grounded_sources(CONTEXT)
        duplicated = valid_draft().replace(
            "## Chế tài",
            """### Quy định xử phạt
[[QUOTE:S3]]

- **Nội dung điều luật:** Quy định này xác định chế tài đối với hành vi chia sẻ dữ liệu trái phép.
- **Hành vi phù hợp:** Dữ liệu đã được chia sẻ cho bên thứ ba.
- **Phân tích:** Tình tiết người dùng nêu là chia sẻ dữ liệu cho bên thứ ba, trong khi đoạn luật cấm việc chia sẻ trái phép; sự trùng khớp về hành vi làm phát sinh căn cứ áp dụng quy định này.
- **Điều kiện còn thiếu:** Cần xác định chủ thể thực hiện là cá nhân hay tổ chức trước khi chọn mức xử phạt.
- **Căn cứ pháp lý:** [[CITE:S3]]
- **Đánh giá:** 🟡 Chưa đủ điều kiện kết luận: Cần làm rõ chủ thể áp dụng. [[CITE:S3]]

## Chế tài""",
        )

        validation = validate_grounded_draft(duplicated, sources)

        self.assertFalse(validation.is_valid)
        self.assertTrue(
            any("cùng một mẫu" in issue for issue in validation.issues),
            validation.issues,
        )

    def test_incomplete_retrieval_cannot_claim_green_confidence(self):
        sources = build_grounded_sources(CONTEXT)
        validation = validate_grounded_draft(
            valid_draft(),
            sources,
            is_complete=False,
        )
        self.assertFalse(validation.is_valid)
        self.assertTrue(any("không được gắn mức" in issue for issue in validation.issues))


if __name__ == "__main__":
    unittest.main()
