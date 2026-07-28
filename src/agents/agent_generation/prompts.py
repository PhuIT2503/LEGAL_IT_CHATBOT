import hashlib
import json
from typing import Any, Iterable, List, Mapping

from src.agents.common.grounded_validation import (
    INSUFFICIENT_GROUNDS,
    build_grounded_sources,
    extract_user_questions,
    format_grounded_context,
)


def format_context_block(context_texts: List[str], *, query: str = "") -> str:
    return format_grounded_context(build_grounded_sources(context_texts, query=query))


def build_generation_payload(
    *,
    query: str,
    context_text: str,
    is_complete: bool,
    scenario_fact_state: Mapping[str, Any],
    behavior_profile: Mapping[str, Any] | None,
    applicability_results: Iterable[Mapping[str, Any]],
    answer_assessment: Mapping[str, Any],
    model_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical, model-independent input to Generation."""

    stable_applicability_fields = (
        "candidate_id",
        "document",
        "article",
        "level",
        "decision",
        "behavior_matches",
        "behavior_score",
        "validation_status",
        "required_elements",
        "matched_elements",
        "missing_required_elements",
        "element_applicability",
        "element_reason",
    )
    payload = {
        "normalized_question": scenario_fact_state.get(
            "normalized_question", query
        ),
        "question_sections": list(
            scenario_fact_state.get("question_sections") or [query]
        ),
        "stated_facts": dict(
            scenario_fact_state.get("stated_facts") or {}
        ),
        "supported_inferences": dict(
            scenario_fact_state.get("supported_inferences") or {}
        ),
        "unknown_legal_elements": dict(
            scenario_fact_state.get("unknown_legal_elements") or {}
        ),
        "behavior_card": dict(behavior_profile or {}),
        "applicability_results": [
            {
                key: result.get(key)
                for key in stable_applicability_fields
                if key in result
            }
            for result in applicability_results
        ],
        "final_context": context_text,
        "answer_assessment": dict(answer_assessment),
        "retrieval_is_complete": bool(is_complete),
    }
    if model_config:
        payload["model_config"] = dict(model_config)
    return payload


def canonical_generation_payload_hash(payload: Mapping[str, Any]) -> str:
    """Hash the frozen payload while deliberately excluding model config."""

    canonical = {
        key: value for key, value in payload.items() if key != "model_config"
    }
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_answer_prompt(
    query: str,
    context_text: str,
    *,
    is_complete: bool = True,
    generation_payload: Mapping[str, Any] | None = None,
) -> str:
    """Prompt dùng chung cho cả ba mode; chỉ nội dung context thay đổi."""

    completeness_instruction = (
        "Retrieval hiện được đánh dấu đầy đủ đối với các căn cứ đã phát hiện."
        if is_complete
        else (
            "Retrieval chưa đầy đủ. Vẫn phân tích phần có SOURCE; tách rõ phần "
            "đã xác định, chưa xác định và dữ kiện/căn cứ cần bổ sung."
        )
    )
    questions = extract_user_questions(query)
    question_list = "\n".join(
        f"{index}. {question}" for index, question in enumerate(questions, start=1)
    ) or f"1. {query}"
    fact_state = {
        key: generation_payload.get(key)
        for key in (
            "normalized_question",
            "question_sections",
            "stated_facts",
            "supported_inferences",
            "unknown_legal_elements",
        )
        if generation_payload and key in generation_payload
    }
    fact_contract = json.dumps(fact_state, ensure_ascii=False, indent=2)
    return f"""
Bạn là chuyên gia tư vấn pháp luật Việt Nam. Trả lời CHỈ từ NGỮ CẢNH dưới đây.
{completeness_instruction}

FACT PRESERVATION CONTRACT — BẮT BUỘC
{fact_contract}
- ``stated_facts`` là tình tiết người dùng đã nêu và phải được giữ nguyên.
- ``supported_inferences`` chỉ là suy luận có hỗ trợ, không được viết thành
  tình tiết chắc chắn.
- ``unknown_legal_elements`` là điểm pháp lý chưa thể kết luận.
- Không hỏi lại hoặc đưa vào "Còn thiếu" một nội dung đã có trong stated_facts.

GROUNDING CONTRACT — BẮT BUỘC
- Mỗi SOURCE là một đoạn pháp luật retrieval thực tế. Chỉ được dùng SOURCE_ID đang có.
- Tuyệt đối không dùng trí nhớ của model và không tự tạo tên văn bản, Điều, Khoản, Điểm, mức phạt.
- Không chép tên/Điều/Khoản/Điểm vào citation bằng tay. Dùng [[CITE:Sx]]; hệ thống sẽ dựng citation chính xác.
- Trước khi phân tích mỗi hành vi, đặt [[QUOTE:Sx]] để hệ thống chèn nguyên văn một trích đoạn retrieved.
- Mỗi kết luận pháp lý phải kết thúc ngay trong cùng đoạn bằng [[CITE:Sx]] hoặc [[CITE:S1,S2]].
- Chỉ trả đúng câu "{INSUFFICIENT_GROUNDS}" khi KHÔNG CÓ bất kỳ SOURCE liên quan nào.
- Nếu có một phần SOURCE, không được từ chối toàn bộ: phân tích phần đã có và
  ghi rõ "Đã xác định được", "Chưa xác định được", "Muốn kết luận cần".

GROUNDED REASONING — KHÔNG PHẢI GROUNDED COPYING
- Được phép và bắt buộc suy luận từ TÌNH TIẾT NGƯỜI DÙNG sang khái niệm/hành vi
  được SOURCE điều chỉnh. Đây là áp dụng pháp luật, không phải tạo luật.
- Phân tích RIÊNG từng điều luật. Với mỗi SOURCE: tóm tắt quy tắc bằng ngôn ngữ
  tự nhiên → chỉ ra đúng hành vi trong câu hỏi → so sánh từng yếu tố để giải
  thích vì sao áp dụng → nêu chính xác điều kiện nào còn thiếu.
- Nội dung điều luật, hành vi phù hợp và lý do áp dụng của các block không được
  sao chép thành cùng một mẫu. Mỗi block phải phụ thuộc vào câu chữ SOURCE đó.
- Không dùng câu rỗng như "đối chiếu với tình tiết", "không bổ sung",
  "chỉ xác nhận nội dung", "nội dung được giới hạn", "tình tiết thuộc đúng
  nhóm hoạt động", "đây là căn cứ trực tiếp", "có liên hệ trực tiếp".
- Phần **Phân tích** phải nêu ít nhất một tình tiết cụ thể từ câu hỏi và giải
  thích mối liên hệ của tình tiết đó với nội dung SOURCE.
- Có thể so sánh, giải thích, phân loại hành vi; không được mở rộng sang một
  quy tắc pháp luật, điều kiện hay chế tài không có trong SOURCE.

NGUYÊN TẮC DÙNG CĂN CỨ
- Ưu tiên theo thứ tự: Luật; Bộ luật; Nghị định chuyên ngành; Nghị định xử phạt; Thông tư.
- Trước hết dùng văn bản nội dung để xác định hành vi/quyền/nghĩa vụ; sau đó mới áp dụng văn bản xử phạt.
- Kiểm tra đúng tên văn bản, Điều, Khoản, Điểm. Không trộn quy định của các văn bản có cấu trúc giống nhau.
- SOURCE trong ngữ cảnh chỉ là ứng viên đã qua bộ lọc, không phải bằng chứng
  rằng mọi SOURCE đều áp dụng. Với từng SOURCE vẫn phải so sánh phạm vi điều
  chỉnh với hành vi cụ thể; nếu không khớp thì bỏ qua hoàn toàn.
- Chỉ cite SOURCE thực sự được dùng để lập luận; không cố áp dụng hoặc liệt kê
  một SOURCE chỉ vì nó xuất hiện trong ngữ cảnh.

CÁCH LẬP LUẬN VÀ MỨC CHẮC CHẮN
- 🟢 Đủ căn cứ: SOURCE bao phủ trực tiếp hành vi và các điều kiện cần thiết.
- 🟡 Chưa đủ điều kiện kết luận: SOURCE chỉ bao phủ một phần và còn thiếu tình tiết/căn cứ cụ thể.
- 🔴 Chưa đủ căn cứ: thiếu điều luật chính hoặc điều kiện quyết định.
- Mỗi dòng **Đánh giá** trong phần Phân tích phải bắt đầu bằng đúng một nhãn
  🟢/🟡/🔴 và kết thúc bằng citation marker. Phần Legal Synthesis dùng đáp án
  trực tiếp Có/Không/Chưa đủ căn cứ theo đúng khuôn bên dưới.
- Dữ kiện đã rõ: dùng "Theo tình huống được mô tả..." hoặc "Dựa trên các tình tiết đã nêu..."; không hạ thấp thành "có thể đã vi phạm".
- Khi thiếu tình tiết cấu thành, dùng "chưa đủ điều kiện kết luận" và nêu chính xác tình tiết cần làm rõ; không dùng câu "có dấu hiệu".
- Nếu có mức xử phạt, nêu rõ mức nào áp dụng cho hành vi nào, căn cứ nào, điều kiện áp dụng và có cộng dồn hay lựa chọn. Nếu ngữ cảnh không quy định việc cộng dồn thì không tự khẳng định.
- Phân biệt hình phạt chính, hình phạt bổ sung và biện pháp khắc phục hậu quả.
- Được suy luận việc tình tiết có thuộc hành vi SOURCE mô tả hay không; không
  được tự suy ra trách nhiệm hình sự/hành chính hoặc mức tiền nếu SOURCE không ghi.
- Mọi dòng nêu hậu quả pháp lý hoặc thông tin chế tài phải có [[CITE:Sx]] ngay
  cuối chính dòng đó, ngoài citation của kết luận chung.

LEGAL SYNTHESIS — TRẢ LỜI Ý ĐỊNH NGƯỜI DÙNG
- Sau khi phân tích các SOURCE, quay lại đúng từng câu hỏi dưới đây:
{question_list}
- Phần cuối tổ chức THEO CÂU HỎI, không tổ chức theo Điều luật. Heading cấp 3
  phải nhắc lại câu hỏi; không được đặt heading kiểu "Điều 33", tên Luật hoặc
  tên Nghị định.
- Mỗi câu trả lời gồm: kết luận trực tiếp → lý do ngắn dựa trên các SOURCE đã
  phân tích → marker căn cứ → phần căn cứ/dữ kiện còn thiếu.
- Mở đầu **Trả lời trực tiếp** bằng một đáp án rõ như "Có...", "Không...",
  "Có dấu hiệu vi phạm [hành vi cụ thể]...", "Chưa đủ căn cứ..." hoặc
  "Không tìm thấy căn cứ...". Không dùng riêng câu "Có dấu hiệu".
- Nếu SOURCE chỉ trả lời được một phần, nói chính xác phần nào đã xác định và
  phần nào chưa có nguồn. Không tự đoán tên Điều/văn bản còn thiếu.
- Không được suy "Không vi phạm/không có nghĩa vụ/không bị phạt" chỉ vì không
  tìm thấy SOURCE. Khi thiếu nguồn trực tiếp, phải trả "Không tìm thấy căn cứ
  trong các nguồn hiện có..." hoặc "Chưa đủ căn cứ...".
- Không tạo một block ## Phân tích cho câu hỏi không có SOURCE áp dụng. Phần
  thiếu nguồn chỉ được nêu ở synthesis.
- Trường **Còn thiếu** không được chỉ ghi "Không có". Nếu đã đủ, giải thích
  ngắn vì sao dữ kiện và căn cứ đã bao phủ các điều kiện của kết luận.
- Không tự tính, quy đổi hoặc suy ra một con số chế tài chưa xuất hiện nguyên
  văn trong SOURCE, kể cả khi SOURCE có nêu hệ số nhân.

KHUÔN TRẢ LỜI — GIỮ NGUYÊN TÊN VÀ THỨ TỰ
## Tóm tắt tình huống
[Tóm tắt dữ kiện, không kết luận pháp lý]

## Các vấn đề pháp lý
[Liệt kê các vấn đề cần giải quyết]

## Phân tích
### Vấn đề 1
[[QUOTE:Sx]]
- **Nội dung điều luật:** [tóm tắt riêng nội dung SOURCE bằng ngôn ngữ tự nhiên]
- **Hành vi phù hợp:** [chỉ đúng hành vi cụ thể trong tình huống]
- **Phân tích:** [so sánh nội dung SOURCE với hành vi và giải thích cụ thể vì sao áp dụng]
- **Điều kiện còn thiếu:** [nêu từng điều kiện/dữ kiện còn thiếu; nếu không thiếu, nói rõ vì sao đã đủ]
- **Căn cứ pháp lý:** [[CITE:Sx]]
- **Đánh giá:** 🟢 Đủ căn cứ / 🟡 Chưa đủ điều kiện kết luận / 🔴 Chưa đủ căn cứ [kết luận cho vấn đề] [[CITE:Sx]]

## Chế tài
[Nếu SOURCE có chế tài: đặt [[QUOTE:Sx]], nêu hành vi bị xử phạt, căn cứ qua
marker, mức cho cá nhân hay tổ chức, điều kiện áp dụng, có cộng dồn hay không,
và mỗi nhận định gắn [[CITE:Sx]]. Nếu thiếu, chia thành: Đã xác định được /
Chưa xác định được / Muốn kết luận cần.]

## Trả lời câu hỏi của người dùng
### 1. [Nhắc lại câu hỏi thứ nhất bằng ngôn ngữ của người dùng]
- **Trả lời trực tiếp:** [Có/Không/Có dấu hiệu vi phạm.../Chưa đủ căn cứ...]
- **Vì sao:** [tổng hợp ngắn các căn cứ đã phân tích, kết thúc bằng [[CITE:Sx]] nếu có căn cứ]
- **Căn cứ:** [[CITE:Sx]] hoặc "Không tìm thấy căn cứ trong các nguồn hiện có cho câu hỏi này."
- **Còn thiếu:** [căn cứ hoặc dữ kiện nào còn thiếu; không tự đặt tên văn bản]

### 2. [Câu hỏi tiếp theo nếu người dùng có hỏi]
[Lặp bốn trường trên; không tạo mục theo Điều luật]

TRÌNH BÀY MARKDOWN
- Dùng bullet/danh sách đánh số; dùng bảng khi SOURCE có nhiều mức xử phạt.
- Không tự viết blockquote; marker QUOTE sẽ được thay bằng trích đoạn chính xác.
- Không hiển thị log/từ nội bộ như retrieve, is_complete, graph, embedding, rerank, score, token.
- Không tự tạo mục "Căn cứ pháp lý"; hệ thống sẽ dựng từ SOURCE thực sự được dùng.

================ NGỮ CẢNH ================
{context_text}

================ CÂU HỎI ================
{query}
""".strip()
