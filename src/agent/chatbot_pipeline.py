import logging
import operator
from typing import TypedDict, Annotated, Sequence, List, Dict, Any, Optional

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from src.retrieval.qdrant_hybrid_search import hybrid_search, _point_id, _make_client
from src.retrieval.bm25_sparse import BM25SparseVectorizer, bm25_index_path
from src.llm.embedding_model import load_embedding_model
from src.knowledge_graph.neo4j_ingest import Neo4jGraphIngestor
from src.knowledge_graph.graph_builder import to_dieu_node_id
from src.agent.critic_query import CriticQueryEngine

logger = logging.getLogger(__name__)

MODES = ("naive", "article_expand", "critic")


class ChatbotState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query: str
    mode: str
    is_chit_chat: bool
    retrieved_chunks: List[Dict[str, Any]]
    retrieved_dieu_ids: List[str]
    dieu_scores: Dict[str, float]
    context_texts: List[str]
    article_expand_dieu_ids: List[str]
    draft_response: str
    critic_report: Dict[str, Any]
    graph_context: str
    graph_fetched_dieu_ids: List[str]
    final_response: str


class ChatbotPipeline:
    """
    Luồng RAG pháp luật hỗ trợ 3 kịch bản (để đánh giá A/B/C trong khóa luận):

    - mode="naive"       (Kịch bản 1 — RAG truyền thống, baseline):
      retrieval (top-k thuần) -> LLM trả lời thẳng. Không graph, không critic.
      Kỳ vọng: dễ thiếu hình phạt bổ sung/điều khoản dẫn chiếu nằm ngoài top-k.

    - mode="article_expand" (Kịch bản 2 — RAG mở rộng toàn Điều, KHÔNG dùng
      quan hệ Knowledge Graph, KHÔNG có Critic Agent):
      retrieval (đủ top_k Điều KHÁC NHAU đạt ngưỡng điểm article_expand_score_ratio
      — xem retrieve_documents) -> với MỖI Điều, chỉ lấy
      TOÀN VĂN CHÍNH Điều đó (không đi theo bất kỳ quan hệ KG nào — không
      THAM_CHIEU sang Điều khác, không dò HanhVi/CheTai/ChuThe/...) -> nhồi
      vào LLM, trả lời 1 lần. Mục đích: cô lập đúng 1 biến số so với Kịch bản
      3 — "mở rộng lên toàn Điều" tự nó KHÔNG cần Knowledge Graph (chỉ cần
      biết Điều có những chunk con nào), nên Kịch bản này xử lý tốt câu hỏi mà
      thông tin nằm ĐẦY ĐỦ trong 1 Điều (structural đa Khoản) nhưng KHÔNG tự
      lấy được chế tài kép/tham chiếu nằm ở Điều KHÁC — đúng khoảng trống mà
      Kịch bản 3 phải lấp bằng KG.

    - mode="critic"      (Kịch bản 3 — đề xuất của khóa luận):
      retrieval (top-k thuần) -> LLM sinh câu trả lời NHÁP -> Critic Agent (KG)
      đối chiếu nháp với graph, CHỈ khi phát hiện thật sự thiếu (hình phạt kép
      chính+bổ sung nằm khác Khoản, hoặc Điều khác được tham chiếu chưa lấy)
      mới xét bốc phần đó -> mỗi ứng viên còn phải qua CỔNG LỌC NGỮ NGHĨA bằng
      LLM (_is_candidate_relevant — 1 lệnh gọi nhỏ, riêng biệt, chỉ hỏi "đúng
      hành vi câu hỏi không?") trước khi thực sự bơm vào ngữ cảnh -> ép LLM sửa
      lại câu trả lời nháp bằng đúng phần đã qua lọc. Nếu không phát hiện/không
      ứng viên nào qua được cổng lọc thì dùng thẳng câu trả lời nháp — không tốn
      thêm 1 lượt LLM vô ích. Đây là đóng góp cốt lõi của khóa luận: vừa đủ
      recall (nhờ KG) vừa giữ ngữ cảnh gọn VÀ SẠCH (chỉ bốc đúng phần thiếu VÀ
      thật sự liên quan, không blind-dump như Kịch bản 2).
    """

    def __init__(
        self,
        llm,
        qdrant_child_col: str = "legal_child_chunks",
        qdrant_parent_col: str = "legal_parent_chunks",
        qdrant_url: Optional[str] = None,
        qdrant_path: str = "data/.qdrant",
        embedding_model_name: str = "data/ai_vietnamese_embedding_v2_finetuned_final",
        embedding_device: str = "cpu",
        embedding_max_seq_length: int = 256,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_pass: str = "legal_kg_2024",
        top_k: int = 5,
        prefetch_limit: int = 20,
        critic_score_ratio: float = 0.7,
        critic_max_dieu: int = 3,
        article_expand_score_ratio: float = 0.6,
    ):
        self.llm = llm
        self.qdrant_child_col = qdrant_child_col
        self.qdrant_parent_col = qdrant_parent_col
        self.qdrant_path = qdrant_path
        self.top_k = top_k
        self.prefetch_limit = prefetch_limit
        # Critic Agent chỉ chạy completeness-check trên các Điều có điểm >=
        # critic_score_ratio * điểm Điều cao nhất (KHÔNG cứng "chỉ top-1", KHÔNG
        # chạy tràn lan trên mọi Điều trong top-k). Lý do dùng TỶ LỆ thay vì cắt
        # cứng theo thứ hạng: nếu top-k lẫn phải Điều không thật sự liên quan
        # (nhiễu retrieval, điểm thấp hẳn so với Điều đứng đầu), mỗi Điều đó lại
        # kéo theo tham chiếu RIÊNG của nó -> càng nhiều Điều bị kiểm tra, càng dễ
        # fetch phải nội dung lạc đề, khiến câu trả lời cuối bị "Lost in the
        # Middle" y như Kịch bản 2 — đã quan sát thấy thực tế (Điều 74 về nhãn
        # hiệu bị fetch nhầm khi hỏi về sáng chế, do lọt top-k với điểm thấp).
        # Ngược lại, nếu có NHIỀU Điều cùng đạt điểm cao gần nhau (câu hỏi thực sự
        # cần đối chiếu ≥2 Điều), tỷ lệ này vẫn giữ được TẤT CẢ — không bỏ sót như
        # cách cắt cứng "chỉ top-1" trước đây. critic_max_dieu là van an toàn,
        # tránh trường hợp hiếm khi quá nhiều Điều cùng đạt điểm cao.
        self.critic_score_ratio = critic_score_ratio
        self.critic_max_dieu = critic_max_dieu

        # article_expand xét trên 1 tập RỘNG hơn top-k thuần (retrieve_documents,
        # để đủ top_k Điều KHÁC NHAU) — BẮT BUỘC vẫn phải lọc theo tỷ lệ điểm
        # (KHÔNG lấy mù toàn bộ top_k đầu tiên của tập rộng): đã quan sát thực
        # tế lấy mù không lọc gì khiến article_expand dính phải Điều lạc hẳn
        # chủ đề (từ Nghị định/Luật khác hoàn toàn), sập điểm hoàn toàn xuống
        # dưới cả naive (0.642 -> 0.15 trên cat1). KHÔNG cap max_dieu như
        # critic (critic_max_dieu=3) — article_expand vẫn ưu tiên lấy ĐỦ top_k
        # Điều nếu có đủ ứng viên đạt tỷ lệ này.
        self.article_expand_score_ratio = article_expand_score_ratio

        # Client/model/bm25 load 1 LẦN DUY NHẤT khi khởi tạo pipeline, dùng lại
        # cho mọi câu hỏi trong phiên — tránh load lại embedding model mỗi query.
        self.qdrant_client = _make_client(qdrant_path, qdrant_url)
        self.embedding_model = load_embedding_model(
            embedding_model_name, device=embedding_device, max_seq_length=embedding_max_seq_length
        )
        try:
            self.bm25 = BM25SparseVectorizer.load(bm25_index_path(qdrant_path, qdrant_child_col))
        except FileNotFoundError:
            logger.warning(
                f"Không tìm thấy BM25 index tại {bm25_index_path(qdrant_path, qdrant_child_col)} — "
                f"hybrid search sẽ chỉ dùng dense vector. Hãy chạy qdrant_local_ingest.py trước."
            )
            self.bm25 = None

        # Config Neo4j
        self.neo4j_ingestor = Neo4jGraphIngestor(uri=neo4j_uri, user=neo4j_user, password=neo4j_pass)
        self.critic_query = CriticQueryEngine(self.neo4j_ingestor)

        # Map id Dieu chuẩn (Neo4j) -> id gốc của parent chunk trong Qdrant (giữ hoa/thường).
        # Cần vì node Dieu trong Neo4j dùng van_ban_id viết thường, còn payload Qdrant
        # giữ nguyên hoa/thường của tên file gốc -> không thể suy ngược 1-1 bằng string.
        self._dieu_parent_lookup: dict = {}
        self._build_dieu_parent_lookup()

        # Map id Dieu chuẩn -> tổng số chunk con (Khoản/Điểm) THẬT trong Qdrant —
        # nguồn đếm ĐỘC LẬP với Neo4j, dùng làm lưới an toàn cho
        # structurally_incomplete (xem docstring _build_dieu_child_chunk_count).
        self._dieu_child_chunk_count: dict = {}
        self._build_dieu_child_chunk_count()

        self.workflow = self._build_graph()

        # Đếm dồn token qua MỌI lệnh gọi LLM trong 1 lần run() — phục vụ so sánh
        # "chi phí" giữa 3 kịch bản (số lệnh gọi LLM khác nhau: naive/article_expand
        # chỉ 1 lần sinh câu trả lời, critic có thêm draft + gate lọc từng ứng viên +
        # có thể regenerate). Đồng thời tách riêng theo TAG (router/draft/gate/
        # final_generate/chit_chat) để suy ra CHÍNH XÁC token của riêng lệnh gọi
        # SINH RA CÂU TRẢ LỜI CUỐI CÙNG (xem run() — đây là chỉ số đúng cho câu
        # chuyện "hiệu quả ngữ cảnh", khác với tổng chi phí cả pipeline).
        # Reset ở đầu mỗi run() — instance này chạy tuần tự từng câu hỏi
        # (run_evaluation.py), không có tranh chấp đa luồng.
        self._token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "call_count": 0}
        self._token_usage_by_tag = {}

    def _invoke_llm(self, prompt: str, tag: str = "other"):
        """Wrapper DUY NHẤT quanh self.llm.invoke() — mọi lệnh gọi LLM trong pipeline
        phải đi qua đây để cộng dồn token usage vào self._token_usage (tổng) VÀ
        self._token_usage_by_tag[tag] (tách riêng theo loại lệnh gọi, xem run()).

        tag dùng để xác định lệnh gọi nào THỰC SỰ sinh ra câu trả lời cuối cùng:
        - "router"/"gate": không bao giờ là câu trả lời cuối.
        - "draft": là câu trả lời cuối CHỈ KHI critic không phát hiện thiếu gì
          (finalize_draft — không có lệnh regenerate nào chạy thêm).
        - "final_generate": generate_single_pass_response — luôn LÀ câu trả lời
          cuối bất cứ khi nào được gọi (naive/article_expand gọi trực tiếp;
          critic gọi qua regenerate_response khi phát hiện thiếu).
        - "chit_chat": câu trả lời cuối khi câu hỏi được route thành chit-chat.
        """
        resp = self.llm.invoke(prompt)
        usage = None
        if getattr(resp, "usage_metadata", None):
            um = resp.usage_metadata
            usage = (um.get("input_tokens", 0), um.get("output_tokens", 0), um.get("total_tokens", 0))
        elif isinstance(getattr(resp, "response_metadata", None), dict):
            tu = resp.response_metadata.get("token_usage") or resp.response_metadata.get("usage")
            if tu:
                usage = (tu.get("prompt_tokens", 0), tu.get("completion_tokens", 0), tu.get("total_tokens", 0))
        if usage:
            self._token_usage["prompt_tokens"] += usage[0]
            self._token_usage["completion_tokens"] += usage[1]
            self._token_usage["total_tokens"] += usage[2]
            self._token_usage["call_count"] += 1
            bucket = self._token_usage_by_tag.setdefault(
                tag, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "call_count": 0}
            )
            bucket["prompt_tokens"] += usage[0]
            bucket["completion_tokens"] += usage[1]
            bucket["total_tokens"] += usage[2]
            bucket["call_count"] += 1
        return resp

    def _build_dieu_parent_lookup(self):
        """Quét 1 lần collection parent để dựng bảng tra id Dieu chuẩn -> parent id gốc Qdrant."""
        try:
            offset = None
            while True:
                points, offset = self.qdrant_client.scroll(
                    collection_name=self.qdrant_parent_col,
                    with_payload=True,
                    limit=256,
                    offset=offset,
                )
                for point in points:
                    payload = point.payload or {}
                    raw_dieu_id = payload.get("dieu_id", "")
                    raw_van_ban_id = payload.get("van_ban_id", "")
                    if not raw_dieu_id or not raw_van_ban_id:
                        continue
                    canonical = to_dieu_node_id(raw_van_ban_id, raw_dieu_id)
                    if canonical:
                        self._dieu_parent_lookup[canonical] = payload.get("id", "")
                if not offset:
                    break
            logger.info(f"Dieu->parent lookup: {len(self._dieu_parent_lookup)} Điều.")
        except Exception as e:
            logger.warning(f"Không dựng được dieu_parent_lookup (Qdrant parent collection có thể chưa có dữ liệu): {e}")

    def _build_dieu_child_chunk_count(self):
        """
        Quét 1 lần collection child, đếm số chunk_id DUY NHẤT thuộc mỗi Điều —
        dùng làm "tổng số phần" ĐỘC LẬP VỚI NEO4J cho check structurally_incomplete
        (critic_query.find_structurally_incomplete_dieu, tham số total_parts_override).

        LÝ DO CẦN NGUỒN THỨ 2 (phát hiện qua test thật): đếm "tổng số Khoản" bằng
        cách đếm cạnh CO_KHOAN trong Neo4j có thể bị THIẾU nếu bước ingest KG
        (LLM extraction) bỏ sót liên kết cho 1 số Khoản của 1 số Điều — đã xác
        nhận cụ thể: Điều 21 (Nghị định 15/2020) Neo4j chỉ thấy 1 Khoản dù văn
        bản gốc có nhiều hơn; Điều 26 Neo4j chỉ thấy cấu trúc dưới Khoản 1, dù
        Qdrant đã retrieve được cả Khoản 3, Khoản 5 (tức Khoản 3/5 CÓ TỒN TẠI
        trong dữ liệu ingest, chỉ là Neo4j không link tới chúng). Khi Neo4j đếm
        thiếu, "total_parts" bị đánh giá thấp bằng đúng "retrieved_parts" ->
        structurally_incomplete KHÔNG BAO GIỜ trigger được cho Điều đó, dù thực
        tế còn thiếu thật -> Critic Agent bỏ sót (bug đã quan sát qua test thật,
        khiến critic thua graph_blind ở các Điều bị ingest KG thiếu).

        Đếm qua Qdrant child collection thay vì Neo4j vì Qdrant chunk trực tiếp
        từ cấu trúc văn bản gốc lúc ingest (qdrant_local_ingest.py, chunking dựa
        trên regex/cấu trúc Khoản/Điểm thật) — KHÔNG qua bước LLM extraction dễ
        sai sót như lúc build đồ thị Neo4j, nên đáng tin cậy hơn làm "tổng số".
        """
        try:
            offset = None
            seen_chunk_ids: dict = {}
            while True:
                points, offset = self.qdrant_client.scroll(
                    collection_name=self.qdrant_child_col,
                    with_payload=True,
                    limit=256,
                    offset=offset,
                )
                for point in points:
                    payload = point.payload or {}
                    raw_dieu_id = payload.get("dieu_id", "")
                    raw_van_ban_id = payload.get("van_ban_id", "")
                    chunk_id = payload.get("id")
                    if not raw_dieu_id or not raw_van_ban_id or not chunk_id:
                        continue
                    canonical = to_dieu_node_id(raw_van_ban_id, raw_dieu_id)
                    if canonical:
                        seen_chunk_ids.setdefault(canonical, set()).add(chunk_id)
                if not offset:
                    break
            self._dieu_child_chunk_count = {d: len(ids) for d, ids in seen_chunk_ids.items()}
            logger.info(f"Dieu->so chunk con (Qdrant, doc lap Neo4j): {len(self._dieu_child_chunk_count)} Điều.")
        except Exception as e:
            logger.warning(f"Không dựng được dieu_child_chunk_count (Qdrant child collection có thể chưa có dữ liệu): {e}")

    def _fetch_parent_content(self, canonical_dieu_id: str) -> Optional[str]:
        """Lấy toàn văn 1 Điều từ Qdrant bằng id Dieu chuẩn (qua bảng tra self._dieu_parent_lookup)."""
        raw_parent_id = self._dieu_parent_lookup.get(canonical_dieu_id)
        if not raw_parent_id:
            return None
        try:
            points = self.qdrant_client.retrieve(
                collection_name=self.qdrant_parent_col,
                ids=[_point_id(raw_parent_id)],
                with_payload=True,
            )
            if points:
                return (points[0].payload or {}).get("content", "")
        except Exception as e:
            logger.warning(f"Không fetch được parent content cho {canonical_dieu_id}: {e}")
        return None

    def _compute_focus_dieu_ids(
        self,
        retrieved_dieu_ids: List[str],
        dieu_scores: Dict[str, float],
        score_ratio: Optional[float] = None,
        max_dieu: Optional[int] = None,
    ) -> List[str]:
        """
        Tập Điều "trong phạm vi xem xét" — lọc theo tỷ lệ điểm so với Điều điểm
        cao nhất, cap tối đa max_dieu Điều (xem lý do dùng TỶ LỆ thay vì cắt
        cứng theo thứ hạng trong __init__). Mặc định dùng critic_score_ratio/
        critic_max_dieu (gọi từ critic_check, KHÔNG truyền tham số) — đây LÀ cơ
        chế đề xuất của khóa luận, tránh 1 Điều điểm thấp (nhiễu retrieval) kéo
        theo tham chiếu riêng làm loãng ngữ cảnh.

        article_expand cũng dùng lại hàm này (retrieve_documents) nhưng truyền
        RIÊNG article_expand_score_ratio (xem __init__) và max_dieu=top_k
        (KHÔNG cap thấp như critic_max_dieu) — vì article_expand xét trên tập
        RỘNG hơn (wide_chunks). BẮT BUỘC phải lọc theo tỷ lệ này (KHÔNG lấy mù
        top_k đầu tiên của tập rộng, bỏ qua điểm số) — đã quan sát thực tế lấy
        mù không lọc gì khiến article_expand dính phải Điều lạc hẳn chủ đề (từ
        Nghị định/Luật khác hoàn toàn do độ liên quan giảm nhanh trong tập 20
        chunk), sập điểm hoàn toàn xuống dưới cả naive (0.642 -> 0.15 trên cat1).
        """
        score_ratio = self.critic_score_ratio if score_ratio is None else score_ratio
        max_dieu = self.critic_max_dieu if max_dieu is None else max_dieu
        if not retrieved_dieu_ids:
            return []
        top_score = dieu_scores.get(retrieved_dieu_ids[0], 0.0)
        focus_dieu_ids = []
        for d in retrieved_dieu_ids:
            if top_score > 0 and dieu_scores.get(d, 0.0) < top_score * score_ratio:
                continue
            focus_dieu_ids.append(d)
            if len(focus_dieu_ids) >= max_dieu:
                break
        return focus_dieu_ids

    def _is_candidate_relevant(self, query: str, candidate_content: str) -> bool:
        """
        Cổng lọc NGỮ NGHĨA thật (dùng LLM) trước khi fetch/bơm 1 ứng viên vào
        ngữ cảnh — đây là mảnh còn thiếu của Critic Agent: 3 check trong
        critic_query.py (missing_references, compound_penalty, structurally_incomplete)
        CHỈ dựa vào tín hiệu CẤU TRÚC/TOPOLOGY của graph (đếm Khoản, cạnh
        THAM_CHIEU, cạnh CheTai) — hoàn toàn KHÔNG biết nội dung ứng viên có thật
        sự nói về ĐÚNG hành vi trong câu hỏi hay không. critic_score_ratio cũng
        không giúp gì ở đây vì nó tái dùng lại chính điểm retrieval gốc (thứ đã
        để lọt Điều nhiễu vào top-k từ đầu), không phải 1 bộ lọc độc lập.

        Cố ý ĐẶT Ở ĐÂY (trước khi fetch) thay vì dặn dò trong prompt sinh câu trả
        lời cuối (đã thử ở regenerate_response và REVERT — xem lịch sử đo: model
        7B không đủ tin cậy tự lọc khi phải làm đồng thời 2 việc — vừa viết câu
        trả lời vừa thẩm định nhiều khối nội dung trộn lẫn trong 1 ngữ cảnh lớn,
        completeness_rate giảm từ 0.6 xuống 0.317 trên bộ test thật). Tách thành
        1 lệnh gọi LLM NHỎ, CHUYÊN ĐÚNG 1 VIỆC (yes/no trên ĐÚNG 1 ứng viên, không
        lẫn việc sinh văn bản) — cùng dạng câu hỏi phân loại đơn giản đã kiểm
        chứng hoạt động ổn định ở judge_fact_covered (score_evaluation.py). Nhờ
        lọc TRƯỚC khi fetch, ứng viên bị loại sẽ KHÔNG bao giờ xuất hiện trong
        graph_context — vừa giảm nhiễu vừa giảm kích thước ngữ cảnh so với
        article_expand (vốn không lọc gì cả).

        LƯU Ý (đã thử và sửa, phát hiện qua test trên bộ đa dạng 4 category):
        bản prompt đầu tiên chỉ hỏi "có nói về ĐÚNG hành vi/tình huống" — khung
        "hành vi" này khớp tốt với câu hỏi kiểu chế tài/xử phạt (nhóm
        same_dieu_compound_penalty) nhưng gây FALSE NEGATIVE có hệ thống với câu
        hỏi thủ tục/cấu trúc/quyền lợi (nhóm structural_multi_part, cross_reference)
        — quan sát cụ thể: Điều 65 "Kết quả hòa giải" (nêu đúng nội dung văn bản
        hòa giải cần có) bị gate loại dù khớp hoàn toàn câu hỏi, chỉ vì nó không
        mô tả 1 "hành vi vi phạm". Sửa bằng cách mở rộng tiêu chí sang "cần
        thiết/liên quan để trả lời đúng câu hỏi" nói chung, không giới hạn ở
        hành vi vi phạm.
        """
        if not candidate_content:
            return False
        prompt = (
            "Bạn là trợ lý kiểm tra độ liên quan, CHỈ làm đúng 1 việc: xác định 1 đoạn văn bản pháp luật có "
            "THỰC SỰ CẦN THIẾT để trả lời ĐÚNG câu hỏi hay không. Có thể là quy định về đúng hành vi/tình huống "
            "được hỏi (nếu câu hỏi về chế tài, xử phạt), HOẶC đúng chủ thể/thủ tục/điều kiện/thành phần nội dung "
            "mà câu hỏi yêu cầu (nếu câu hỏi về quyền, nghĩa vụ, quy trình, cấu trúc văn bản...) — KHÔNG bắt buộc "
            "phải là hành vi vi phạm. Chỉ trả lời KHÔNG liên quan nếu đoạn văn bản nói về CHỦ ĐỀ KHÁC hẳn, hoặc "
            "hành vi/đối tượng tương tự nhưng khác điều kiện, khác loại giấy phép/dịch vụ so với câu hỏi.\n\n"
            f"CÂU HỎI của người dùng: {query}\n\n"
            f"ĐOẠN VĂN BẢN ỨNG VIÊN (hệ thống tự động phát hiện qua cấu trúc Knowledge Graph, CHƯA xác nhận có "
            f"thật sự liên quan):\n{candidate_content[:3000]}\n\n"
            "Đoạn văn bản này có cần thiết/liên quan để trả lời ĐÚNG câu hỏi trên không? "
            "Chỉ trả lời đúng 1 từ: 'yes' hoặc 'no'."
        )
        try:
            resp = self._invoke_llm(prompt, tag="gate")
            return "yes" in resp.content.strip().lower()
        except Exception as e:
            logger.warning(f"Relevance gate lỗi, mặc định coi là liên quan (fail-open): {e}")
            return True

    def _any_chunk_relevant(self, query: str, texts: List[str]) -> bool:
        """
        Kiểm tra từng chunk RIÊNG LẺ (không gộp chung) — chỉ cần 1 chunk khớp là
        đủ coi Điều đó liên quan. BẮT BUỘC kiểm tra riêng lẻ thay vì nối chuỗi
        rồi hỏi 1 lần: 1 Điều thường có NHIỀU chunk được retrieve ứng với NHIỀU
        Khoản/hành vi khác nhau trong cùng Điều — nếu gộp chung, 1 chunk lạc đề
        (vd Khoản 1 nói về nghĩa vụ khác) có thể làm loãng/nhiễu tín hiệu của
        chunk ĐÚNG (vd Khoản 3 đúng hành vi câu hỏi), khiến LLM đánh giá sai
        "không liên quan" cho cả Điều dù có ít nhất 1 chunk thật sự khớp (bug đã
        quan sát được: Điều 99 có chunk Khoản 3 đúng + chunk Khoản 1 khác hành vi
        gộp chung bị loại nhầm).
        """
        for text in texts:
            if text and self._is_candidate_relevant(query, text):
                return True
        return False

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(ChatbotState)

        workflow.add_node("router", self.route_query)
        workflow.add_node("chit_chat", self.handle_chit_chat)
        workflow.add_node("retrieval", self.retrieve_documents)
        workflow.add_node("generate_single_pass", self.generate_single_pass_response)
        workflow.add_node("expand_article", self.expand_full_article)
        workflow.add_node("generate_draft", self.generate_draft_response)
        workflow.add_node("critic_check", self.critic_check)
        workflow.add_node("regenerate", self.regenerate_response)
        workflow.add_node("finalize_draft", self.finalize_draft)

        workflow.add_edge(START, "router")
        workflow.add_conditional_edges(
            "router",
            lambda x: "chit_chat" if x.get("is_chit_chat") else "retrieval",
            {"chit_chat": "chit_chat", "retrieval": "retrieval"},
        )
        workflow.add_edge("chit_chat", END)

        # Rẽ nhánh theo mode NGAY SAU retrieval — cả 3 kịch bản dùng chung 1
        # bước retrieval (top-k thuần) để đảm bảo so sánh công bằng, chỉ khác
        # nhau ở PHẦN SAU retrieval.
        workflow.add_conditional_edges(
            "retrieval",
            lambda x: x.get("mode", "critic"),
            {
                "naive": "generate_single_pass",
                "article_expand": "expand_article",
                "critic": "generate_draft",
            },
        )

        # Kịch bản 1: retrieval -> generate thẳng.
        # Kịch bản 2: retrieval -> mở rộng toàn văn Điều -> generate (dùng chung
        # node generate_single_pass — graph_context rỗng ở Kịch bản 1 nên hành
        # vi y hệt).
        workflow.add_edge("expand_article", "generate_single_pass")
        workflow.add_edge("generate_single_pass", END)

        # Kịch bản 3: retrieval -> sinh nháp -> critic đối chiếu KG -> có thiếu
        # thì sửa lại (regenerate), không thiếu thì dùng thẳng nháp (finalize_draft).
        workflow.add_edge("generate_draft", "critic_check")
        workflow.add_conditional_edges(
            "critic_check",
            lambda x: "regenerate" if x.get("graph_context") else "finalize_draft",
            {"regenerate": "regenerate", "finalize_draft": "finalize_draft"},
        )
        workflow.add_edge("regenerate", END)
        workflow.add_edge("finalize_draft", END)

        return workflow.compile()

    def route_query(self, state: ChatbotState):
        logger.info("Routing query...")
        query = state["query"]

        prompt = (
            "Bạn là trợ lý ảo. Hãy phân loại câu hỏi sau đây thuộc loại nào:\n"
            "1. 'chit_chat': Những câu chào hỏi, cảm ơn, hỏi thăm thông thường.\n"
            "2. 'legal': Câu hỏi về pháp luật, hình phạt, luật thương mại, an ninh mạng, CNTT, v.v.\n\n"
            f"Câu hỏi: {query}\n\n"
            "Chỉ trả về đúng 1 từ: 'chit_chat' hoặc 'legal'."
        )

        resp = self._invoke_llm(prompt, tag="router")
        content = resp.content.strip().lower()

        is_chit_chat = "chit_chat" in content
        logger.info(f"Routed as: {'chit_chat' if is_chit_chat else 'legal'}")

        return {"is_chit_chat": is_chit_chat}

    def handle_chit_chat(self, state: ChatbotState):
        logger.info("Handling chit-chat...")
        query = state["query"]

        prompt = (
            "Bạn là trợ lý ảo pháp luật nhiệt tình, thân thiện. "
            "Hãy trả lời tin nhắn sau của người dùng một cách ngắn gọn, lịch sự:\n\n"
            f"Người dùng: {query}"
        )
        resp = self._invoke_llm(prompt, tag="chit_chat")

        return {"final_response": resp.content, "messages": [AIMessage(content=resp.content)]}

    def retrieve_documents(self, state: ChatbotState):
        """
        RAG THUẦN — chỉ lấy đúng top-k chunk, KHÔNG expand, KHÔNG dùng KG.
        Dùng chung cho cả 3 kịch bản để đảm bảo so sánh công bằng.

        Trả về 2 danh sách Điều RIÊNG BIỆT, phục vụ 2 mục đích khác nhau:
        - retrieved_dieu_ids/dieu_scores: suy TRỰC TIẾP từ ĐÚNG top-k chunk
          (như thiết kế gốc, KHÔNG đổi) — dùng cho naive (MRR/nDCG chung) và
          critic (_compute_focus_dieu_ids) — 2 kịch bản này giữ nguyên hành vi
          cũ hoàn toàn.
        - article_expand_dieu_ids: suy từ 1 tập chunk RỘNG HƠN top-k
          (wide_chunks), lọc qua _compute_focus_dieu_ids với
          article_expand_score_ratio (xem __init__) rồi cap tối đa top_k —
          CHỈ dùng riêng cho expand_full_article. Lý do tách riêng: nếu suy
          Điều trực tiếp từ top-k CHUNK, nhiều chunk top-k có thể trùng vào
          CÙNG 1 Điều (vd top-5 chunk nhưng chỉ thuộc 2 Điều khác nhau, do 1
          Điều nhiều Khoản cùng khớp câu hỏi) — khiến article_expand chỉ có 2
          Điều để "mở rộng toàn Điều" dù top_k=5, trái đúng bản chất baseline
          này (mở rộng lên K ĐIỀU khác nhau, không phải K chunk). BẮT BUỘC lọc
          theo tỷ lệ điểm (KHÔNG lấy mù top_k đầu tiên của tập rộng) — đã
          quan sát thực tế lấy mù khiến article_expand dính phải Điều LẠC HẲN
          chủ đề (từ Nghị định/Luật khác hoàn toàn) do độ liên quan giảm rất
          nhanh trong tập 20 chunk, sập điểm hoàn toàn xuống dưới cả naive.
        """
        logger.info("Retrieving documents via Hybrid Search (top-k thuần, không expand)...")
        query = state["query"]

        # Tập chunk RỘNG hơn top_k, CHỈ dùng để suy article_expand_dieu_ids —
        # không vượt quá prefetch_limit (giới hạn candidate pool thật sự có
        # sẵn để xếp hạng, hỏi rộng hơn cũng không có thêm ứng viên mới).
        wide_limit = min(self.top_k * 4, self.prefetch_limit)

        try:
            result = hybrid_search(
                query=query,
                child_collection=self.qdrant_child_col,
                parent_collection=self.qdrant_parent_col,
                limit=wide_limit,
                prefetch_limit=self.prefetch_limit,
                fusion="rrf",
                include_parent=False,
                client=self.qdrant_client,
                model=self.embedding_model,
                bm25=self.bm25,
            )
            wide_hits = result.get("children", [])
        except Exception as e:
            logger.warning(f"Qdrant search failed (Qdrant có thể chưa ingest dữ liệu): {e}")
            wide_hits = []

        wide_chunks = []
        for p in wide_hits:
            payload = getattr(p, "payload", {}) or {}
            wide_chunks.append({
                "chunk_id": payload.get("id"),
                "text": payload.get("content", ""),
                "dieu_id_raw": payload.get("dieu_id", ""),
                "van_ban_id_raw": payload.get("van_ban_id", ""),
                "score": getattr(p, "score", None),
            })

        # retrieved_chunks/context_texts: ĐÚNG top-k chunk thuần — GIỮ NGUYÊN
        # như thiết kế gốc (cắt từ đầu tập rộng, thứ tự score giảm dần không đổi).
        retrieved_chunks = wide_chunks[: self.top_k]
        context_texts = [c["text"] for c in retrieved_chunks if c["text"]]

        # retrieved_dieu_ids/dieu_scores: suy TỪ ĐÚNG top-k chunk (KHÔNG đổi
        # so với thiết kế gốc) — dùng cho naive (MRR/nDCG) và critic
        # (_compute_focus_dieu_ids), GIỮ NGUYÊN hành vi cũ.
        dieu_best_score: dict[str, float] = {}
        dieu_order: list[str] = []
        for c in retrieved_chunks:
            d = to_dieu_node_id(c["van_ban_id_raw"], c["dieu_id_raw"])
            if not d:
                continue
            score = c["score"] or 0.0
            if d not in dieu_best_score:
                dieu_order.append(d)
                dieu_best_score[d] = score
            else:
                dieu_best_score[d] = max(dieu_best_score[d], score)
        retrieved_dieu_ids = dieu_order

        # article_expand_dieu_ids: thu thập TOÀN BỘ Điều khác nhau trong tập
        # rộng (không cắt sớm) cùng điểm tốt nhất, rồi lọc qua
        # _compute_focus_dieu_ids với article_expand_score_ratio (chặt hơn
        # critic_score_ratio — xem __init__) + cap tối đa top_k. KHÔNG lấy mù
        # toàn bộ top_k đầu tiên của tập rộng — độ liên quan giảm nhanh trong
        # tập 20 chunk, lấy mù sẽ dính phải Điều lạc hẳn chủ đề (đã quan sát
        # thực tế, xem __init__).
        wide_dieu_best_score: dict[str, float] = {}
        wide_dieu_order: list[str] = []
        for c in wide_chunks:
            d = to_dieu_node_id(c["van_ban_id_raw"], c["dieu_id_raw"])
            if not d:
                continue
            score = c["score"] or 0.0
            if d not in wide_dieu_best_score:
                wide_dieu_order.append(d)
                wide_dieu_best_score[d] = score
            else:
                wide_dieu_best_score[d] = max(wide_dieu_best_score[d], score)
        article_expand_dieu_ids = self._compute_focus_dieu_ids(
            wide_dieu_order, wide_dieu_best_score,
            score_ratio=self.article_expand_score_ratio, max_dieu=self.top_k,
        )

        logger.info(
            f"Retrieved {len(retrieved_chunks)} chunk top-k -> {len(retrieved_dieu_ids)} Điều (naive/critic); "
            f"article_expand: {len(article_expand_dieu_ids)} Điều khác nhau (quét rộng {len(wide_chunks)} chunk)."
        )
        return {
            "retrieved_chunks": retrieved_chunks,
            "retrieved_dieu_ids": retrieved_dieu_ids,
            "dieu_scores": dieu_best_score,
            "context_texts": context_texts,
            "article_expand_dieu_ids": article_expand_dieu_ids,
        }

    def _format_context_block(self, context_texts: List[str]) -> str:
        context_text = "\n\n".join(f"Tài liệu {i+1} (độ liên quan giảm dần):\n{t}" for i, t in enumerate(context_texts))
        return context_text or "Không tìm thấy tài liệu liên quan."

    def _build_answer_prompt(self, query: str, context_text: str) -> str:
        """
        Prompt sinh câu trả lời — dùng CHUNG cho cả 3 kịch bản (naive/
        article_expand qua generate_single_pass_response; critic qua
        generate_draft_response VÀ qua generate_single_pass_response khi
        mode="critic" ở bước regenerate — draft và regenerate BẮT BUỘC dùng
        giống nhau để không lẫn biến số giữa "sinh nháp" và "sinh lại").

        LƯU Ý (đã thử và REVERT — thí nghiệm CoT 4 bước + few-shot 1 ví dụ):
        từng đổi hẳn sang bắt model viết "Suy luận" theo 4 bước trước khi kết
        luận, kỳ vọng giảm nhầm lẫn Khoản/văn bản. Đo trên bộ test thật cho
        kết quả TỆ HƠN hẳn ở CẢ 3 mode: completeness_rate cat1 (10 mẫu) giảm
        mạnh — article_expand 0.642 -> 0.442, critic 0.758 -> 0.475. Nguyên
        nhân: bước "lọc Tài liệu" khiến model 7B quá tay, tự đánh giá 1 Tài
        liệu ĐÚNG là "không đề cập trực tiếp"/"không có quy định cụ thể" chỉ
        vì cách diễn đạt câu hỏi không khớp Y HỆT câu chữ trong Tài liệu. Đây
        là CÙNG BẢN CHẤT lỗi với thí nghiệm "nghi ngờ graph_context" đã revert
        ở regenerate_response trước đó (completeness giảm 0.6 -> 0.317) — chỉ
        khác ở chỗ lần này lỗi đến từ 1 bước suy luận có cấu trúc thay vì 1
        câu dặn trực tiếp. Kết luận: KHÔNG dùng CoT/few-shot dạng "tự lọc rồi
        mới trả lời" cho model 7B ở tác vụ này; giữ nguyên prompt ngắn, ra
        lệnh trực tiếp cho CẢ 3 mode.
        """
        return (
            "Bạn là một chuyên gia tư vấn pháp lý xuất sắc. "
            "Hãy trả lời câu hỏi của người dùng một cách chính xác, đầy đủ và dễ hiểu nhất dựa trên NGỮ CẢNH được cung cấp.\n\n"
            "Các Tài liệu được sắp theo thứ tự ĐỘ LIÊN QUAN GIẢM DẦN — Tài liệu 1 khớp với hành vi trong câu hỏi nhất. "
            "Chỉ dùng Tài liệu nếu nó nói về ĐÚNG hành vi trong câu hỏi; đừng lẫn sang hành vi tương tự nhưng khác. "
            "Mỗi Tài liệu có ghi rõ TÊN VĂN BẢN trong ngoặc vuông ở đầu — luôn kiểm tra kỹ tên văn bản đó có khớp "
            "ĐÚNG văn bản pháp luật được hỏi hay không TRƯỚC KHI dùng nội dung; nhiều văn bản luật khác nhau có "
            "điều khoản CẤU TRÚC GIỐNG NHAU (vd \"Hiệu lực thi hành\", \"Giải thích từ ngữ\", \"Phạm vi điều "
            "chỉnh\") — đừng vì thấy cấu trúc/hành văn quen thuộc mà tưởng nhầm là văn bản khác hoặc phủ nhận "
            "thông tin đúng chỉ vì có Tài liệu tương tự của văn bản khác đứng gần đó.\n\n"
            "Nếu NGỮ CẢNH không chứa thông tin để trả lời, hãy nói rằng bạn không biết, đừng tự bịa ra.\n"
            "Một hành vi vi phạm có thể có cả hình phạt chính LẪN hình phạt bổ sung/biện pháp khắc phục hậu quả — "
            "đọc kỹ TOÀN BỘ ngữ cảnh và liệt kê ĐẦY ĐỦ tất cả các loại chế tài liên quan, không chỉ hình phạt chính.\n\n"
            f"================ NGỮ CẢNH ================\n{context_text}\n\n"
            f"Câu hỏi: {query}"
        )

    def generate_single_pass_response(self, state: ChatbotState):
        """
        Sinh câu trả lời 1 LẦN DUY NHẤT từ context_texts (+ graph_context nếu có,
        dùng cho Kịch bản 2 — graph_context rỗng ở Kịch bản 1 nên hành vi y hệt
        RAG thuần).
        """
        logger.info("Generating single-pass response...")
        query = state["query"]
        context_texts = list(state.get("context_texts", []))
        graph_context = state.get("graph_context", "")

        if graph_context:
            context_texts.append(graph_context)

        context_text = self._format_context_block(context_texts)
        prompt = self._build_answer_prompt(query, context_text)

        resp = self._invoke_llm(prompt, tag="final_generate")
        return {"final_response": resp.content, "messages": [AIMessage(content=resp.content)]}

    def expand_full_article(self, state: ChatbotState):
        """
        Kịch bản 2 (RAG mở rộng toàn Điều, KHÔNG dùng quan hệ Knowledge Graph,
        KHÔNG có Critic Agent) — dùng để SO SÁNH, KHÔNG phải cơ chế đề xuất.

        Mở rộng article_expand_dieu_ids — đủ top_k Điều KHÁC NHAU, suy từ 1
        tập chunk RỘNG hơn top-k thuần rồi lọc qua article_expand_score_ratio
        (xem retrieve_documents/__init__) — KHÔNG dùng retrieved_dieu_ids (tập
        hẹp suy từ ĐÚNG top-k chunk, dành riêng cho naive/critic) và KHÔNG
        dùng critic_score_ratio/critic_max_dieu (đó là ngưỡng riêng của
        critic_check). article_expand vẫn CÓ lọc theo tỷ lệ điểm — KHÔNG lấy
        mù hoàn toàn (đã thử và bỏ: lấy mù top_k Điều đầu tiên của tập rộng,
        không lọc gì, khiến baseline dính phải Điều lạc hẳn chủ đề từ Nghị
        định/Luật khác, sập điểm xuống dưới cả naive — 0.642 -> 0.15 trên
        cat1) — chỉ khác critic ở việc KHÔNG có bước "phát hiện thiếu qua KG"
        và dùng cap riêng (bằng top_k thay vì critic_max_dieu thấp hơn).

        Với MỖI Điều, chỉ lấy TOÀN VĂN CHÍNH Điều đó (qua Qdrant parent chunk)
        — KHÔNG đi theo bất kỳ quan hệ nào trong Knowledge Graph (không
        THAM_CHIEU sang Điều khác, không dò HanhVi/CheTai/ChuThe/...). Đây
        CHÍNH LÀ điểm khác biệt cốt lõi với Kịch bản 3: baseline này chỉ biết
        "trong phạm vi retrieval, Điều nào cần xem đầy đủ", còn Critic Agent
        còn biết dùng KG để tìm thông tin NẰM NGOÀI phạm vi đó (Điều khác
        được tham chiếu, chế tài phụ nằm ở Khoản/Điều khác).
        """
        logger.info("Mở rộng toàn văn Điều trong phạm vi (không dùng quan hệ KG, không lọc thêm)...")
        retrieved_dieu_ids = state.get("article_expand_dieu_ids", [])
        if not retrieved_dieu_ids:
            return {"graph_context": "", "graph_fetched_dieu_ids": []}

        graph_text = ""
        for dieu_id in retrieved_dieu_ids:
            content = self._fetch_parent_content(dieu_id)
            if content:
                graph_text += f"[Toàn văn Điều {dieu_id}]\n{content}\n\n"

        logger.info(f"Full-article expand: {len(graph_text)} ký tự, {len(retrieved_dieu_ids)} Điều.")
        return {"graph_context": graph_text, "graph_fetched_dieu_ids": retrieved_dieu_ids}

    def generate_draft_response(self, state: ChatbotState):
        """
        Kịch bản 3, bước 1: sinh câu trả lời NHÁP chỉ từ top-k chunk thô — y hệt
        đầu vào của Kịch bản 1 (dùng chung nguyên văn prompt với
        generate_single_pass_response qua _build_answer_prompt — bắt buộc
        giống hệt nhau để 3 kịch bản chỉ khác nhau ở NỘI DUNG ngữ cảnh, không
        khác cách sinh câu trả lời).
        """
        logger.info("Critic Agent: sinh câu trả lời nháp từ top-k thuần...")
        query = state["query"]
        context_text = self._format_context_block(state.get("context_texts", []))
        prompt = self._build_answer_prompt(query, context_text)

        resp = self._invoke_llm(prompt, tag="draft")
        return {"draft_response": resp.content}

    def critic_check(self, state: ChatbotState):
        """
        Kịch bản 3, bước 2 — trái tim của khóa luận: dùng Knowledge Graph để
        PHÁT HIỆN xem câu trả lời nháp (dựa trên top-k thuần) có khả năng đang
        thiếu gì không, rồi TỰ ĐI LẤY (fetch) CHÍNH XÁC phần còn thiếu đó — chứ
        KHÔNG bốc tràn lan như Kịch bản 2:

        1. Same-Điều (chế tài): một HanhVi trong Điều đã lấy có CẢ hình phạt
           chính lẫn bổ sung trong graph -> 2 phần này thường nằm ở 2 Khoản
           khác nhau -> fetch toàn văn CHÍNH Điều đó.
        2. Same-Điều (tổng quát): Điều đã lấy có nhiều Khoản/Điểm hơn số phần
           thực sự đã retrieve — KHÔNG giới hạn riêng chế tài, áp dụng cho MỌI
           cấu trúc nhiều Khoản (danh sách miễn trừ, điều kiện, v.v.) mà quan hệ
           HanhVi/CheTai không có mặt trong graph để dò riêng -> fetch toàn văn.
        3. Cross-Điều: Điều đã lấy có tham chiếu (THAM_CHIEU) sang Điều KHÁC mà
           top-k chưa lấy được -> fetch toàn văn Điều kia.
        4. Cross-Điều BẮC CẦU (multi-hop): sau khi fetch xong 1 Điều mới ở check
           #3, quét lại xem CHÍNH Điều mới đó có tham chiếu tiếp sang Điều khác
           nữa không (A -> B -> C), lặp tới khi không còn gì mới hoặc chạm giới
           hạn an toàn (max_hops=2, tổng số Điều fetch <= 2*critic_max_dieu) —
           tránh lặp vô hạn nếu tham chiếu vòng tròn (A<->B) hoặc 1 Điều bị quá
           nhiều Điều khác tham chiếu tới (hub document).

        Cả 4 check trên chỉ phát hiện qua TÍN HIỆU CẤU TRÚC graph (đếm Khoản,
        cạnh THAM_CHIEU/CheTai) — KHÔNG biết nội dung ứng viên có thật liên quan
        tới câu hỏi hay không. Trước khi fetch/bơm vào graph_context, MỖI ứng
        viên còn phải qua `_is_candidate_relevant()` (cổng lọc ngữ nghĩa bằng
        LLM, 1 lệnh gọi nhỏ/ứng viên) — ứng viên bị từ chối sẽ KHÔNG xuất hiện
        trong ngữ cảnh cuối (ghi lại ở critic_report["rejected_by_relevance_gate"]).

        Nếu không phát hiện gì (graph_context rỗng) -> _build_graph() sẽ đi
        thẳng sang finalize_draft, KHÔNG tốn thêm 1 lượt gọi LLM vô ích.

        LƯU Ý: chỉ kiểm tra completeness cho các Điều có điểm >= critic_score_ratio
        * điểm Điều cao nhất (KHÔNG cứng "chỉ top-1", KHÔNG chạy tràn lan cho mọi
        Điều trong top-k) — tránh việc 1 Điều nhiễu (lọt vào top-k với điểm thấp
        hẳn, không thật sự liên quan tới câu hỏi) kéo theo tham chiếu riêng của
        nó, làm loãng/lạc đề câu trả lời cuối; đồng thời vẫn giữ được TẤT CẢ nếu
        có nhiều Điều cùng đạt điểm cao gần nhau (câu hỏi thực sự cần đối chiếu
        nhiều Điều), không bỏ sót như cách cắt cứng theo thứ hạng.
        """
        logger.info("Critic Agent (Neo4j): đối chiếu câu trả lời nháp với Knowledge Graph...")
        query = state["query"]
        all_dieu_ids = state.get("retrieved_dieu_ids", [])
        dieu_scores = state.get("dieu_scores", {})

        # focus_dieu_ids: phạm vi QUÉT completeness — TÍNH CHUNG với
        # expand_full_article qua _compute_focus_dieu_ids (xem docstring
        # hàm đó — bắt buộc 2 kịch bản dùng cùng 1 tập ứng viên để so sánh công bằng).
        # all_dieu_ids: TOÀN BỘ Điều retrieval THỰC SỰ đã lấy — dùng để biết 1
        # Điều tham chiếu tới có sẵn rồi hay chưa. Phải truyền riêng 2 tập này
        # cho check_retrieval_completeness, KHÔNG được dùng chung 1 tập đã lọc —
        # nếu không, 1 Điều bị lọc khỏi focus (vì điểm thấp) nhưng thực ra ĐÃ
        # được retrieve sẽ bị báo nhầm là "thiếu" (bug đã phát hiện qua test thật).
        focus_dieu_ids = self._compute_focus_dieu_ids(all_dieu_ids, dieu_scores)
        logger.info(f"Critic Agent: phạm vi kiểm tra {len(focus_dieu_ids)}/{len(all_dieu_ids)} Điều (score_ratio={self.critic_score_ratio}).")

        if not focus_dieu_ids:
            return {"graph_context": "", "critic_report": {}, "graph_fetched_dieu_ids": []}

        # Đếm số Khoản/Điểm DUY NHẤT (theo chunk_id) thực sự đã retrieve cho mỗi
        # Điều — dùng để phát hiện tổng quát "Điều nhiều Khoản nhưng chưa lấy đủ
        # phần" (không giới hạn riêng chế tài, xem check_retrieval_completeness).
        retrieved_part_counts: Dict[str, set] = {}
        dieu_to_retrieved_text: Dict[str, list] = {}
        for c in state.get("retrieved_chunks", []):
            d = to_dieu_node_id(c["van_ban_id_raw"], c["dieu_id_raw"])
            if d and c.get("chunk_id"):
                retrieved_part_counts.setdefault(d, set()).add(c["chunk_id"])
            if d and c.get("text"):
                dieu_to_retrieved_text.setdefault(d, []).append(c["text"])
        retrieved_part_counts = {d: len(chunk_ids) for d, chunk_ids in retrieved_part_counts.items()}

        # Đếm "tổng số phần" LẤY MAX giữa Neo4j (bên trong check_retrieval_completeness)
        # và Qdrant (self._dieu_child_chunk_count, độc lập, không qua LLM extraction) —
        # phòng Neo4j ingest thiếu cạnh CO_KHOAN cho 1 số Điều (xem
        # _build_dieu_child_chunk_count và find_structurally_incomplete_dieu).
        total_parts_override = {d: self._dieu_child_chunk_count.get(d, 0) for d in focus_dieu_ids}

        report = self.critic_query.check_retrieval_completeness(
            focus_dieu_ids, all_dieu_ids, retrieved_part_counts, total_parts_override
        )
        graph_text = ""
        fetched_dieu_ids: set = set()

        rejected_dieu_ids: list = []

        if report["missing_references"]:
            for ref in report["missing_references"]:
                missing_dieu_id = ref.get("missing_dieu_id")
                if not missing_dieu_id or missing_dieu_id in fetched_dieu_ids:
                    continue
                content = self._fetch_parent_content(missing_dieu_id)
                if content and not self._is_candidate_relevant(query, content):
                    logger.info(f"Relevance gate: BỎ QUA Điều {missing_dieu_id} (missing_reference) — LLM đánh giá không liên quan tới câu hỏi.")
                    rejected_dieu_ids.append(missing_dieu_id)
                    continue
                graph_text += f"[Điều liên quan do Critic Agent (Knowledge Graph) phát hiện — {ref.get('reason', '')}]\n"
                if content:
                    graph_text += f"{content}\n\n"
                    fetched_dieu_ids.add(missing_dieu_id)
                else:
                    graph_text += "(Không lấy được toàn văn — Qdrant parent collection chưa có Điều này.)\n\n"

        if report["compound_penalty_behaviors"]:
            for p in report["compound_penalty_behaviors"]:
                dieu_id = p.get("dieu_id")
                if not dieu_id or dieu_id in fetched_dieu_ids:
                    continue
                content = self._fetch_parent_content(dieu_id)
                # Dieu nay DA co mat trong focus qua retrieval (co chunk that su
                # da retrieve) - relevance gate phai so voi CHINH CAC CHUNK DO
                # (ngan, cu the, kiem tra RIENG LE qua _any_chunk_relevant), KHONG
                # so voi toan van Dieu cat ngan tu dau (gay false-negative loai
                # nham Dieu DUNG - da quan sat o D105/D102) VA KHONG gop chung
                # nhieu chunk lam 1 (gay false-negative khac khi 1 chunk lac de
                # lam nhieu tin hieu chunk dung - da quan sat o D99).
                anchor_texts = dieu_to_retrieved_text.get(dieu_id) or ([content] if content else [])
                if anchor_texts and not self._any_chunk_relevant(query, anchor_texts):
                    logger.info(f"Relevance gate: BỎ QUA Điều {dieu_id} (compound_penalty) — LLM đánh giá không liên quan tới câu hỏi.")
                    rejected_dieu_ids.append(dieu_id)
                    continue
                graph_text += (
                    f"[Toàn văn Điều {dieu_id} do Critic Agent (Knowledge Graph) tự bổ sung — phát hiện hành vi "
                    f"'{p.get('hanh_vi_mo_ta', '')}' có CẢ hình phạt chính lẫn bổ sung/biện pháp khắc phục hậu quả "
                    f"(2 phần này thường nằm ở Khoản khác nhau, top-k dễ chỉ trúng 1 phần) — đảm bảo không bỏ sót]\n"
                )
                if content:
                    graph_text += f"{content}\n\n"
                    fetched_dieu_ids.add(dieu_id)
                else:
                    graph_text += "(Không lấy được toàn văn Điều này từ Qdrant.)\n\n"

        if report["structurally_incomplete_dieu"]:
            for si in report["structurally_incomplete_dieu"]:
                dieu_id = si.get("dieu_id")
                if not dieu_id or dieu_id in fetched_dieu_ids:
                    continue
                content = self._fetch_parent_content(dieu_id)
                # Tuong tu compound_penalty o tren: kiem tra RIENG LE tung chunk
                # DA retrieve (khong gop chung, khong so toan van cat ngan).
                anchor_texts = dieu_to_retrieved_text.get(dieu_id) or ([content] if content else [])
                if anchor_texts and not self._any_chunk_relevant(query, anchor_texts):
                    logger.info(f"Relevance gate: BỎ QUA Điều {dieu_id} (structurally_incomplete) — LLM đánh giá không liên quan tới câu hỏi.")
                    rejected_dieu_ids.append(dieu_id)
                    continue
                graph_text += (
                    f"[Toàn văn Điều {dieu_id} do Critic Agent (Knowledge Graph) tự bổ sung — Điều này có "
                    f"{si.get('total_parts')} Khoản/Điểm nhưng chỉ retrieve được {si.get('retrieved_parts')} phần, "
                    f"có thể còn nội dung liên quan ở Khoản/Điểm khác chưa lấy được]\n"
                )
                if content:
                    graph_text += f"{content}\n\n"
                    fetched_dieu_ids.add(dieu_id)
                else:
                    graph_text += "(Không lấy được toàn văn Điều này từ Qdrant.)\n\n"

        # VÒNG LẶP BỔ SUNG (multi-hop tham chiếu): Điều VỪA fetch ở 3 check trên
        # có thể TỰ NÓ tham chiếu sang 1 Điều khác nữa (A -> B -> C) mà vòng đầu
        # chưa biết tới — find_missing_references() ở trên chỉ quét THAM_CHIEU
        # xuất phát từ focus_dieu_ids BAN ĐẦU, chưa quét từ B (Điều vừa fetch).
        # Sau khi fetch B, quét lại THAM_CHIEU xuất phát từ CHÍNH B để tìm C,
        # lặp tới khi không còn gì mới hoặc chạm max_hops. Đây CHÍNH LÀ phần
        # "đánh giá tiếp, khi nào đủ thì trả kết quả cuối cùng" — nhưng lặp lại
        # CHECK CẤU TRÚC (khách quan, dựa trên graph) chứ KHÔNG lặp lại việc để
        # LLM tự phán xét câu trả lời của chính nó (đã thử kiểu đó ở
        # regenerate_response và REVERT vì model 7B tự vứt bỏ luôn cả thông tin
        # ĐÚNG — xem lịch sử đo). Giới hạn max_hops + MAX_TOTAL_FETCH để tránh
        # lặp vô hạn nếu 2 Điều tham chiếu vòng tròn lẫn nhau (vd A<->B) hoặc 1
        # Điều được tham chiếu bởi quá nhiều Điều khác (hub document).
        all_known_dieu_ids = set(all_dieu_ids) | fetched_dieu_ids
        frontier = list(fetched_dieu_ids)
        multi_hop_refs: list = []
        max_hops = 2
        MAX_TOTAL_FETCH = self.critic_max_dieu * 2
        for _hop in range(max_hops):
            if not frontier or len(fetched_dieu_ids) >= MAX_TOTAL_FETCH:
                break
            deeper_refs = self.critic_query.find_missing_references(frontier, list(all_known_dieu_ids))
            next_frontier = []
            for ref in deeper_refs:
                if len(fetched_dieu_ids) >= MAX_TOTAL_FETCH:
                    break
                missing_dieu_id = ref.get("missing_dieu_id")
                if not missing_dieu_id or missing_dieu_id in all_known_dieu_ids:
                    continue
                all_known_dieu_ids.add(missing_dieu_id)
                content = self._fetch_parent_content(missing_dieu_id)
                if content and not self._is_candidate_relevant(query, content):
                    logger.info(f"Relevance gate: BỎ QUA Điều {missing_dieu_id} (tham chiếu bắc cầu) — LLM đánh giá không liên quan tới câu hỏi.")
                    rejected_dieu_ids.append(missing_dieu_id)
                    continue
                graph_text += f"[Điều liên quan do Critic Agent (Knowledge Graph, tham chiếu bắc cầu từ {', '.join(frontier)}) phát hiện — {ref.get('reason', '')}]\n"
                if content:
                    graph_text += f"{content}\n\n"
                    fetched_dieu_ids.add(missing_dieu_id)
                    next_frontier.append(missing_dieu_id)
                else:
                    graph_text += "(Không lấy được toàn văn — Qdrant parent collection chưa có Điều này.)\n\n"
            multi_hop_refs.extend(deeper_refs)
            frontier = next_frontier

        if multi_hop_refs:
            report["multi_hop_references"] = multi_hop_refs

        # PHÒNG NGỪA "LOST IN THE MIDDLE" liên văn bản (phát hiện qua test thật
        # — case cat4_01 "Luật Dữ liệu 2024 hiệu lực thi hành"): khi ngữ cảnh
        # top-k trộn lẫn NHIỀU văn bản khác nhau có điều khoản CẤU TRÚC GIỐNG
        # NHAU (vd "Hiệu lực thi hành" ở 3 luật khác nhau), draft dễ tự PHỦ
        # NHẬN thông tin ĐÚNG dù nó nằm ngay trong context — chỉ vì bị nhiễu
        # bởi đoạn tương tự của văn bản khác đứng gần đó. Đây KHÔNG phải lỗi
        # "thiếu dữ liệu" (3 check completeness ở trên đúng khi báo is_complete)
        # mà là nguy cơ "hiểu sai dữ liệu đã có" — 1 loại lỗi khác hẳn, cần tín
        # hiệu khác hẳn để phát hiện (không phải hỏi LLM tự nghi ngờ — đã thử
        # kiểu đó ở regenerate_response và REVERT vì phản tác dụng). Tín hiệu
        # dùng ở đây KHÁCH QUAN, không qua LLM: nếu top-k có chunk từ >= 2 văn
        # bản khác nhau VÀ không có gì để fetch qua 4 check trên (sắp đi thẳng
        # finalize_draft, dùng nguyên draft) -> chủ động LẶP LẠI toàn văn Điều
        # điểm cao nhất, buộc quy trình rẽ sang regenerate (sinh lại có ngữ
        # cảnh được củng cố) thay vì dùng thẳng draft chưa được kiểm chứng lại
        # — mô phỏng đúng cơ chế "lặp lại thông tin đúng" mà article_expand có
        # sẵn (luôn re-dump toàn văn), nhưng CÓ ĐIỀU KIỆN, chỉ khi thật sự có
        # nguy cơ nhầm lẫn liên văn bản, không làm tràn lan như article_expand.
        if not fetched_dieu_ids and all_dieu_ids:
            distinct_van_ban = {
                c.get("van_ban_id_raw") for c in state.get("retrieved_chunks", []) if c.get("van_ban_id_raw")
            }
            if len(distinct_van_ban) >= 2:
                top_dieu_id = all_dieu_ids[0]
                content = self._fetch_parent_content(top_dieu_id)
                if content:
                    logger.info(
                        f"Critic Agent: ngữ cảnh trộn {len(distinct_van_ban)} văn bản khác nhau, không phát hiện "
                        f"thiếu gì về cấu trúc — chủ động lặp lại toàn văn Điều {top_dieu_id} để phòng nhầm lẫn."
                    )
                    graph_text += (
                        f"[Toàn văn Điều {top_dieu_id} — Critic Agent CHỦ ĐỘNG LẶP LẠI để phòng ngừa nhầm lẫn: "
                        f"ngữ cảnh có nội dung từ {len(distinct_van_ban)} văn bản khác nhau, dễ nhầm điều khoản "
                        f"cấu trúc giống nhau giữa các văn bản (vd \"Hiệu lực thi hành\")]\n{content}\n\n"
                    )
                    fetched_dieu_ids.add(top_dieu_id)
                    report["reinforced_top_dieu_multi_van_ban"] = top_dieu_id

        if rejected_dieu_ids:
            report["rejected_by_relevance_gate"] = rejected_dieu_ids

        return {"graph_context": graph_text, "critic_report": report, "graph_fetched_dieu_ids": list(fetched_dieu_ids)}

    def regenerate_response(self, state: ChatbotState):
        """
        Kịch bản 3, bước 3: sinh LẠI câu trả lời cuối cùng TỪ ĐẦU bằng đúng
        prompt/cơ chế của generate_single_pass_response — dùng context_texts gốc
        + graph_context Critic Agent vừa fetch/lọc — KHÔNG còn "sửa" draft_response
        bằng cách yêu cầu LLM hợp nhất 2 khối văn bản riêng biệt (draft cũ + phần
        bổ sung mới).

        LƯU Ý (đã thử và bỏ, phát hiện qua test thật): cách cũ đưa CẢ draft_response
        LẪN graph_context, yêu cầu LLM "tích hợp tự nhiên" 2 phần lại — đây là tác
        vụ khó hơn hẳn "đọc ngữ cảnh rồi trả lời thẳng": model 7B phải vừa đọc lại
        draft, vừa đọc graph_context mới, vừa suy luận phần nào đã có/chưa có/
        trùng lặp rồi hợp nhất — dễ ĐÁNH RƠI thông tin dù graph_context đã có ĐỦ
        và ĐÚNG (quan sát cụ thể: graph_context có đủ 9 Khoản Điều 98, gồm cả
        Khoản 5 "buộc thu hồi/hoàn trả tên miền", nhưng câu trả lời HỢP NHẤT lại bỏ
        sót đúng Khoản 5 đó — trong khi sinh lại từ đầu với CÙNG NGỮ CẢNH, y hệt
        cách Kịch bản 1/2 làm, thì KHÔNG bỏ sót). Sinh lại từ đầu bằng chung 1 CƠ
        CHẾ SINH CÂU TRẢ LỜI CUỐI với Kịch bản 1/2 cũng giúp phép so sánh 3 kịch
        bản công bằng hơn: giờ cả 3 chỉ còn khác nhau đúng ở NỘI DUNG ngữ cảnh
        được đưa vào, không còn khác nhau ở CÁCH SINH câu trả lời từ ngữ cảnh đó.

        (Câu dặn "liệt kê đầy đủ chế tài" + "trả lời dứt khoát, không né tránh" đã
        có sẵn trong prompt dùng chung generate_single_pass_response.)
        """
        logger.info("Critic Agent: sinh lại câu trả lời cuối từ ngữ cảnh gốc + phần Critic Agent bổ sung...")
        return self.generate_single_pass_response(state)

    def finalize_draft(self, state: ChatbotState):
        """Critic không phát hiện thiếu gì -> câu trả lời nháp đã đủ, dùng thẳng làm câu trả lời cuối."""
        logger.info("Critic Agent: không phát hiện thiếu — dùng thẳng câu trả lời nháp.")
        draft = state.get("draft_response", "")
        return {"final_response": draft, "messages": [AIMessage(content=draft)]}

    def run(self, query: str, mode: str = "critic") -> Dict[str, Any]:
        """
        mode: "naive" (Kịch bản 1) | "article_expand" (Kịch bản 2) | "critic" (Kịch bản 3 — đề xuất khóa luận).

        Trả về dict đầy đủ để tiện so sánh khi đánh giá khóa luận:
        {final_response, draft_response, retrieved_chunks, retrieved_dieu_ids,
         critic_report, graph_context, mode}
        """
        if mode not in MODES:
            raise ValueError(f"mode phải là một trong {MODES}, nhận được: {mode!r}")

        # Reset đếm token — mỗi run() là 1 câu hỏi độc lập, không cộng dồn qua các câu.
        self._token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "call_count": 0}
        self._token_usage_by_tag = {}

        initial_state = {
            "query": query,
            "mode": mode,
            "messages": [HumanMessage(content=query)],
        }
        result = self.workflow.invoke(initial_state)

        # Token của ĐÚNG lệnh gọi sinh ra câu trả lời cuối cùng (KHÔNG cộng router/
        # gate/draft-bị-bỏ) — chỉ số đúng cho "hiệu quả ngữ cảnh", tách biệt khỏi
        # token_usage (tổng chi phí cả pipeline, xem _invoke_llm). Xác định bằng
        # đường đi THỰC TẾ đã chạy, không suy đoán từ mode:
        # - chit_chat: câu trả lời cuối luôn đến từ handle_chit_chat (tag chit_chat).
        # - naive/article_expand: luôn đến từ generate_single_pass_response (tag
        #   final_generate) — gọi trực tiếp, không có nhánh khác.
        # - critic: đến từ regenerate_response (cũng tag final_generate, vì
        #   regenerate_response gọi lại generate_single_pass_response) NẾU
        #   critic_check phát hiện thiếu; ngược lại (finalize_draft) câu trả lời
        #   cuối CHÍNH LÀ draft_response (tag draft) — không có lệnh gọi nào khác.
        empty_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "call_count": 0}
        by_tag = self._token_usage_by_tag
        if result.get("is_chit_chat"):
            final_answer_usage = by_tag.get("chit_chat", empty_usage)
        elif mode == "critic" and not by_tag.get("final_generate"):
            final_answer_usage = by_tag.get("draft", empty_usage)
        else:
            final_answer_usage = by_tag.get("final_generate", empty_usage)

        return {
            "final_response": result.get("final_response", ""),
            "draft_response": result.get("draft_response", ""),
            "retrieved_chunks": result.get("retrieved_chunks", []),
            "retrieved_dieu_ids": result.get("retrieved_dieu_ids", []),
            "critic_report": result.get("critic_report", {}),
            "token_usage": dict(self._token_usage),
            "final_answer_token_usage": dict(final_answer_usage),
            "graph_context": result.get("graph_context", ""),
            "graph_fetched_dieu_ids": result.get("graph_fetched_dieu_ids", []),
            "mode": mode,
            "is_chit_chat": result.get("is_chit_chat", False),
        }
