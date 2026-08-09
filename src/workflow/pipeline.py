import logging
import time
from typing import Any, Callable, Dict, Optional

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage

from src.retrieval.qdrant_hybrid_search import _make_client
from src.retrieval.bm25_sparse import BM25SparseVectorizer, bm25_index_path
from src.embedding.embedding_model import load_embedding_model

from src.agents.common.llm_client import LLMClient
from src.agents.common.dieu_content_store import DieuContentStore
from src.agents.agent_router.pipeline import RouterAgent
from src.agents.agent_retrieval.pipeline import RetrievalAgent
from src.agents.agent_article_expand.pipeline import ArticleExpandAgent
from src.agents.agent_generation.pipeline import GenerationAgent
from src.agents.agent_critic.pipeline import CriticAgent

from src.workflow.state import WorkflowState
from src.workflow.node_agent_router import make_node_agent_router
from src.workflow.node_agent_retrieval import make_node_agent_retrieval
from src.workflow.node_agent_article_expand import make_node_agent_article_expand
from src.workflow.node_agent_generate_draft import make_node_agent_generate_draft
from src.workflow.node_agent_critic import make_node_agent_critic
from src.workflow.node_agent_generate_final import make_node_agent_generate_final
from src.workflow.node_agent_regenerate import make_node_agent_regenerate
from src.workflow.node_finalize_draft import finalize_draft_node

logger = logging.getLogger(__name__)

MODES = ("naive", "article_expand", "critic")


class ChatbotWorkflow:
    """
    Luồng RAG pháp luật hỗ trợ 3 kịch bản (để đánh giá A/B/C trong khóa luận),
    lắp ráp từ 5 agent độc lập (src/agents/agent_*) — mỗi agent làm đúng 1 việc,
    tự có state/pipeline (LangGraph) riêng. Workflow này chỉ chịu trách nhiệm:
    dựng tài nguyên dùng chung 1 LẦN DUY NHẤT (embedding model, Qdrant client,
    BM25, LLMClient, DieuContentStore), khởi tạo từng agent, và nối chúng lại
    thành 1 graph cấp trên theo đúng topology dưới đây.

    - mode="naive"       (Kịch bản 1 — RAG truyền thống, baseline):
      agent_retrieval (top-k thuần) -> agent_generation trả lời thẳng. Không
      graph, không critic. Kỳ vọng: dễ thiếu hình phạt bổ sung/điều khoản dẫn
      chiếu nằm ngoài top-k.

    - mode="article_expand" (Kịch bản 2 — RAG mở rộng toàn Điều, KHÔNG dùng
      quan hệ Knowledge Graph, KHÔNG có Critic Agent):
      agent_retrieval (đủ top_k Điều KHÁC NHAU đạt ngưỡng điểm
      article_expand_score_ratio) -> agent_article_expand lấy TOÀN VĂN CHÍNH
      từng Điều (không đi theo bất kỳ quan hệ KG nào) -> agent_generation trả
      lời 1 lần. Mục đích: cô lập đúng 1 biến số so với Kịch bản 3 — "mở rộng
      lên toàn Điều" tự nó KHÔNG cần Knowledge Graph, nên Kịch bản này xử lý
      tốt câu hỏi mà thông tin nằm ĐẦY ĐỦ trong 1 Điều nhưng KHÔNG tự lấy được
      chế tài kép/tham chiếu nằm ở Điều KHÁC — đúng khoảng trống mà Kịch bản 3
      phải lấp bằng KG.

    - mode="critic"      (Kịch bản 3 — đề xuất của khóa luận):
      agent_retrieval (top-k thuần) -> agent_generation sinh câu trả lời NHÁP
      -> agent_critic (KG) đối chiếu nháp với graph, CHỈ khi phát hiện thật sự
      thiếu mới xét bốc phần đó -> mỗi ứng viên còn phải qua CỔNG LỌC NGỮ NGHĨA
      bằng LLM trước khi thực sự bơm vào ngữ cảnh -> agent_generation sinh lại
      câu trả lời bằng đúng phần đã qua lọc. Nếu không phát hiện/không ứng viên
      nào qua được cổng lọc thì dùng thẳng câu trả lời nháp — không tốn thêm 1
      lượt LLM vô ích. Đây là đóng góp cốt lõi của khóa luận: vừa đủ recall
      (nhờ KG) vừa giữ ngữ cảnh gọn VÀ SẠCH (chỉ bốc đúng phần thiếu VÀ thật sự
      liên quan, không blind-dump như Kịch bản 2).
    """

    def __init__(
        self,
        llm,
        qdrant_child_col: str = "legal_child_chunks_base",
        qdrant_parent_col: str = "legal_parent_chunks_base",
        # Mặc định đọc Qdrant từ container (service qdrant trong docker-compose.yml,
        # cổng 6333 ở máy host). qdrant_path chỉ còn là đường lui cho chế độ
        # embedded/local-file (vd chạy đánh giá offline trên Colab, không có server)
        # — CHỈ dùng khi qdrant_url = None.
        qdrant_url: Optional[str] = "http://localhost:6333",
        qdrant_path: str = "data/.qdrant",
        # BM25 index luôn là file cục bộ (không nằm trong Qdrant), mặc định
        # data/bm25/<qdrant_child_col>.bm25.json.
        bm25_dir: Optional[str] = "data/bm25",
        embedding_model_name: str = "data/ai_vietnamese_embedding_v2_finetuned_final",
        embedding_device: str = "cpu",
        embedding_max_seq_length: int = 256,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_pass: str = "legal_kg_2024",
        top_k: int = 5,
        prefetch_limit: int = 20,
        critic_score_ratio: float = 0.6,
        critic_max_dieu: int = 4,
        article_expand_score_ratio: float = 0.4,
        skip_router: bool = False,
    ):
        # skip_router=True: bỏ hẳn bước phân loại chit_chat/legal, mọi câu hỏi
        # đi thẳng vào retrieval — dùng khi CHẮC CHẮN không có câu chit-chat
        # nào (vd chạy đánh giá trên test set thuần pháp luật), tránh tốn 1
        # lệnh gọi LLM/câu vô ích và loại rủi ro router phân loại nhầm. Chat
        # tương tác thật (run_chatbot.py) vẫn giữ router mặc định.
        self.skip_router = skip_router
        self.top_k = top_k

        # Thời gian chạy THỰC TẾ của từng node trong lần run() gần nhất (giây) —
        # phục vụ chế độ Dev mode của app.py: hiện đúng những bước ĐÃ chạy (mỗi
        # kịch bản đi 1 nhánh khác nhau) kèm chi phí thời gian. Ghi vào thuộc
        # tính instance là an toàn vì mỗi pipeline chỉ chạy 1 câu hỏi tại 1 thời
        # điểm (app.py giữ khóa riêng theo embedding_key quanh cả lượt run).
        self._node_timings: Dict[str, float] = {}

        # Client/model/bm25 load 1 LẦN DUY NHẤT khi khởi tạo, dùng lại cho
        # mọi câu hỏi trong phiên — tránh load lại embedding model mỗi query.
        qdrant_client = _make_client(qdrant_path, qdrant_url)
        embedding_model = load_embedding_model(
            embedding_model_name, device=embedding_device, max_seq_length=embedding_max_seq_length
        )
        bm25_file = bm25_index_path(bm25_dir or qdrant_path, qdrant_child_col)
        try:
            bm25 = BM25SparseVectorizer.load(bm25_file)
        except FileNotFoundError:
            logger.warning(
                f"Không tìm thấy BM25 index tại {bm25_file} — hybrid search sẽ chỉ dùng dense "
                f"vector. Hãy chạy src/indexing/qdrant_ingest.py (hoặc scripts/migrate_qdrant_to_server.py) trước."
            )
            bm25 = None

        self._llm_client = LLMClient(llm)
        dieu_content_store = DieuContentStore(qdrant_client, qdrant_parent_col, qdrant_child_col)

        self._router_agent = RouterAgent(self._llm_client)
        self._retrieval_agent = RetrievalAgent(
            qdrant_client, embedding_model, bm25,
            qdrant_child_col, qdrant_parent_col,
            top_k, prefetch_limit, article_expand_score_ratio,
        )
        self._article_expand_agent = ArticleExpandAgent(dieu_content_store)
        self._generation_agent = GenerationAgent(self._llm_client)
        self._critic_agent = CriticAgent(
            self._llm_client, dieu_content_store,
            neo4j_uri, neo4j_user, neo4j_pass,
            critic_score_ratio=critic_score_ratio, critic_max_dieu=critic_max_dieu,
        )

        self.workflow = self._build_graph()

    @property
    def llm(self):
        return self._llm_client.llm

    @llm.setter
    def llm(self, value):
        self._llm_client.llm = value

    def _timed(self, name: str, node: Callable) -> Callable:
        """Bọc 1 node để ghi lại thời gian chạy — KHÔNG đổi input/output của node.

        Nhờ đó Dev mode biết được ĐÚNG những node nào đã thực sự chạy trong lượt
        vừa rồi (mỗi kịch bản/nhánh rẽ đi qua tập node khác nhau) mà không phải
        suy đoán lại từ mode.
        """
        def wrapped(state):
            start = time.perf_counter()
            try:
                return node(state)
            finally:
                self._node_timings[name] = time.perf_counter() - start
        return wrapped

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(WorkflowState)

        if not self.skip_router:
            workflow.add_node("router", self._timed("router", make_node_agent_router(self._router_agent)))
        workflow.add_node("retrieval", self._timed("retrieval", make_node_agent_retrieval(self._retrieval_agent)))
        workflow.add_node("generate_single_pass", self._timed("generate_single_pass", make_node_agent_generate_final(self._generation_agent)))
        workflow.add_node("expand_article", self._timed("expand_article", make_node_agent_article_expand(self._article_expand_agent)))
        workflow.add_node("generate_draft", self._timed("generate_draft", make_node_agent_generate_draft(self._generation_agent)))
        workflow.add_node("critic_check", self._timed("critic_check", make_node_agent_critic(self._critic_agent)))
        workflow.add_node("regenerate", self._timed("regenerate", make_node_agent_regenerate(self._generation_agent)))
        workflow.add_node("finalize_draft", self._timed("finalize_draft", finalize_draft_node))

        if self.skip_router:
            workflow.add_edge(START, "retrieval")
        else:
            workflow.add_edge(START, "router")
            workflow.add_conditional_edges(
                "router",
                lambda x: "chit_chat" if x.get("is_chit_chat") else "retrieval",
                {"chit_chat": END, "retrieval": "retrieval"},
            )

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

    def run(self, query: str, mode: str = "critic") -> Dict[str, Any]:
        """
        mode: "naive" (Kịch bản 1) | "article_expand" (Kịch bản 2) | "critic" (Kịch bản 3 — đề xuất khóa luận).

        Trả về dict đầy đủ để tiện so sánh khi đánh giá khóa luận:
        {final_response, draft_response, retrieved_chunks, retrieved_dieu_ids,
         critic_report, graph_context, mode}
        """
        if mode not in MODES:
            raise ValueError(f"mode phải là một trong {MODES}, nhận được: {mode!r}")

        # Reset đếm token + thời gian node — mỗi run() là 1 câu hỏi độc lập,
        # không cộng dồn qua các câu.
        self._llm_client.reset_usage()
        self._node_timings = {}
        run_started_at = time.perf_counter()

        initial_state = {
            "query": query,
            "mode": mode,
            "messages": [HumanMessage(content=query)],
        }
        result = self.workflow.invoke(initial_state)

        # Token của ĐÚNG lệnh gọi sinh ra câu trả lời cuối cùng (KHÔNG cộng router/
        # gate/draft-bị-bỏ) — chỉ số đúng cho "hiệu quả ngữ cảnh", tách biệt khỏi
        # token_usage (tổng chi phí cả pipeline). Xác định bằng đường đi THỰC TẾ
        # đã chạy, không suy đoán từ mode:
        # - chit_chat: câu trả lời cuối luôn đến từ agent_router (tag chit_chat).
        # - naive/article_expand: luôn đến từ generate_single_pass (tag
        #   final_generate) — gọi trực tiếp, không có nhánh khác.
        # - critic: đến từ regenerate (cũng tag final_generate, vì regenerate
        #   gọi lại generate_final) NẾU critic_check phát hiện thiếu; ngược lại
        #   (finalize_draft) câu trả lời cuối CHÍNH LÀ draft_response (tag draft)
        #   — không có lệnh gọi nào khác.
        empty_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "call_count": 0}
        by_tag = self._llm_client.token_usage_by_tag
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
            "token_usage": dict(self._llm_client.token_usage),
            "final_answer_token_usage": dict(final_answer_usage),
            "graph_context": result.get("graph_context", ""),
            "graph_fetched_dieu_ids": result.get("graph_fetched_dieu_ids", []),
            "mode": mode,
            "is_chit_chat": result.get("is_chit_chat", False),
            # 3 trường dưới CHỈ để quan sát/gỡ lỗi (Dev mode của app.py, phân
            # tích chi phí khi đánh giá) — không tham gia vào logic trả lời.
            "token_usage_by_tag": {tag: dict(usage) for tag, usage in by_tag.items()},
            "node_timings": dict(self._node_timings),
            "elapsed_seconds": time.perf_counter() - run_started_at,
        }
