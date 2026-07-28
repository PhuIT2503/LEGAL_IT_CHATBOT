import logging
from typing import Any, Dict, Optional

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage

from src.retrieval.qdrant_hybrid_search import _make_client
from src.retrieval.bm25_sparse import BM25SparseVectorizer, bm25_index_path
from src.llm.embedding_model import load_embedding_model

from src.agents.common.llm_client import LLMClient
from src.agents.common.dieu_content_store import DieuContentStore
from src.agents.common.grounded_validation import (
    INSUFFICIENT_GROUNDS,
    build_grounded_sources,
    build_safe_grounded_fallback,
    build_repair_prompt,
    chunk_markdown_for_streaming,
    render_grounded_answer,
    salvage_grounded_draft,
    validate_grounded_draft,
)
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
      agent_retrieval lấy top-k, sau đó recursive-expand ngay trong agent này:
      hoàn thiện Điều mới lấy một phần và lần theo tham chiếu nhiều tầng
      (có visited/max_depth/max_iterations). Chỉ sau khi completeness được
      xác định mới chuyển sang generation. Node critic phía sau vẫn được
      giữ trong topology/API hiện có, nhưng không fetch lặp lại.
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
        critic_score_ratio: float = 0.6,
        critic_max_dieu: int = 4,
        article_expand_score_ratio: float = 0.4,
        recursive_max_depth: int = 3,
        recursive_max_iterations: int = 5,
        grounding_repair_attempts: int = 1,
        skip_router: bool = False,
    ):
        # skip_router=True: bỏ hẳn bước phân loại chit_chat/legal, mọi câu hỏi
        # đi thẳng vào retrieval — dùng khi CHẮC CHẮN không có câu chit-chat
        # nào (vd chạy đánh giá trên test set thuần pháp luật), tránh tốn 1
        # lệnh gọi LLM/câu vô ích và loại rủi ro router phân loại nhầm. Chat
        # tương tác thật (run_chatbot.py) vẫn giữ router mặc định.
        self.skip_router = skip_router
        self.top_k = top_k
        self.grounding_repair_attempts = max(0, grounding_repair_attempts)

        # Client/model/bm25 load 1 LẦN DUY NHẤT khi khởi tạo, dùng lại cho
        # mọi câu hỏi trong phiên — tránh load lại embedding model mỗi query.
        qdrant_client = _make_client(qdrant_path, qdrant_url)
        embedding_model = load_embedding_model(
            embedding_model_name, device=embedding_device, max_seq_length=embedding_max_seq_length
        )
        try:
            bm25 = BM25SparseVectorizer.load(bm25_index_path(qdrant_path, qdrant_child_col))
        except FileNotFoundError:
            logger.warning(
                f"Không tìm thấy BM25 index tại {bm25_index_path(qdrant_path, qdrant_child_col)} — "
                f"hybrid search sẽ chỉ dùng dense vector. Hãy chạy qdrant_local_ingest.py trước."
            )
            bm25 = None

        self._llm_client = LLMClient(llm)
        dieu_content_store = DieuContentStore(qdrant_client, qdrant_parent_col, qdrant_child_col)

        self._router_agent = RouterAgent(self._llm_client)
        self._critic_agent = CriticAgent(
            self._llm_client, dieu_content_store,
            neo4j_uri, neo4j_user, neo4j_pass,
            critic_score_ratio=critic_score_ratio, critic_max_dieu=critic_max_dieu,
        )
        self._retrieval_agent = RetrievalAgent(
            qdrant_client, embedding_model, bm25,
            qdrant_child_col, qdrant_parent_col,
            top_k, prefetch_limit, article_expand_score_ratio,
            dieu_content_store=dieu_content_store,
            critic_query_engine=self._critic_agent.critic_query,
            critic_score_ratio=critic_score_ratio,
            critic_max_dieu=critic_max_dieu,
            recursive_max_depth=recursive_max_depth,
            recursive_max_iterations=recursive_max_iterations,
        )
        self._article_expand_agent = ArticleExpandAgent(dieu_content_store)
        self._generation_agent = GenerationAgent(self._llm_client)

        self.workflow = self._build_graph()

    @property
    def llm(self):
        return self._llm_client.llm

    @llm.setter
    def llm(self, value):
        self._llm_client.llm = value

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(WorkflowState)

        if not self.skip_router:
            workflow.add_node("router", make_node_agent_router(self._router_agent))
        workflow.add_node("retrieval", make_node_agent_retrieval(self._retrieval_agent))
        workflow.add_node("generate_single_pass", make_node_agent_generate_final(self._generation_agent))
        workflow.add_node("expand_article", make_node_agent_article_expand(self._article_expand_agent))
        workflow.add_node("generate_draft", make_node_agent_generate_draft(self._generation_agent))
        workflow.add_node("critic_check", make_node_agent_critic(self._critic_agent))
        workflow.add_node("regenerate", make_node_agent_regenerate(self._generation_agent))
        workflow.add_node("finalize_draft", finalize_draft_node)

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

        # Kịch bản 3: recursive completeness đã nằm bên trong retrieval
        # trước generate. Giữ các node/edge cũ để không phá API và dữ
        # liệu đánh giá; critic_check sẽ no-op khi cờ recursive_retrieval_done.
        workflow.add_edge("generate_draft", "critic_check")
        workflow.add_conditional_edges(
            "critic_check",
            lambda x: "regenerate" if x.get("graph_context") else "finalize_draft",
            {"regenerate": "regenerate", "finalize_draft": "finalize_draft"},
        )
        workflow.add_edge("regenerate", END)
        workflow.add_edge("finalize_draft", END)

        return workflow.compile()

    def run(
        self,
        query: str,
        mode: str = "critic",
        *,
        stream_callback=None,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """
        mode: "naive" (Kịch bản 1) | "article_expand" (Kịch bản 2) | "critic" (Kịch bản 3 — đề xuất khóa luận).

        Trả về dict đầy đủ để tiện so sánh khi đánh giá khóa luận:
        {final_response, draft_response, retrieved_chunks, retrieved_dieu_ids,
         critic_report, graph_context, mode}
        """
        if mode not in MODES:
            raise ValueError(f"mode phải là một trong {MODES}, nhận được: {mode!r}")

        # Reset đếm token — mỗi run() là 1 câu hỏi độc lập, không cộng dồn qua các câu.
        self._llm_client.reset_usage()
        # Chit-chat không có kết luận pháp lý nên có thể stream trực tiếp.
        # Với câu hỏi pháp luật, chỉ stream bản đã qua grounding validator ở
        # cuối hàm; người dùng không bao giờ nhìn thấy hallucination tạm thời.
        stream_tags = {"chit_chat"}
        self._llm_client.set_stream_callback(stream_callback, tags=stream_tags)

        initial_state = {
            "query": query,
            "mode": mode,
            "messages": [HumanMessage(content=query)],
            "progress_callback": progress_callback,
        }
        try:
            result = self.workflow.invoke(initial_state)

            if not result.get("is_chit_chat"):
                if progress_callback:
                    progress_callback("validate")

                # graph_context chỉ khác rỗng ở đường regenerate cũ. Gộp vào
                # cùng grounding pool để validator thấy đúng mọi đoạn model đã đọc.
                context_texts = list(result.get("context_texts", []))
                if result.get("graph_context"):
                    context_texts.append(result["graph_context"])
                # Generation và validator phải dựng cùng một mapping SOURCE_ID.
                # Expansion chỉ tăng candidate recall, không được thay query
                # dùng để chọn/rank nguồn cuối cùng.
                sources = build_grounded_sources(
                    context_texts,
                    query=query,
                )
                candidate = (result.get("final_response") or "").strip()
                generation_candidates = [candidate]
                retrieval_is_complete = result.get("retrieval_is_complete", True)

                validation = validate_grounded_draft(
                    candidate,
                    sources,
                    is_complete=retrieval_is_complete,
                    query=query,
                )
                for attempt in range(self.grounding_repair_attempts):
                    if validation.is_valid:
                        break
                    logger.debug(
                        "Grounding validation lỗi (lượt %s): %s",
                        attempt + 1,
                        "; ".join(validation.issues),
                    )
                    repair_prompt = build_repair_prompt(
                        query=query,
                        draft=candidate,
                        issues=validation.issues,
                        sources=sources,
                        is_complete=result.get("retrieval_is_complete", True),
                    )
                    candidate = self._llm_client.invoke(
                        repair_prompt, tag="grounding_repair"
                    ).content.strip()
                    generation_candidates.append(candidate)
                    validation = validate_grounded_draft(
                        candidate,
                        sources,
                        is_complete=retrieval_is_complete,
                        query=query,
                    )

                if not validation.is_valid:
                    logger.warning(
                        "Grounding validation vẫn lỗi sau repair; thử giữ các block hợp lệ: %s",
                        "; ".join(validation.issues),
                    )
                    salvaged_candidates = [
                        salvage_grounded_draft(
                            query=query,
                            draft=draft_candidate,
                            sources=sources,
                        )
                        for draft_candidate in generation_candidates
                    ]
                    valid_salvaged = [
                        salvaged
                        for salvaged in salvaged_candidates
                        if salvaged != INSUFFICIENT_GROUNDS
                        and validate_grounded_draft(
                            salvaged,
                            sources,
                            is_complete=retrieval_is_complete,
                            query=query,
                        ).is_valid
                    ]
                    if valid_salvaged:
                        # Nhiều block hợp lệ hơn = giữ được nhiều căn cứ hơn.
                        candidate = max(valid_salvaged, key=lambda value: value.count("\n### "))
                        validation = validate_grounded_draft(
                            candidate,
                            sources,
                            is_complete=retrieval_is_complete,
                            query=query,
                        )
                    else:
                        candidate = build_safe_grounded_fallback(
                            query=query,
                            sources=sources,
                            is_complete=retrieval_is_complete,
                        )

                result["final_response"] = render_grounded_answer(
                    candidate,
                    sources,
                    is_complete=result.get("retrieval_is_complete", True),
                    answer_assessment=result.get("answer_assessment"),
                )

                if stream_callback:
                    for chunk in chunk_markdown_for_streaming(result["final_response"]):
                        stream_callback(chunk)

            if progress_callback:
                progress_callback("complete")
        finally:
            self._llm_client.set_stream_callback(None)

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
        elif by_tag.get("grounding_repair"):
            final_answer_usage = by_tag.get("grounding_repair", empty_usage)
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
            "retrieval_is_complete": result.get("retrieval_is_complete", True),
            "retrieval_is_relevant": result.get("retrieval_is_relevant", True),
            "retrieval_relevance": result.get("retrieval_relevance", 0.0),
            "expanded_query": result.get("expanded_query", query),
        }
