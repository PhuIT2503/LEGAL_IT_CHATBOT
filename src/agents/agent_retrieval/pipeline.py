"""Hybrid search dùng chung; nhánh Critic recursive-expand trước generate.

Naive/article_expand giữ nguyên baseline. Critic tái sử dụng tín hiệu
completeness hiện có để lấy đủ toàn Điều và tham chiếu có giới hạn.
"""

from langgraph.graph import StateGraph, START, END

from src.agents.agent_retrieval.state import RetrievalState
from src.agents.agent_retrieval.node_hybrid_search import hybrid_search_node
from src.agents.agent_retrieval.recursive_retrieval import recursive_retrieve
from src.agents.common.cross_encoder_reranker import get_cross_encoder_reranker


class RetrievalAgent:
    def __init__(
        self,
        qdrant_client,
        embedding_model,
        bm25,
        qdrant_child_col: str,
        qdrant_parent_col: str,
        top_k: int,
        prefetch_limit: int,
        article_expand_score_ratio: float,
        dieu_content_store=None,
        critic_query_engine=None,
        critic_score_ratio: float = 0.6,
        critic_max_dieu: int = 4,
        recursive_max_depth: int = 3,
        recursive_max_iterations: int = 5,
    ):
        self.qdrant_client = qdrant_client
        self.embedding_model = embedding_model
        self.bm25 = bm25
        self.qdrant_child_col = qdrant_child_col
        self.qdrant_parent_col = qdrant_parent_col
        self.top_k = top_k
        self.prefetch_limit = prefetch_limit
        self.article_expand_score_ratio = article_expand_score_ratio
        self.dieu_content_store = dieu_content_store
        self.critic_query_engine = critic_query_engine
        self.critic_score_ratio = critic_score_ratio
        self.critic_max_dieu = critic_max_dieu
        self.recursive_max_depth = recursive_max_depth
        self.recursive_max_iterations = recursive_max_iterations
        self.cross_encoder_reranker = get_cross_encoder_reranker()
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(RetrievalState)
        workflow.add_node("hybrid_search", self._node)
        workflow.add_edge(START, "hybrid_search")
        workflow.add_edge("hybrid_search", END)
        return workflow.compile()

    def _node(self, state):
        progress = state.get("progress_callback")
        if progress:
            progress("search")
        result = hybrid_search_node(
            state,
            qdrant_client=self.qdrant_client,
            embedding_model=self.embedding_model,
            bm25=self.bm25,
            qdrant_child_col=self.qdrant_child_col,
            qdrant_parent_col=self.qdrant_parent_col,
            top_k=self.top_k,
            prefetch_limit=self.prefetch_limit,
            article_expand_score_ratio=self.article_expand_score_ratio,
            cross_encoder_reranker=self.cross_encoder_reranker,
        )
        if progress:
            progress("retrieve")

        # Chỉ nhánh Critic (chế độ mặc định của chatbot) recursive
        # retrieve. Hai baseline vẫn giữ nguyên để không phá các phép đo
        # so sánh của khoá luận.
        if (
            state.get("mode") == "critic"
            and self.dieu_content_store is not None
            and self.critic_query_engine is not None
        ):
            if progress:
                progress("analyze")
            expanded = recursive_retrieve(
                {**state, **result},
                dieu_content_store=self.dieu_content_store,
                critic_query_engine=self.critic_query_engine,
                critic_score_ratio=self.critic_score_ratio,
                critic_max_dieu=self.critic_max_dieu,
                max_depth=self.recursive_max_depth,
                max_iterations=self.recursive_max_iterations,
            )
            result.update(expanded)
        else:
            result.setdefault("retrieval_is_complete", True)
            result.setdefault("recursive_retrieval_done", False)
        return result

    def run(self, state: dict) -> dict:
        return self.graph.invoke(state)
