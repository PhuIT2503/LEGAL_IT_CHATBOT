"""
agent_retrieval — 1 job: hybrid search (dense + BM25 RRF) lấy top-k chunk thuần,
KHÔNG expand, KHÔNG dùng Knowledge Graph. Dùng chung bởi cả 3 kịch bản
(naive/article_expand/critic) để đảm bảo so sánh công bằng.
"""

from langgraph.graph import StateGraph, START, END

from src.agents.agent_retrieval.state import RetrievalState
from src.agents.agent_retrieval.node_hybrid_search import hybrid_search_node


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
    ):
        self.qdrant_client = qdrant_client
        self.embedding_model = embedding_model
        self.bm25 = bm25
        self.qdrant_child_col = qdrant_child_col
        self.qdrant_parent_col = qdrant_parent_col
        self.top_k = top_k
        self.prefetch_limit = prefetch_limit
        self.article_expand_score_ratio = article_expand_score_ratio
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(RetrievalState)
        workflow.add_node("hybrid_search", self._node)
        workflow.add_edge(START, "hybrid_search")
        workflow.add_edge("hybrid_search", END)
        return workflow.compile()

    def _node(self, state):
        return hybrid_search_node(
            state,
            qdrant_client=self.qdrant_client,
            embedding_model=self.embedding_model,
            bm25=self.bm25,
            qdrant_child_col=self.qdrant_child_col,
            qdrant_parent_col=self.qdrant_parent_col,
            top_k=self.top_k,
            prefetch_limit=self.prefetch_limit,
            article_expand_score_ratio=self.article_expand_score_ratio,
        )

    def run(self, state: dict) -> dict:
        return self.graph.invoke(state)
