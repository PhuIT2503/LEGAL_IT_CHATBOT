"""
agent_article_expand — 1 job: với mỗi Điều trong article_expand_dieu_ids, lấy
TOÀN VĂN chính Điều đó qua Qdrant (không dùng Knowledge Graph). Đây là Kịch
bản 2 (baseline so sánh), KHÔNG phải cơ chế đề xuất của khóa luận.
"""

from langgraph.graph import StateGraph, START, END

from src.agents.agent_article_expand.state import ArticleExpandState
from src.agents.agent_article_expand.node_expand_article import expand_article_node


class ArticleExpandAgent:
    def __init__(self, dieu_content_store):
        self.dieu_content_store = dieu_content_store
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(ArticleExpandState)
        workflow.add_node("expand_article", self._node)
        workflow.add_edge(START, "expand_article")
        workflow.add_edge("expand_article", END)
        return workflow.compile()

    def _node(self, state):
        return expand_article_node(state, dieu_content_store=self.dieu_content_store)

    def run(self, state: dict) -> dict:
        return self.graph.invoke(state)
