"""
agent_critic — 1 job: đối chiếu draft_response với Knowledge Graph (Neo4j) để
phát hiện phần còn thiếu (chế tài kép, cấu trúc chưa đủ, tham chiếu chéo Điều
khác — kể cả bắc cầu multi-hop), lọc qua cổng ngữ nghĩa (relevance_gate) rồi
fetch đúng phần thiếu vào graph_context. Đây là đóng góp cốt lõi của khóa luận.

Neo4j là tài nguyên CHỈ agent_critic dùng — không nơi nào khác trong pipeline
chạm tới Neo4j — nên CriticAgent tự xây driver/CriticQueryEngine riêng thay vì
nhận từ workflow cấp trên.
"""

from langgraph.graph import StateGraph, START, END

from src.knowledge_graph.neo4j_ingest import Neo4jGraphIngestor
from src.agents.agent_critic.critic_query import CriticQueryEngine
from src.agents.agent_critic.state import CriticState
from src.agents.agent_critic.node_critic_check import critic_check_node


class CriticAgent:
    def __init__(
        self,
        llm_client,
        dieu_content_store,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_pass: str,
        critic_score_ratio: float = 0.6,
        critic_max_dieu: int = 4,
    ):
        self.llm_client = llm_client
        self.dieu_content_store = dieu_content_store
        self.critic_score_ratio = critic_score_ratio
        self.critic_max_dieu = critic_max_dieu

        self.neo4j_ingestor = Neo4jGraphIngestor(uri=neo4j_uri, user=neo4j_user, password=neo4j_pass)
        self.critic_query = CriticQueryEngine(self.neo4j_ingestor)

        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(CriticState)
        workflow.add_node("critic_check", self._node)
        workflow.add_edge(START, "critic_check")
        workflow.add_edge("critic_check", END)
        return workflow.compile()

    def _node(self, state):
        return critic_check_node(
            state,
            llm_client=self.llm_client,
            dieu_content_store=self.dieu_content_store,
            critic_query_engine=self.critic_query,
            critic_score_ratio=self.critic_score_ratio,
            critic_max_dieu=self.critic_max_dieu,
        )

    def run(self, state: dict) -> dict:
        return self.graph.invoke(state)
