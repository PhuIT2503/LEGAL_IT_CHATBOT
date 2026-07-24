"""
agent_generation — 1 job: sinh câu trả lời từ ngữ cảnh (context_texts +
graph_context tùy chọn). Có 2 entrypoint riêng biệt (không phải 1 node
tham số hóa if/else) vì đây là 2 THỜI ĐIỂM khác nhau trong pipeline dùng
CHUNG 1 cơ chế sinh câu trả lời:
- run_draft(): sinh câu trả lời NHÁP (Kịch bản 3, trước khi Critic Agent kiểm
  tra) — chỉ dùng context_texts, chưa có graph_context.
- run_final(): sinh câu trả lời CUỐI CÙNG — dùng trực tiếp bởi naive/
  article_expand, và dùng lại bởi critic ở bước "regenerate" (sinh lại từ
  đầu bằng context_texts gốc + graph_context Critic Agent vừa bổ sung).
"""

from langgraph.graph import StateGraph, START, END

from src.agents.agent_generation.state import GenerationState
from src.agents.agent_generation.node_generate_draft import generate_draft_node
from src.agents.agent_generation.node_generate_final import generate_final_node


class GenerationAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self._draft_graph = self._build_graph(generate_draft_node)
        self._final_graph = self._build_graph(generate_final_node)

    def _build_graph(self, node_fn):
        workflow = StateGraph(GenerationState)
        workflow.add_node("generate", lambda s: node_fn(s, llm_client=self.llm_client))
        workflow.add_edge(START, "generate")
        workflow.add_edge("generate", END)
        return workflow.compile()

    def run_draft(self, state: dict) -> dict:
        return self._draft_graph.invoke(state)

    def run_final(self, state: dict) -> dict:
        return self._final_graph.invoke(state)
