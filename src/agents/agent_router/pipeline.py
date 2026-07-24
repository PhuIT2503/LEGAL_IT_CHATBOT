"""
agent_router — 1 job: phân loại câu hỏi chit_chat/legal, và nếu chit_chat thì
tự trả lời luôn (không cần đi qua retrieval/generation). Nội bộ là 1 LangGraph
2 node (router -> chit_chat|END) để giữ đúng 1 entrypoint sạch (.run()) cho
workflow cấp trên gọi vào — router+chit_chat gộp làm 1 lượt gọi duy nhất, tránh
phải re-invoke lại từ đầu (tốn thêm 1 lệnh LLM phân loại vô ích) nếu tách thành
2 node độc lập ở tầng workflow.
"""

from langgraph.graph import StateGraph, START, END

from src.agents.agent_router.state import RouterState
from src.agents.agent_router.node_router import route_query_node
from src.agents.agent_router.node_chit_chat import chit_chat_node


class RouterAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(RouterState)
        workflow.add_node("router", lambda s: route_query_node(s, llm_client=self.llm_client))
        workflow.add_node("chit_chat", lambda s: chit_chat_node(s, llm_client=self.llm_client))
        workflow.add_edge(START, "router")
        workflow.add_conditional_edges(
            "router",
            lambda x: "chit_chat" if x.get("is_chit_chat") else "end",
            {"chit_chat": "chit_chat", "end": END},
        )
        workflow.add_edge("chit_chat", END)
        return workflow.compile()

    def run(self, state: dict) -> dict:
        return self.graph.invoke(state)
