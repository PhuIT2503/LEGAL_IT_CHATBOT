import os
import logging
from typing import TypedDict, Annotated, Sequence
import operator

# LangChain / LangGraph imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Local imports
from src.knowledge_graph.neo4j_ingest import Neo4jGraphIngestor
from src.agent.critic_query import CriticQueryEngine

logger = logging.getLogger(__name__)

# State definition for LangGraph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query: str
    retrieved_dieu_ids: list[str]  # id Dieu chuẩn Neo4j — xem docstring critic_query.py
    draft_response: str
    critic_feedback: str
    is_complete: bool

class LegalCriticAgent:
    """
    Critic Agent sử dụng LangGraph và Neo4j để đánh giá tính đầy đủ của câu trả lời.
    Hỗ trợ linh hoạt LLM (Gemini, Qwen, OpenAI, v.v.).
    """
    def __init__(self, llm, neo4j_uri="bolt://localhost:7688", neo4j_user="neo4j", neo4j_password="password"):
        self.llm = llm
        
        # Init Neo4j and Query Engine
        self.neo4j_ingestor = Neo4jGraphIngestor(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)
        self.query_engine = CriticQueryEngine(self.neo4j_ingestor)
        
        # Init LangGraph Memory
        self.memory = MemorySaver()
        
        # Build Workflow
        self.workflow = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        
        workflow.add_node("evaluate_with_kg", self.evaluate_with_kg)
        workflow.add_node("generate_feedback", self.generate_feedback)
        
        workflow.add_edge(START, "evaluate_with_kg")
        workflow.add_edge("evaluate_with_kg", "generate_feedback")
        workflow.add_edge("generate_feedback", END)
        
        return workflow.compile(checkpointer=self.memory)

    def evaluate_with_kg(self, state: AgentState):
        """
        Dùng Neo4j để kiểm tra xem các Điều đã retrieve có thiếu hình phạt bổ sung
        hoặc tham chiếu chéo quan trọng không.
        """
        logger.info("Evaluating retrieved context using Neo4j...")
        dieu_ids = state.get("retrieved_dieu_ids", [])

        if not dieu_ids:
            return {"critic_feedback": "Không có Điều nào được xác định từ retrieval."}

        # Tìm hành vi có chế tài kép (chính+bổ sung, thường nằm ở 2 Khoản khác nhau) bằng Neo4j
        compound_penalties = self.query_engine.find_compound_penalty_behaviors(dieu_ids)
        missing_refs = self.query_engine.find_missing_references(dieu_ids)

        feedback_context = ""
        if compound_penalties:
            feedback_context += "Hành vi có chế tài kép (chính+bổ sung) — cần lấy lại toàn văn Điều để đủ cả 2 phần:\n"
            for p in compound_penalties:
                feedback_context += (
                    f"- Hành vi: {p.get('hanh_vi_mo_ta', 'Unknown')} (Điều {p.get('dieu_id', '')})\n"
                    f"  + Chính: {p.get('mo_ta_hinh_phat_chinh')}\n"
                    f"  + Bổ sung: {p.get('mo_ta_hinh_phat_bo_sung')}\n"
                )

        if missing_refs:
            feedback_context += "\nThiếu các Điều được tham chiếu:\n"
            for ref in missing_refs:
                feedback_context += f"- {ref.get('reason', '')}\n"

        if not feedback_context:
            feedback_context = "Các Điều hiện tại đã bao phủ đầy đủ thông tin hình phạt và không thiếu tham chiếu chéo quan trọng."

        return {"critic_feedback": feedback_context}

    def generate_feedback(self, state: AgentState):
        """
        Dùng LLM (Qwen/Gemini) để nhận xét cuối cùng dựa trên Graph context và draft_response.
        """
        logger.info("Generating LLM critic feedback...")
        query = state.get("query", "")
        draft = state.get("draft_response", "")
        graph_feedback = state.get("critic_feedback", "")
        
        system_prompt = (
            "Bạn là một chuyên gia pháp lý (Critic Agent). Nhiệm vụ của bạn là đánh giá xem Bản nháp câu trả lời "
            "có giải quyết đầy đủ câu hỏi của người dùng hay không, dựa vào Báo cáo từ Knowledge Graph.\n\n"
            f"Báo cáo từ Knowledge Graph:\n{graph_feedback}\n\n"
            "Nếu Báo cáo chỉ ra có thiếu hình phạt bổ sung, hãy yêu cầu hệ thống bổ sung nó. "
            "Nếu Bản nháp đã đủ, hãy khen ngợi và phê duyệt."
        )
        
        user_message = f"Câu hỏi của người dùng: {query}\n\nBản nháp câu trả lời:\n{draft}"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
        
        response = self.llm.invoke(messages)
        
        is_complete = "thiếu" not in graph_feedback.lower() and "thiếu" not in response.content.lower()
        
        return {
            "messages": [AIMessage(content=response.content)],
            "is_complete": is_complete
        }

    def run(self, query: str, retrieved_dieu_ids: list[str], draft_response: str, thread_id: str = "default_thread"):
        """
        Chạy Critic Agent pipeline. Sử dụng thread_id để quản lý short-term memory (LangGraph Checkpointer).

        Args:
            retrieved_dieu_ids: id Dieu chuẩn Neo4j đã retrieve (vd "ngh_nh_15_2020_..._D90"),
                quy đổi từ payload Qdrant bằng graph_builder.to_dieu_node_id().
        """
        config = {"configurable": {"thread_id": thread_id}}
        initial_state = {
            "query": query,
            "retrieved_dieu_ids": retrieved_dieu_ids,
            "draft_response": draft_response,
            "messages": []
        }
        
        # Execute workflow
        result = self.workflow.invoke(initial_state, config=config)
        return result

# ==========================================
# EXAMPLE USAGE (Linh hoạt cấu hình LLM)
# ==========================================
if __name__ == "__main__":
    # Để dùng Gemini:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
    except ImportError:
        # Fallback to local Qwen via OpenAI interface if Gemini is not installed
        from langchain_openai import ChatOpenAI
        # Ví dụ kết nối tới Ollama / vLLM local:
        llm = ChatOpenAI(
            model="Qwen2.5-7B-Instruct", 
            openai_api_key="EMPTY", 
            openai_api_base="http://localhost:8000/v1"
        )
    
    # Khởi tạo Critic Agent
    agent = LegalCriticAgent(llm=llm, neo4j_uri="bolt://localhost:7688")
    
    # Test Run
    test_query = "Tôi thông báo sai thông tin bưu chính thì bị phạt như thế nào?"
    test_retrieved_dieu_ids = ["ngh_nh_15_2020_n_cp_s_a_i_b_sung_ngh_nh_14_2022_D5"]
    test_draft = "Bạn sẽ bị phạt tiền từ 3 đến 5 triệu đồng."

    print("--- CHẠY LẦN 1 (Lưu short-term memory theo thread_id='user_123') ---")
    result = agent.run(test_query, test_retrieved_dieu_ids, test_draft, thread_id="user_123")
    print("Critic Feedback:\n", result["messages"][-1].content)
    print("Đủ thông tin chưa:", result["is_complete"])
    
    # Chạy lần 2 để test memory (nếu cần tương tác chat)
    # Lịch sử hội thoại đã được checkpointer của LangGraph lưu trữ tự động.
