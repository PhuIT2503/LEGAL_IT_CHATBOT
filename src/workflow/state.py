import operator
from typing import Annotated, Any, Dict, List, Sequence, TypedDict

from langchain_core.messages import BaseMessage


class WorkflowState(TypedDict):
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
