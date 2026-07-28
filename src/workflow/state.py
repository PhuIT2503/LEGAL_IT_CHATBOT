import operator
from typing import Annotated, Any, Callable, Dict, List, Sequence, TypedDict

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
    context_records: List[Dict[str, Any]]
    behavior_profile: Dict[str, List[str]]
    scenario_fact_state: Dict[str, Any]
    article_expand_dieu_ids: List[str]
    draft_response: str
    critic_report: Dict[str, Any]
    graph_context: str
    graph_fetched_dieu_ids: List[str]
    final_response: str
    retrieval_is_complete: bool
    recursive_retrieval_done: bool
    progress_callback: Callable[[str], None]
    expanded_query: str
    query_expansion_terms: List[str]
    retrieval_contract: Dict[str, Any]
    retrieval_is_relevant: bool
    retrieval_relevance: float
    retrieval_decisions: List[Dict[str, Any]]
    answer_assessment: Dict[str, Any]
    generation_payload: Dict[str, Any]
