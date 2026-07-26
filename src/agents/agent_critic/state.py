from typing import Any, Dict, List, TypedDict


class CriticState(TypedDict):
    query: str
    retrieved_dieu_ids: List[str]
    dieu_scores: Dict[str, float]
    retrieved_chunks: List[Dict[str, Any]]
    graph_context: str
    critic_report: Dict[str, Any]
    graph_fetched_dieu_ids: List[str]
    recursive_retrieval_done: bool
    retrieval_is_complete: bool
