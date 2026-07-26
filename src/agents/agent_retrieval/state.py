from typing import Any, Callable, Dict, List, TypedDict


class RetrievalState(TypedDict):
    query: str
    retrieved_chunks: List[Dict[str, Any]]
    retrieved_dieu_ids: List[str]
    dieu_scores: Dict[str, float]
    context_texts: List[str]
    context_records: List[Dict[str, Any]]
    behavior_profile: Dict[str, List[str]]
    article_expand_dieu_ids: List[str]
    mode: str
    retrieval_is_complete: bool
    recursive_retrieval_done: bool
    critic_report: Dict[str, Any]
    graph_context: str
    graph_fetched_dieu_ids: List[str]
    progress_callback: Callable[[str], None]
    expanded_query: str
    query_expansion_terms: List[str]
    retrieval_is_relevant: bool
    retrieval_relevance: float
