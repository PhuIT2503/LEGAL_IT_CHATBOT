from typing import Any, Dict, List, TypedDict


class RetrievalState(TypedDict):
    query: str
    retrieved_chunks: List[Dict[str, Any]]
    retrieved_dieu_ids: List[str]
    dieu_scores: Dict[str, float]
    context_texts: List[str]
    article_expand_dieu_ids: List[str]
