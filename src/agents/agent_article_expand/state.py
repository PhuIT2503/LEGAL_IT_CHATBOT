from typing import List, TypedDict


class ArticleExpandState(TypedDict):
    article_expand_dieu_ids: List[str]
    graph_context: str
    graph_fetched_dieu_ids: List[str]
