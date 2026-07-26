import operator
from typing import Annotated, Any, Callable, Dict, List, Sequence, TypedDict

from langchain_core.messages import BaseMessage


class GenerationState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query: str
    context_texts: List[str]
    context_records: List[Dict[str, Any]]
    behavior_profile: Dict[str, List[str]]
    graph_context: str
    draft_response: str
    final_response: str
    retrieval_is_complete: bool
    retrieval_is_relevant: bool
    retrieval_decisions: List[Dict[str, Any]]
    critic_report: Dict[str, Any]
    progress_callback: Callable[[str], None]
