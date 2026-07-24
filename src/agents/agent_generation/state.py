import operator
from typing import Annotated, List, Sequence, TypedDict

from langchain_core.messages import BaseMessage


class GenerationState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query: str
    context_texts: List[str]
    graph_context: str
    draft_response: str
    final_response: str
