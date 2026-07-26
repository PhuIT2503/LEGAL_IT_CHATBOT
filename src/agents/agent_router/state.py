import operator
from typing import Annotated, Callable, Sequence, TypedDict

from langchain_core.messages import BaseMessage


class RouterState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query: str
    is_chit_chat: bool
    final_response: str
    progress_callback: Callable[[str], None]
