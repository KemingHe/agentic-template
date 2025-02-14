from typing import Iterator

from langchain_core.output_parsers import StrOutputParser

from setup_openai import openai_lite_model
from setup_prompt import simple_prompt

simple_chain = simple_prompt | openai_lite_model | StrOutputParser()


def get_simple_response_stream(
    user_query: str,
    chat_history: str,
) -> Iterator[str]:
    return simple_chain.stream(
        {
            "chat_history": chat_history,
            "user_query": user_query,
        }
    )
