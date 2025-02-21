from time import perf_counter
from typing import Iterator, Callable

import pytest
from langchain_core.runnables import Runnable

# from llms.groq import groq_llama_regular_model
from llms.openai import openai_regular_model
from chains.web_search import get_single_web_search_chain_response_stream
# from agents.web_search_agent import get_web_search_agent_response_stream


class StreamLatencyTracker:
    def __init__(self):
        self.ttfb: float = 0
        self.total_time: float = 0
        self.bytes_received: int = 0

    def track_stream(self, stream: Iterator[str]) -> Iterator[str]:
        start_time: float = perf_counter()
        is_first_chunk: bool = True

        for chunk in stream:
            if is_first_chunk:
                self.ttfb = perf_counter() - start_time
                is_first_chunk = False

            self.bytes_received += len(chunk)
            yield chunk

        self.total_time = perf_counter() - start_time


def create_stream_runner(llm: Runnable) -> Callable:
    def run_stream() -> float:
        tracker = StreamLatencyTracker()
        query: str = "What is the weather in Beijing?"
        chat_history: str = ""

        stream: Iterator[str] = get_single_web_search_chain_response_stream(
            llm=llm, user_query=query, chat_history=chat_history, track_metrics=False
        )
        # Consume first chunk to measure TTFB
        list(tracker.track_stream(stream))[0]
        return tracker.ttfb

    return run_stream


# Groq benchmark is inconsistent, suspect rate limiting, disabled for now
# @pytest.mark.benchmark(
#     min_rounds=5,
#     warmup=False,
#     disable_gc=True,
#     group="streaming-groq",
# )
# def test_groq_web_search_chain_stream_latency(benchmark):
#     benchmark(create_stream_runner(groq_llama_regular_model))


@pytest.mark.benchmark(
    min_rounds=5,
    warmup=False,
    disable_gc=True,
    group="streaming-openai",
)
def test_openai_web_search_chain_stream_latency(benchmark):
    benchmark(create_stream_runner(openai_regular_model))
