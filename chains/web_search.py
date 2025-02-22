from typing import Iterator
from time import perf_counter
from pprint import pprint

from langchain_core.output_parsers import StrOutputParser

from tools.web_search import ddg_text_search
from tools.weather import get_weather_data
from tasks.search import SingleSearchTask, MultiSearchTask
from prompts.single_task import (
    single_task_orchestrator_prompt,
    single_task_orchestrator_parser,
)
from prompts.multi_task import (
    multi_task_orchestrator_prompt,
    multi_task_orchestrator_parser,
)
from prompts.summarizer import summarizer_prompt, simple_summarizer_prompt
from chains.types import ChainConfig, ChainInputs


def get_multi_web_search_chain_response_stream(
    config: ChainConfig,
    inputs: ChainInputs,
) -> Iterator[str]:
    orchestrator_llm = config.orchestrator_llm
    summarizer_llm = config.summarizer_llm
    track_metrics = config.track_metrics
    user_query = inputs.user_query
    chat_history = inputs.chat_history

    start_time: float | None = perf_counter() if track_metrics else None

    orchestrator_chain = (
        multi_task_orchestrator_prompt
        | orchestrator_llm
        | multi_task_orchestrator_parser
    )
    search_task: MultiSearchTask = orchestrator_chain.invoke(
        {
            "user_query": user_query,
            "chat_history": chat_history,
        }
    )

    pprint(search_task)

    web_search_results: list[str] = []
    if search_task.should_search_web:
        for web_task in search_task.web_tasks:
            web_search_results.append(
                ddg_text_search(
                    query=web_task.query,
                    query_count=web_task.query_count,
                )
            )

    pprint(web_search_results)

    weather_search_results: list[str] = []
    if search_task.should_search_weather:
        for weather_task in search_task.weather_tasks:
            weather_search_results.append(
                get_weather_data(
                    location=weather_task.location,
                )
            )

    pprint(weather_search_results)

    summarizer_chain = summarizer_prompt | summarizer_llm | StrOutputParser()
    result_stream: Iterator[str] = summarizer_chain.stream(
        {
            "user_query": user_query,
            "chat_history": chat_history,
            "web_search_results": web_search_results,
            "weather_search_results": weather_search_results,
        }
    )

    for chunk in result_stream:
        if track_metrics and start_time:
            ttfb: float = perf_counter() - start_time
            # logging.info(f"Time to first byte: {ttfb:.4f}s")
            pprint(f"Web Search Chain - time to first byte: {ttfb:.4f}s")
            start_time = None  # Only log TTFB once
        yield chunk


def get_single_web_search_chain_response_stream(
    config: ChainConfig,
    inputs: ChainInputs,
) -> Iterator[str]:
    orchestrator_llm = config.orchestrator_llm
    summarizer_llm = config.summarizer_llm
    track_metrics = config.track_metrics
    user_query = inputs.user_query
    chat_history = inputs.chat_history

    start_time: float | None = perf_counter() if track_metrics else None

    orchestrator_chain = (
        single_task_orchestrator_prompt
        | orchestrator_llm
        | single_task_orchestrator_parser
    )
    search_task: SingleSearchTask = orchestrator_chain.invoke(
        {
            "user_query": user_query,
            "chat_history": chat_history,
        }
    )

    pprint(search_task)

    web_search_results: str | None = None
    if search_task.should_search_web:
        web_search_results = ddg_text_search(
            query=search_task.web_query,
            query_count=search_task.web_query_count,
        )

    pprint(web_search_results)

    weather_search_results: str | None = None
    if search_task.should_search_weather:
        weather_search_results = get_weather_data(
            location=search_task.weather_query,
        )

    pprint(weather_search_results)

    summarizer_chain = summarizer_prompt | summarizer_llm | StrOutputParser()
    result_stream: Iterator[str] = summarizer_chain.stream(
        {
            "user_query": user_query,
            "chat_history": chat_history,
            "web_search_results": web_search_results,
            "weather_search_results": weather_search_results,
        }
    )

    for chunk in result_stream:
        if track_metrics and start_time:
            ttfb: float = perf_counter() - start_time
            # logging.info(f"Time to first byte: {ttfb:.4f}s")
            pprint(f"Web Search Chain - time to first byte: {ttfb:.4f}s")
            start_time = None  # Only log TTFB once
        yield chunk


def get_simple_chain_response_stream(
    config: ChainConfig,
    inputs: ChainInputs,
) -> Iterator[str]:
    summarizer_llm = config.summarizer_llm
    track_metrics = config.track_metrics
    user_query = inputs.user_query
    chat_history = inputs.chat_history

    start_time: float | None = perf_counter() if track_metrics else None

    simple_summarizer_chain = (
        simple_summarizer_prompt | summarizer_llm | StrOutputParser()
    )
    result_stream: Iterator[str] = simple_summarizer_chain.stream(
        {
            "user_query": user_query,
            "chat_history": chat_history,
        }
    )

    for chunk in result_stream:
        if track_metrics and start_time:
            ttfb: float = perf_counter() - start_time
            # logging.info(f"Time to first byte: {ttfb:.4f}s")
            pprint(f"Simple Chain - time to first byte: {ttfb:.4f}s")
            start_time = None  # Only log TTFB once
        yield chunk
