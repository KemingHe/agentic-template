from duckduckgo_search import DDGS
import json


def simple_search(
    query: str,
    count: int = 3,
) -> str:
    results: list[dict[str, str]] = DDGS().text(
        keywords=query,
        safesearch="strict",
        max_results=count,
    )
    formatted_results = json.dumps(results, indent=2)
    return formatted_results
