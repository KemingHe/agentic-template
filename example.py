# from setup_openai import openai_lite_model
# from setup_pinecone import get_relevant_docs
from duckduckgo_search import DDGS
import json

# openai_lite_model.invoke("what's a hephalump?")
# get_relevant_docs(query="what's autonomous driving?", k=1)
search_results = DDGS().text(
    keywords="what's a hephalump?",
    safesearch="strict",
    max_results=3,
)
formatted_results = json.dumps(search_results, indent=2)
print(formatted_results)  # For verification
