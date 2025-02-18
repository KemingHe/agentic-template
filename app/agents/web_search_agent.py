from typing import Iterator

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import StructuredTool
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from llms.openai import openai_regular_model
from tools.weather import get_weather_data
from tools.web_search import ddg_text_search
from chains.web_search_chain import summarizer_chain

tools = [
    StructuredTool.from_function(
        func=ddg_text_search,
        name="web_search",
        description="Search the web for current information",
    ),
    StructuredTool.from_function(
        func=get_weather_data,
        name="weather_search",
        description="Get weather information for a location",
    ),
]

# Add this before the agent creation
agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a knowledgeable research assistant that helps users find information through web searches and weather data. Follow these guidelines:

1. TOOL SELECTION:
- Use 'web_search' for factual queries, current events, or specific information needs
- Use 'weather_search' for any weather-related queries about specific locations
- You can use multiple tools if the query requires both types of information

2. SEARCH STRATEGY:
- Create precise, focused search queries using key terms
- Request 1-5 web results based on complexity of the query
- For weather queries, provide specific location names with state/country when possible

3. ANALYSIS:
- Synthesize search results into coherent responses
- Cite web sources when presenting information
- Present weather data in a clear, natural way
- Maintain context from previous interactions

4. RESPONSE FORMAT:
- Start with direct answers
- Include relevant supporting details
- Use bullet points for multiple pieces of information
- Keep responses concise but informative

Remember: Only use tools when necessary. For general conversation or questions not requiring current information, respond directly without tool use.""",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_openai_functions_agent(
    llm=openai_regular_model, tools=tools, prompt=agent_prompt
)

# Use AgentExecutor for dynamic tool routing
agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)

# Create final chain with LCEL
final_chain = (
    RunnablePassthrough.assign(agent_result=agent_executor)
    | {
        "user_query": lambda x: x["input"],
        "chat_history": lambda x: x["chat_history"],
        "web_search_results": lambda x: x["agent_result"],
        "weather_search_results": lambda x: x[
            "agent_result"
        ],  # backwards-compatible hack, remove in future
    }
    | summarizer_chain
).with_config({"run_mode": "stream"})


def get_agent_response_stream(
    user_query: str,
    chat_history: str,
) -> Iterator[str]:
    """
    Stream responses from the agent using LCEL (LangChain Expression Language).
    """
    return final_chain.stream({"input": user_query, "chat_history": chat_history})
