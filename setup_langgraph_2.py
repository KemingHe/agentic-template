from typing import Literal
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from setup_openai import openai_lite_model
from setup_duckduckgo import simple_search

# Define tools
@tool
def web_search(query: str) -> str:
    """Search the web for information"""
    return simple_search(query)

@tool
def rag_search(query: str) -> str:
    """Search internal knowledge base"""
    return f"Simulated RAG response for: {query}"

tools = {"web": [web_search], "rag": [rag_search]}

# Define agent nodes
def orchestrator_agent(state: MessagesState):
    messages = state["messages"]
    response = openai_lite_model.invoke(messages)
    return {"messages": [response], "next": "web_agent"}

def web_agent(state: MessagesState):
    web_tool_node = ToolNode(tools["web"])
    result = web_tool_node.invoke(state)
    return {"messages": result["messages"], "next": "summary_agent"}

def rag_agent(state: MessagesState):
    rag_tool_node = ToolNode(tools["rag"])
    result = rag_tool_node.invoke(state)
    return {"messages": result["messages"], "next": "summary_agent"}

def summary_agent(state: MessagesState):
    messages = state["messages"]
    response = openai_lite_model.invoke(messages)
    return {"messages": [response]}

# Build the graph
workflow = StateGraph(MessagesState)

# Add nodes
workflow.add_node("orchestrator", orchestrator_agent)
workflow.add_node("web_agent", web_agent)
workflow.add_node("rag_agent", rag_agent)
workflow.add_node("summary_agent", summary_agent)

# Add edges
workflow.add_edge(START, "orchestrator")
workflow.add_conditional_edges(
    "orchestrator",
    lambda state: state.get("next", END)
)
workflow.add_edge("web_agent", "summary_agent")
workflow.add_edge("rag_agent", "summary_agent")
workflow.add_edge("summary_agent", END)

# Compile with memory persistence
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    final_state = app.invoke({
        "messages": [{
            "role": "user",
            "content": "What are the latest news about AI?"
        }]
    })
    print(final_state["messages"][-1].content)