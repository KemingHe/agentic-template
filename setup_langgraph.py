from typing import Literal

from langchain_core.tools import tool
from langchain_core.messages import BaseMessage

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph, MessagesState

from langgraph.prebuilt import ToolNode

from setup_openai import openai_lite_model
from setup_duckduckgo import simple_search


# Define the tools availble to the agents
@tool
def search(query: str) -> str:
    """Search the web for information"""
    return simple_search(query)


tools = [search]
tool_node = ToolNode(tools)
model = openai_lite_model.bind_tools(tools)


# Define the function to determine whether to continue or not
def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]

    # Route to the "tools" node if LLM makes a tool call, else stop and reply to user
    if last_message.tool_calls:
        return "tools"
    return END


# Define the funciton that calls the model
def call_model(state: MessagesState):
    messages = state["messages"]
    response: BaseMessage = model.invoke(messages)

    # Return a list to add to the existing list of messages
    return {"messages": response}


# Define the agentic graph and add the agent and tool nodes
workflow = StateGraph(MessagesState)
workflow.add_node("first_agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("final_agent", call_model)

# Set the first agent node as the entrypoint
workflow.add_edge(START, "first_agent")

# Add conditional edges from first agent: either to tools or end
workflow.add_conditional_edges(
    "first_agent",
    lambda state: "tools" if state["messages"][-1].tool_calls else END
)

# After tools, always go to final agent for response
workflow.add_edge("tools", "final_agent")

# Final agent always ends
workflow.add_edge("final_agent", END)

# Persist memory between graph runs
checkpointer = MemorySaver()

# Compile the worlfow into a LangChain runnable, passing in the checkpointer (memory)
app = workflow.compile(checkpointer=checkpointer)

# Run the agentic app
final_state = app.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What's the weather in SF?",
            }
        ]
    },
    config={"configurable": {"thread_id": 42}},
)

print(final_state["messages"][-1].content)
