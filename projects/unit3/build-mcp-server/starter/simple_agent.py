import os
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

# from langchain.chat_models import init_chat_model
from langchain_ollama import ChatOllama
import asyncio
# model = init_chat_model("openai:gpt-4.1")
model = ChatOllama(
    model="llama3.1:8b",
    base_url="http://127.0.0.1:11434",
    temperature=0
)

client = MultiServerMCPClient({
    "code_tools": {
        "command": "python3",
        "args": [os.path.join(
            "/Users/jrvn-hieu/Desktop/Practical/tutorial_mcp_practice/",
            "mcp_app/mcp-course/projects/unit3/build-mcp-server/starter",
            "server.py")],
        "transport": "stdio",
    }
})


tools = asyncio.run(client.get_tools())

def call_model(state: MessagesState):
    response = model.bind_tools(tools).invoke(state["messages"])
    return {"messages": response}

builder = StateGraph(MessagesState)
builder.add_node(call_model)
builder.add_node(ToolNode(tools))
builder.add_edge(START, "call_model")
builder.add_conditional_edges(
    "call_model",
    tools_condition,
)
builder.add_edge("tools", "call_model")
graph = builder.compile()
analyze_response = asyncio.run(graph.ainvoke({"messages": "Can you analyze my changes and suggest a PR template?"}))
# weather_response = await graph.ainvoke({"messages": "what is the weather in nyc?"})
print(analyze_response)
