from dotenv import load_dotenv
load_dotenv()
import os
import json
import asyncio
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.prompts import load_mcp_prompt
# from langchain_core.runnables import RunnableConfig
# from mcp import ClientSession, StdioServerParameters
# from mcp.client.stdio import stdio_client
# from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt import ToolNode, tools_condition
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='my-agent-app.log', level=logging.DEBUG)

# from IPython.display import Image


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


class Agent:
    """Alternative Agent for Claude code"""

    def __init__(self, mcp_client, tools, prompt) -> None:
        # MCP Client
        self.mcp_client = mcp_client

        self.mcp_tools = tools
        self.mcp_prompts = [].append(prompt)
        # LLMs model section
        self.llm_model = ChatOllama(
            model="qwen3:8b",
            base_url="http://127.0.0.1:11434",
            temperature=0
        ).bind_tools(tools=self.mcp_tools)

        self.plan_llm_model = ChatOllama(
            model="qwen2.5-coder:3b",
            base_url="http://127.0.0.1:11434",
            temperature=0
        )

        # Graph section
        graph = StateGraph(AgentState)
        graph.add_node("llm", self._call_model)
        # graph.add_node("tool_call_llm", self._call_tools_model)
        graph.add_node("tools", self._tool_node)
        # graph.add_conditional_edges(
        #     "llm",
        #     self._exists_action,
        #     {True: "action", False: END}

        # )
        graph.add_conditional_edges(
            "llm",
            tools_condition
        )
        
        graph.add_edge("tools", "llm")
        # graph.add_edge("llm", "tool_call_llm")
        # graph.add_edge("tool_call_llm", "action")
        # graph.add_edge("action", "llm")
        graph.set_entry_point("llm")

        self.graph = graph.compile()
    
    @classmethod
    async def create(cls) -> "Agent":
        client = MultiServerMCPClient({
            "code_tools": {
                "command": "python3",
                "args": [os.path.join(
                    "/Users/jrvn-hieu/Desktop/Practical/tutorial_mcp_practice/",
                    "mcp_app/mcp-course/projects/unit3/build-mcp-server/starter",
                    "server.py")],
                "env": {
                    "SLACK_WEBHOOK_URL": os.getenv("SLACK_WEBHOOK_URL", "")
                },
                "transport": "stdio",
            }
        })

        # server_params = StdioServerParameters(
        #     command="python3",
        #     args=[os.path.join(
        #         "/Users/jrvn-hieu/Desktop/Practical/tutorial_mcp_practice/",
        #         "mcp_app/mcp-course/projects/unit3/build-mcp-server/starter",
        #         "server.py")],
        #     env=None,
        # )

        prompts = []

        async with client.session("code_tools") as session:
            tools = await load_mcp_tools(session)
            response = await session.list_prompts()
            for prompt in response.prompts:
                prompts.append(
                    await load_mcp_prompt(
                        session=session,
                        name=prompt.name
                    )
                )
            
            logger.info("%s", prompts)

        # async with stdio_client(server_params) as (read, write):
        #     async with ClientSession(read, write) as session:
        #         await session.initialize()

        #         # List available prompts
        #         response = await session.list_prompts()
        #         logger.info("\nList of mcp prompts:")
        #         for prompt in response.prompts:
        #             mcp_prompt = await client.get_prompt(
        #                 server_name="code_tools",
        #                 prompt_name=prompt.name
        #             )
        #             prompts.append(mcp_prompt)
                
        #         logger.info("%s", prompts)

        # tools = await client.get_tools()

        return cls(client, tools, prompts)

    def _call_model(
        self,
        state: AgentState
    ):
        system = SystemMessage(
            "You are a helpful assistant. "
            "You are allowed to make multiple calls in sequence. \n"
            "These are tools you are allowed to call if need:\n"
            "Tools:\n"
            f"{'\n'.join(self._extract_tool_names().keys())}"
            # "the change_summary argument for suggest_template should be summarized as format could provide value for the template result from get_pr_templates"
            # "If you have enough information to compose the final response. Stop and return the final response."
        )

        logger.info(f"----------mcp tools: {len(self.mcp_tools)}")

        # response = self.llm_model.bind_tools(tools=self.mcp_tools).invoke(
        #     [system] + state["messages"]
        # )
        response = self.llm_model.invoke(
            [system] + state["messages"]
        )
        return {"messages": [response]}  # list → reducer appends
    
    # def _call_tools_model(
    #     self,
    #     state: AgentState
    # ):
    #     system = SystemMessage(
    #         "You are tools call assitant. You decide which tools to call."
    #         "Tools like below instruction to get the results for next step:\n"
    #         "Use analyze_file_changes to get the full diff and list of changed files in the current git repository\n"
    #         "Use get_pr_templates to List available PR templates with their content.\n"
    #         "Use suggest_template to analyze the changes and suggest the most appropriate PR template\n"
    #         "Decide which tools should be called before answer for the final response"
    #     )

    #     response = self.llm_model.invoke(
    #         [system] + state["messages"]
    #     )
    #     return {"messages": [response]}  # list → reducer appends

    def _exists_action(self, state: AgentState):
        # should be return with boolean on conditional edge
        logger.info(f"Run {self._exists_action.__name__}")
        result = state['messages'][-1]
        logger.info(f"Any tool_calls {len(result.tool_calls)}")
        # check if any tool_calls, True if any tool_calls
        return len(result.tool_calls) > 0

    def _extract_tool_names(self):
        return {t.name: t for t in self.mcp_tools}

    async def _tool_node(self, state: AgentState):
        """Tool Node"""
        logger.info("-------Tool Node")
        tools_by_name = self._extract_tool_names()
        logger.info(f"----Tools: {tools_by_name}")
        tool_calls = state["messages"][-1].tool_calls
        logger.info(f"----Tool call: {tool_calls}")
        outputs = []

        for call in tool_calls:
            logger.info(f"loop through tool: {call["name"]}")
            if not call["name"] in tools_by_name:
                logger.info("\n ---------bad tool name ")
                result = "bad tool name, retry?"
            else:
                result = await tools_by_name[call["name"]].ainvoke(call["args"])
            
            outputs.append(
                ToolMessage(
                    tool_call_id=call["id"],
                    name=call["name"],
                    content=json.dumps(result),
                )
            )
        return {"messages": outputs}


if __name__ == "__main__":
    # Test agent with mcp to see whether it could replace Claude code
    # code_agent = await Agent.create()  # must run inside async function
    code_agent = asyncio.run(Agent.create())  # for running outside

    # code_agent.graph.get_graph().draw_mermaid_png(
    #     output_file_path="graph_2.png"
    # )
    # code_agent.graph.get_graph().print_ascii()

    # query = "Can you analyze my changes and suggest a PR template?"
    query="Get recent GitHub Actions events received via webhook"
    messages = [HumanMessage(content=query)]
    # result = code_agent.graph.invoke({"messages": messages})
    # print("=========The result: \n")
    # print(result)

    async def chat():
        async for step in code_agent.graph.astream({"messages": messages}, stream_mode="values"):
            step["messages"][-1].pretty_print()
            print("\n")

    # asyncio.run(chat())
