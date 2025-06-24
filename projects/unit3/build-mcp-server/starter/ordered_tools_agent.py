import os
import json
import asyncio
from typing import TypedDict, Annotated, Dict, Any, Optional
import operator
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.runnables import RunnableConfig
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('my-agent-app.log'),
        logging.StreamHandler()
    ]
)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    file_changes: Optional[str]
    pr_templates: Optional[str]
    suggested_template: Optional[str]
    current_step: str


class Agent:
    """Agent with ordered tool execution"""

    def __init__(self, mcp_client, tools) -> None:
        self.mcp_client = mcp_client
        self.mcp_tools = tools
        
        # LLM model
        self.llm_model = ChatOllama(
            model="qwen3:8b",
            base_url="http://127.0.0.1:11434",
            temperature=0
        )

        # Build graph with ordered execution
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build graph with ordered tool execution"""
        graph = StateGraph(AgentState)
        
        # Add nodes for each step
        graph.add_node("start", self._start_node)
        graph.add_node("analyze_changes", self._analyze_changes_node)
        graph.add_node("get_templates", self._get_templates_node)
        graph.add_node("suggest_template", self._suggest_template_node)
        graph.add_node("final_response", self._final_response_node)
        
        # Define the execution order
        graph.add_edge("start", "analyze_changes")
        graph.add_edge("analyze_changes", "get_templates")
        graph.add_edge("get_templates", "suggest_template")
        graph.add_edge("suggest_template", "final_response")
        graph.add_edge("final_response", END)
        
        graph.set_entry_point("start")
        
        return graph.compile()
    
    @classmethod
    async def create(cls) -> "Agent":
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
        
        tools = await client.get_tools()
        return cls(client, tools)

    def _get_tool_by_name(self, name: str):
        """Get tool by name"""
        for tool in self.mcp_tools:
            if tool.name == name:
                return tool
        return None

    def _start_node(self, state: AgentState) -> Dict[str, Any]:
        """Initialize the process"""
        logger.info("Starting ordered tool execution")
        return {
            "current_step": "analyze_changes",
            "file_changes": None,
            "pr_templates": None,
            "suggested_template": None
        }

    async def _analyze_changes_node(self, state: AgentState) -> Dict[str, Any]:
        """Step 1: Analyze file changes"""
        logger.info("Step 1: Analyzing file changes")
        
        analyze_tool = self._get_tool_by_name("analyze_file_changes")
        if not analyze_tool:
            logger.error("analyze_file_changes tool not found")
            return {
                "file_changes": "Error: analyze_file_changes tool not available",
                "current_step": "get_templates"
            }
        
        try:
            # Call the tool
            result = await analyze_tool.ainvoke({})
            file_changes = json.dumps(result) if isinstance(result, dict) else str(result)
            
            logger.info(f"File changes analyzed: {len(file_changes)} characters")
            
            # Add a message about this step
            step_message = AIMessage(
                content=f"✅ Step 1 completed: Analyzed file changes\n"
                       f"Found changes in repository with {len(str(result))} characters of diff data."
            )
            
            return {
                "messages": [step_message],
                "file_changes": file_changes,
                "current_step": "get_templates"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing file changes: {e}")
            error_message = AIMessage(content=f"❌ Error in Step 1: {str(e)}")
            return {
                "messages": [error_message],
                "file_changes": f"Error: {str(e)}",
                "current_step": "get_templates"
            }

    async def _get_templates_node(self, state: AgentState) -> Dict[str, Any]:
        """Step 2: Get PR templates"""
        logger.info("Step 2: Getting PR templates")
        
        get_templates_tool = self._get_tool_by_name("get_pr_templates")
        if not get_templates_tool:
            logger.error("get_pr_templates tool not found")
            return {
                "pr_templates": "Error: get_pr_templates tool not available",
                "current_step": "suggest_template"
            }
        
        try:
            # Call the tool
            result = await get_templates_tool.ainvoke({})
            pr_templates = json.dumps(result) if isinstance(result, dict) else str(result)
            logger.info(f"------PR templates type true -> dict or false -> str {isinstance(result, dict)}")
            
            logger.info(f"PR templates retrieved: {len(pr_templates)} characters")
            
            # Add a message about this step
            step_message = AIMessage(
                content=f"✅ Step 2 completed: Retrieved PR templates\n"
                       f"Found {len(str(result))} characters of template data."
            )
            
            return {
                "messages": [step_message],
                "pr_templates": pr_templates,
                "current_step": "suggest_template"
            }
            
        except Exception as e:
            logger.error(f"Error getting PR templates: {e}")
            error_message = AIMessage(content=f"❌ Error in Step 2: {str(e)}")
            return {
                "messages": [error_message],
                "pr_templates": f"Error: {str(e)}",
                "current_step": "suggest_template"
            }

    async def _suggest_template_node(self, state: AgentState) -> Dict[str, Any]:
        """Step 3: Suggest template based on changes"""
        logger.info("Step 3: Suggesting template")
        
        suggest_tool = self._get_tool_by_name("suggest_template")
        if not suggest_tool:
            logger.error("suggest_template tool not found")
            return {
                "suggested_template": "Error: suggest_template tool not available",
                "current_step": "final_response"
            }
        
        try:
            # Prepare a summary of changes for the suggest_template tool
            file_changes = state.get("file_changes", "")
            template_type = json.loads(state.get("pr_templates"))[-1].get("type", "feature")
            
            # Create a concise summary from the file changes
            change_summary = self._create_change_summary(file_changes)
            
            # Call the suggest_template tool with the summary
            result = await suggest_tool.ainvoke({
                "changes_summary": change_summary,
                "change_type": template_type
            })
            
            suggested_template = json.dumps(result) if isinstance(result, dict) else str(result)
            
            logger.info(f"Template suggested: {len(suggested_template)} characters")
            
            # Add a message about this step
            step_message = AIMessage(
                content=f"✅ Step 3 completed: Template suggestion generated\n"
                       f"Based on the analysis, suggested appropriate PR template."
            )
            
            return {
                "messages": [step_message],
                "suggested_template": suggested_template,
                "current_step": "final_response"
            }
            
        except Exception as e:
            logger.error(f"Error suggesting template: {e}")
            error_message = AIMessage(content=f"❌ Error in Step 3: {str(e)}")
            return {
                "messages": [error_message],
                "suggested_template": f"Error: {str(e)}",
                "current_step": "final_response"
            }

    def _create_change_summary(self, file_changes: str) -> str:
        """Create a concise summary of file changes for template suggestion"""
        try:
            # Try to parse as JSON first
            if file_changes.startswith('{') or file_changes.startswith('['):
                changes_data = json.loads(file_changes)
                
                # Extract key information
                summary_parts = []
                
                if isinstance(changes_data, dict):
                    if 'files' in changes_data:
                        files = changes_data['files']
                        summary_parts.append(f"Modified {len(files)} files")
                        
                        # Categorize file types
                        file_types = {}
                        for file_path in files:
                            ext = os.path.splitext(file_path)[1]
                            file_types[ext] = file_types.get(ext, 0) + 1
                        
                        if file_types:
                            type_summary = ", ".join([f"{count} {ext or 'no-ext'} files" 
                                                    for ext, count in file_types.items()])
                            summary_parts.append(f"File types: {type_summary}")
                    
                    if 'diff' in changes_data:
                        diff = changes_data['diff']
                        if diff:
                            lines_added = diff.count('+')
                            lines_removed = diff.count('-')
                            summary_parts.append(f"~{lines_added} additions, ~{lines_removed} deletions")
                
                summary = ". ".join(summary_parts)
                
            else:
                # Fallback: create summary from raw text
                lines = file_changes.split('\n')
                summary = f"File changes detected with {len(lines)} lines of diff data"
            
            # Limit summary length
            if len(summary) > 500:
                summary = summary[:497] + "..."
                
            return summary
            
        except Exception as e:
            logger.warning(f"Could not parse file changes, using raw summary: {e}")
            # Fallback summary
            return f"File changes detected ({len(file_changes)} characters of data)"

    def _final_response_node(self, state: AgentState) -> Dict[str, Any]:
        """Step 4: Generate final response"""
        logger.info("Step 4: Generating final response")
        
        # Compile all the results
        file_changes = state.get("file_changes", "No changes analyzed")
        pr_templates = state.get("pr_templates", "No templates found")
        suggested_template = state.get("suggested_template", "No template suggested")
        
        # Create a comprehensive final response
        final_response = AIMessage(
            content=f"""
🎯 **PR Template Analysis Complete**

## Summary
I've analyzed your repository changes and provided a PR template suggestion based on the findings.

## Process Completed:
1. ✅ **File Changes Analysis**: Analyzed current repository changes
2. ✅ **Template Retrieval**: Retrieved available PR templates  
3. ✅ **Template Suggestion**: Matched changes to appropriate template

## Results:
**Suggested Template:**
{suggested_template}

---
*This analysis was performed by examining your current git changes and matching them against available PR templates.*
            """.strip()
        )
        
        return {
            "messages": [final_response],
            "current_step": "completed"
        }

    async def chat(self, query: str) -> None:
        """Chat with the agent"""
        messages = [HumanMessage(content=query)]
        
        try:
            async for step in self.graph.astream(
                {
                    "messages": messages,
                    "current_step": "start",
                    "file_changes": None,
                    "pr_templates": None,
                    "suggested_template": None
                }, 
                stream_mode="values"
            ):
                if step.get("messages"):
                    step["messages"][-1].pretty_print()
                    print("\n" + "="*50 + "\n")
        except Exception as e:
            logger.error(f"Error during chat: {e}")
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    async def main():
        try:
            # Create agent
            code_agent = await Agent.create()
            
            # Test query
            query = "Can you analyze my changes and suggest a PR template?"
            await code_agent.chat(query)
            
        except Exception as e:
            logger.error(f"Error in main: {e}")
            print(f"Failed to run agent: {e}")

    asyncio.run(main())
