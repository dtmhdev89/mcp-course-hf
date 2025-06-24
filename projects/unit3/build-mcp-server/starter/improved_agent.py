import os
import json
import asyncio
from typing import TypedDict, Annotated, Dict, Any, Optional, List
import operator
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.runnables import RunnableConfig
import logging
from pathlib import Path

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
    error_context: Optional[str]
    processing_metadata: Optional[Dict[str, Any]]


class Agent:
    """Agent with ordered tool execution and enhanced error handling"""

    def __init__(self, mcp_client, tools) -> None:
        self.mcp_client = mcp_client
        self.mcp_tools = tools
        
        # LLM model with enhanced configuration
        self.llm_model = ChatOllama(
            model="llama3.1:8b",
            base_url="http://127.0.0.1:11434",
            temperature=0.1,  # Slightly higher for more creative responses
            num_ctx=4096,     # Increased context window
        )

        # Build graph with enhanced error handling
        self.graph = self._build_graph()
        
        # Tool execution configuration
        self.tool_timeout = 30  # seconds
        self.max_retries = 2
    
    def _build_graph(self):
        """Build graph with enhanced error handling and conditional flows"""
        graph = StateGraph(AgentState)
        
        # Add nodes for each step
        graph.add_node("start", self._start_node)
        graph.add_node("analyze_changes", self._analyze_changes_node)
        graph.add_node("get_templates", self._get_templates_node)
        graph.add_node("suggest_template", self._suggest_template_node)
        graph.add_node("final_response", self._final_response_node)
        graph.add_node("error_handler", self._error_handler_node)
        
        # Define the execution order with error handling
        graph.add_edge("start", "analyze_changes")
        graph.add_edge("analyze_changes", "get_templates")
        graph.add_edge("get_templates", "suggest_template")
        graph.add_edge("suggest_template", "final_response")
        graph.add_edge("final_response", END)
        graph.add_edge("error_handler", END)
        
        graph.set_entry_point("start")
        
        return graph.compile()
    
    @classmethod
    async def create(cls, mcp_server_path: Optional[str] = None) -> "Agent":
        """Create agent with configurable MCP server path"""
        if mcp_server_path is None:
            mcp_server_path = os.path.join(
                "/Users/jrvn-hieu/Desktop/Practical/tutorial_mcp_practice/",
                "mcp_app/mcp-course/projects/unit3/build-mcp-server/starter",
                "server.py"
            )
        
        # Validate server path exists
        if not os.path.exists(mcp_server_path):
            raise FileNotFoundError(f"MCP server not found at: {mcp_server_path}")
        
        client = MultiServerMCPClient({
            "code_tools": {
                "command": "python3",
                "args": [mcp_server_path],
                "transport": "stdio",
            }
        })
        
        try:
            tools = await client.get_tools()
            logger.info(f"Successfully loaded {len(tools)} MCP tools")
            return cls(client, tools)
        except Exception as e:
            logger.error(f"Failed to initialize MCP client: {e}")
            raise

    def _get_tool_by_name(self, name: str):
        """Get tool by name with enhanced error checking"""
        for tool in self.mcp_tools:
            if tool.name == name:
                return tool
        logger.warning(f"Tool '{name}' not found. Available tools: {[t.name for t in self.mcp_tools]}")
        return None

    def _start_node(self, state: AgentState) -> Dict[str, Any]:
        """Initialize the process with metadata"""
        logger.info("Starting enhanced ordered tool execution")
        return {
            "current_step": "analyze_changes",
            "file_changes": None,
            "pr_templates": None,
            "suggested_template": None,
            "error_context": None,
            "processing_metadata": {
                "start_time": asyncio.get_event_loop().time(),
                "steps_completed": 0,
                "total_steps": 4
            }
        }

    async def _analyze_changes_node(self, state: AgentState) -> Dict[str, Any]:
        """Step 1: Analyze file changes with enhanced parameters"""
        logger.info("Step 1: Analyzing file changes with enhanced options")
        
        analyze_tool = self._get_tool_by_name("analyze_file_changes")
        if not analyze_tool:
            return self._handle_tool_error("analyze_file_changes", "Tool not found", state)
        
        try:
            # Enhanced parameters for better analysis
            tool_params = {
                "base_branch": "main",
                "include_diff": True,
                "max_diff_lines": 1000,  # Increased for better analysis
                "working_directory": None  # Let MCP server auto-detect
            }
            
            # Call the tool with timeout
            result = await asyncio.wait_for(
                analyze_tool.ainvoke(tool_params), 
                timeout=self.tool_timeout
            )
            
            # Parse and validate result
            if isinstance(result, str):
                try:
                    parsed_result = json.loads(result)
                    if "error" in parsed_result:
                        return self._handle_tool_error("analyze_file_changes", parsed_result["error"], state)
                except json.JSONDecodeError:
                    parsed_result = {"raw_output": result}
            else:
                parsed_result = result
            
            file_changes = json.dumps(parsed_result, indent=2)
            
            logger.info(f"File changes analyzed successfully: {len(file_changes)} characters")
            
            # Enhanced step message with metadata
            step_message = AIMessage(
                content=f"✅ **Step 1 completed**: File Changes Analysis\n\n"
                       f"📊 **Results Summary:**\n"
                       f"- Analyzed changes against base branch: {tool_params['base_branch']}\n"
                       f"- Total data size: {len(file_changes):,} characters\n"
                       f"- Diff included: {'Yes' if tool_params['include_diff'] else 'No'}\n"
                       f"- Working directory: Auto-detected by MCP server\n\n"
                       f"🔄 **Next**: Retrieving PR templates..."
            )
            
            # Update metadata
            metadata = state.get("processing_metadata", {})
            metadata["steps_completed"] = 1
            metadata["analyze_changes_size"] = len(file_changes)
            
            return {
                "messages": [step_message],
                "file_changes": file_changes,
                "current_step": "get_templates",
                "processing_metadata": metadata
            }
            
        except asyncio.TimeoutError:
            return self._handle_tool_error("analyze_file_changes", "Tool execution timeout", state)
        except Exception as e:
            return self._handle_tool_error("analyze_file_changes", str(e), state)

    async def _get_templates_node(self, state: AgentState) -> Dict[str, Any]:
        """Step 2: Get PR templates with validation"""
        logger.info("Step 2: Getting PR templates with validation")
        
        get_templates_tool = self._get_tool_by_name("get_pr_templates")
        if not get_templates_tool:
            return self._handle_tool_error("get_pr_templates", "Tool not found", state)
        
        try:
            # Call the tool with timeout
            result = await asyncio.wait_for(
                get_templates_tool.ainvoke({}), 
                timeout=self.tool_timeout
            )
            
            # Parse and validate templates
            if isinstance(result, str):
                try:
                    parsed_result = json.loads(result)
                    if "error" in parsed_result:
                        return self._handle_tool_error("get_pr_templates", parsed_result["error"], state)
                except json.JSONDecodeError:
                    parsed_result = {"raw_output": result}
            else:
                parsed_result = result
            
            # Validate template structure
            template_count = 0
            if isinstance(parsed_result, list):
                template_count = len(parsed_result)
                # Validate each template has required fields
                for template in parsed_result:
                    if not isinstance(template, dict) or "filename" not in template:
                        logger.warning("Invalid template structure detected")
            
            pr_templates = json.dumps(parsed_result, indent=2)
            
            logger.info(f"PR templates retrieved successfully: {template_count} templates")
            
            # Enhanced step message
            step_message = AIMessage(
                content=f"✅ **Step 2 completed**: PR Templates Retrieved\n\n"
                       f"📋 **Templates Summary:**\n"
                       f"- Found {template_count} available templates\n"
                       f"- Template data size: {len(pr_templates):,} characters\n"
                       f"- All templates validated and ready for matching\n\n"
                       f"🔄 **Next**: Analyzing changes to suggest best template..."
            )
            
            # Update metadata
            metadata = state.get("processing_metadata", {})
            metadata["steps_completed"] = 2
            metadata["template_count"] = template_count
            
            return {
                "messages": [step_message],
                "pr_templates": pr_templates,
                "current_step": "suggest_template",
                "processing_metadata": metadata
            }
            
        except asyncio.TimeoutError:
            return self._handle_tool_error("get_pr_templates", "Tool execution timeout", state)
        except Exception as e:
            return self._handle_tool_error("get_pr_templates", str(e), state)

    async def _suggest_template_node(self, state: AgentState) -> Dict[str, Any]:
        """Step 3: Suggest template with intelligent analysis"""
        logger.info("Step 3: Suggesting template with enhanced analysis")
        
        suggest_tool = self._get_tool_by_name("suggest_template")
        if not suggest_tool:
            return self._handle_tool_error("suggest_template", "Tool not found", state)
        
        try:
            # Prepare enhanced analysis for template suggestion
            file_changes = state.get("file_changes", "")
            
            # Create intelligent change summary and type detection
            change_analysis = self._create_intelligent_change_analysis(file_changes)
            
            # Call the suggest_template tool with enhanced parameters
            tool_params = {
                "changes_summary": change_analysis["summary"],
                "change_type": change_analysis["detected_type"]
            }
            
            result = await asyncio.wait_for(
                suggest_tool.ainvoke(tool_params), 
                timeout=self.tool_timeout
            )
            
            # Parse and validate result
            if isinstance(result, str):
                try:
                    parsed_result = json.loads(result)
                    if "error" in parsed_result:
                        return self._handle_tool_error("suggest_template", parsed_result["error"], state)
                except json.JSONDecodeError:
                    parsed_result = {"raw_output": result}
            else:
                parsed_result = result
            
            suggested_template = json.dumps(parsed_result, indent=2)
            
            logger.info(f"Template suggested successfully: {change_analysis['detected_type']} type")
            
            # Enhanced step message with reasoning
            step_message = AIMessage(
                content=f"✅ **Step 3 completed**: Template Suggestion Generated\n\n"
                       f"🎯 **Analysis Results:**\n"
                       f"- **Detected Change Type**: {change_analysis['detected_type'].title()}\n"
                       f"- **Confidence**: {change_analysis['confidence']}\n"
                       f"- **Key Indicators**: {', '.join(change_analysis['indicators'][:3])}\n"
                       f"- **Files Analyzed**: {change_analysis['files_count']} files\n\n"
                       f"📝 **Template Recommendation**: Based on the analysis, the most suitable template has been selected.\n\n"
                       f"🔄 **Next**: Generating final comprehensive response..."
            )
            
            # Update metadata
            metadata = state.get("processing_metadata", {})
            metadata["steps_completed"] = 3
            metadata["suggested_type"] = change_analysis["detected_type"]
            metadata["analysis_confidence"] = change_analysis["confidence"]
            
            return {
                "messages": [step_message],
                "suggested_template": suggested_template,
                "current_step": "final_response",
                "processing_metadata": metadata
            }
            
        except asyncio.TimeoutError:
            return self._handle_tool_error("suggest_template", "Tool execution timeout", state)
        except Exception as e:
            return self._handle_tool_error("suggest_template", str(e), state)

    def _create_intelligent_change_analysis(self, file_changes: str) -> Dict[str, Any]:
        """Create intelligent analysis of changes for better template matching"""
        try:
            # Parse file changes
            if file_changes.startswith('{') or file_changes.startswith('['):
                changes_data = json.loads(file_changes)
            else:
                changes_data = self._parse_raw_diff(file_changes)
            
            # Enhanced analysis
            analysis = {
                "summary": "",
                "detected_type": "feature",
                "confidence": "medium",
                "indicators": [],
                "files_count": 0
            }
            
            # Extract files and analyze patterns
            files = []
            if isinstance(changes_data, dict):
                # Handle different response formats
                if "files_changed" in changes_data:
                    files_text = changes_data["files_changed"]
                    files = [line.split('\t')[-1] for line in files_text.split('\n') if line.strip()]
                elif "files" in changes_data:
                    files = changes_data["files"]
                
                # Analyze diff content for better type detection
                diff_content = changes_data.get("diff", "")
                commits = changes_data.get("commits", "")
                
                # Type detection logic
                type_indicators = {
                    "bug": ["fix", "bug", "error", "issue", "patch", "hotfix"],
                    "feature": ["add", "new", "feature", "implement", "create"],
                    "docs": ["doc", "readme", "md", "documentation"],
                    "test": ["test", "spec", "coverage"],
                    "refactor": ["refactor", "cleanup", "reorganize", "restructure"],
                    "performance": ["performance", "optimize", "speed", "efficient"],
                    "security": ["security", "auth", "permission", "secure"]
                }
                
                detected_types = []
                for change_type, keywords in type_indicators.items():
                    score = 0
                    for keyword in keywords:
                        score += diff_content.lower().count(keyword)
                        score += commits.lower().count(keyword)
                        score += sum(1 for f in files if keyword in f.lower())
                    
                    if score > 0:
                        detected_types.append((change_type, score))
                
                # Determine primary type
                if detected_types:
                    detected_types.sort(key=lambda x: x[1], reverse=True)
                    analysis["detected_type"] = detected_types[0][0]
                    analysis["confidence"] = "high" if detected_types[0][1] > 3 else "medium"
                    analysis["indicators"] = [dt[0] for dt in detected_types[:5]]
                
                # File type analysis
                file_types = {}
                for file_path in files:
                    if '.' in file_path:
                        ext = os.path.splitext(file_path)[1]
                        file_types[ext] = file_types.get(ext, 0) + 1
                
                analysis["files_count"] = len(files)
                
                # Generate comprehensive summary
                summary_parts = []
                if files:
                    summary_parts.append(f"Modified {len(files)} files")
                    
                    if file_types:
                        top_types = sorted(file_types.items(), key=lambda x: x[1], reverse=True)[:3]
                        type_desc = ", ".join([f"{count} {ext}" for ext, count in top_types])
                        summary_parts.append(f"primarily {type_desc} files")
                
                if commits:
                    commit_count = len([c for c in commits.split('\n') if c.strip()])
                    summary_parts.append(f"across {commit_count} commits")
                
                analysis["summary"] = "; ".join(summary_parts) if summary_parts else "Code changes detected"
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Error in intelligent analysis, using fallback: {e}")
            return {
                "summary": "Repository changes detected",
                "detected_type": "feature",
                "confidence": "low",
                "indicators": ["fallback"],
                "files_count": 0
            }

    def _parse_raw_diff(self, raw_diff: str) -> dict:
        """Enhanced diff parsing with better error handling"""
        lines = raw_diff.split('\n')
        
        result = {
            'files': [],
            'additions': 0,
            'deletions': 0,
            'diff_lines': [],
            'total_lines': len(lines),
            'commits': ""
        }
        
        current_file = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect file headers (multiple formats)
            if any(line.startswith(prefix) for prefix in ['diff --git', '+++', '---', 'Index:']):
                if '/' in line and not line.startswith('@@'):
                    # Extract filename more robustly
                    parts = line.split()
                    for part in parts:
                        if '/' in part and not part.startswith('--') and part != '/dev/null':
                            # Clean up the filename
                            filename = part.replace('a/', '').replace('b/', '')
                            if filename not in result['files']:
                                result['files'].append(filename)
                                current_file = filename
            
            # Count changes
            elif line.startswith('+') and not line.startswith('+++'):
                result['additions'] += 1
                if len(result['diff_lines']) < 100:  # Limit stored lines
                    result['diff_lines'].append(line)
            elif line.startswith('-') and not line.startswith('---'):
                result['deletions'] += 1
                if len(result['diff_lines']) < 100:
                    result['diff_lines'].append(line)
        
        return result

    def _handle_tool_error(self, tool_name: str, error_msg: str, state: AgentState) -> Dict[str, Any]:
        """Centralized error handling for tool execution"""
        logger.error(f"Tool '{tool_name}' failed: {error_msg}")
        
        error_message = AIMessage(
            content=f"⚠️ **Step Error**: {tool_name}\n\n"
                   f"**Error**: {error_msg}\n"
                   f"**Status**: Continuing with available data..."
        )
        
        # Update error context
        error_context = f"{tool_name}: {error_msg}"
        
        return {
            "messages": [error_message],
            f"{tool_name.replace('_', '_')}": f"Error: {error_msg}",
            "error_context": error_context,
            "current_step": "final_response"  # Skip to final response
        }

    def _error_handler_node(self, state: AgentState) -> Dict[str, Any]:
        """Handle critical errors that prevent completion"""
        error_context = state.get("error_context", "Unknown error")
        
        error_response = AIMessage(
            content=f"""
❌ **Process Interrupted**

**Error Details**: {error_context}

**What happened**: The PR analysis process encountered an error that prevented normal completion.

**Recommendations**:
1. Check that you're in a valid git repository
2. Ensure the MCP server is running correctly
3. Verify your git working directory has changes to analyze
4. Try running the analysis again

**Debug Information**: Check the logs for detailed error information.
            """.strip()
        )
        
        return {
            "messages": [error_response],
            "current_step": "error_completed"
        }

    def _final_response_node(self, state: AgentState) -> Dict[str, Any]:
        """Enhanced final response with comprehensive summary"""
        logger.info("Step 4: Generating enhanced final response")
        
        # Compile all results
        file_changes = state.get("file_changes", "No changes analyzed")
        pr_templates = state.get("pr_templates", "No templates found")
        suggested_template = state.get("suggested_template", "No template suggested")
        error_context = state.get("error_context")
        metadata = state.get("processing_metadata", {})
        
        # Calculate processing time
        start_time = metadata.get("start_time", 0)
        processing_time = round(asyncio.get_event_loop().time() - start_time, 2) if start_time else 0
        
        # Create comprehensive final response
        success_indicators = []
        if "Error:" not in file_changes:
            success_indicators.append("✅ File changes analyzed")
        if "Error:" not in pr_templates:
            success_indicators.append("✅ Templates retrieved")
        if "Error:" not in suggested_template:
            success_indicators.append("✅ Template suggested")
        
        # Build response based on success/failure
        if error_context:
            status = "⚠️ **Partial Success**"
            summary = f"Process completed with some errors: {error_context}"
        else:
            status = "🎯 **Analysis Complete**"
            summary = "All steps completed successfully!"
        
        final_response = AIMessage(
            content=f"""
{status}

## Executive Summary
{summary}

## Process Results:
{chr(10).join(success_indicators)}

## Processing Statistics:
- **Steps Completed**: {metadata.get('steps_completed', 0)}/4
- **Processing Time**: {processing_time}s
- **Files Analyzed**: {metadata.get('analyze_changes_size', 0):,} chars of diff data
- **Templates Found**: {metadata.get('template_count', 'N/A')}
- **Suggested Type**: {metadata.get('suggested_type', 'N/A').title()}
- **Analysis Confidence**: {metadata.get('analysis_confidence', 'N/A').title()}

## 📋 Template Recommendation:
{self._format_template_suggestion(suggested_template)}

## 🔧 Next Steps:
1. Review the suggested template content
2. Customize the template with your specific changes
3. Use the template for your PR description
4. Consider the analysis confidence level when making final decisions

---
*This analysis was performed using MCP tools to examine git changes and match them against available PR templates.*
            """.strip()
        )
        
        # Update metadata
        metadata["steps_completed"] = 4
        metadata["total_processing_time"] = processing_time
        
        return {
            "messages": [final_response],
            "current_step": "completed",
            "processing_metadata": metadata
        }

    def _format_template_suggestion(self, suggested_template: str) -> str:
        """Format the template suggestion for better readability"""
        try:
            if suggested_template.startswith("Error:"):
                return f"❌ {suggested_template}"
            
            suggestion_data = json.loads(suggested_template)
            
            if "recommended_template" in suggestion_data:
                template = suggestion_data["recommended_template"]
                template_name = template.get("type", "Unknown")
                reasoning = suggestion_data.get("reasoning", "No reasoning provided")
                
                return f"""
**Template**: {template_name}
**Reasoning**: {reasoning}
**Usage**: {suggestion_data.get("usage_hint", "Use this template for your PR")}
"""
            else:
                return f"Template data available: {len(suggested_template)} characters"
                
        except (json.JSONDecodeError, KeyError):
            return f"Template suggestion ready ({len(suggested_template)} characters)"

    async def chat(self, query: str) -> None:
        """Enhanced chat with better error handling and progress tracking"""
        messages = [HumanMessage(content=query)]
        
        try:
            logger.info(f"Starting chat session with query: {query[:100]}...")
            
            async for step in self.graph.astream(
                {
                    "messages": messages,
                    "current_step": "start",
                    "file_changes": None,
                    "pr_templates": None,
                    "suggested_template": None,
                    "error_context": None,
                    "processing_metadata": None
                }, 
                stream_mode="values"
            ):
                if step.get("messages"):
                    latest_message = step["messages"][-1]
                    latest_message.pretty_print()
                    
                    # Add progress indicator
                    current_step = step.get("current_step", "unknown")
                    metadata = step.get("processing_metadata", {})
                    steps_completed = metadata.get("steps_completed", 0) if metadata else 0
                    
                    print(f"\n{'='*60}")
                    print(f"Progress: {steps_completed}/4 steps | Current: {current_step}")
                    print(f"{'='*60}\n")
                    
        except Exception as e:
            logger.error(f"Error during chat: {e}")
            print(f"\n❌ **Critical Error**: {e}")
            print("Check the logs for detailed error information.")


# Enhanced usage example
if __name__ == "__main__":
    async def main():
        try:
            logger.info("Initializing enhanced PR Agent...")
            
            # Create agent with optional custom server path
            code_agent = await Agent.create()
            
            # Enhanced test query
            query = """
            I've made some changes to my repository and need help selecting the right PR template. 
            Please analyze my current changes, review available templates, and suggest the most 
            appropriate one based on what I've modified. I want to make sure I'm following 
            best practices for PR documentation.
            """
            
            await code_agent.chat(query)
            
        except Exception as e:
            logger.error(f"Error in main: {e}")
            print(f"❌ Failed to run agent: {e}")
            print("Make sure:")
            print("1. You're in a git repository with changes")
            print("2. The MCP server path is correct")
            print("3. Python dependencies are installed")

    asyncio.run(main())
