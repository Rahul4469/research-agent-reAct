import json
import re
from typing import Any

from research_agent.core.models import (
    AgentResponse,
    Message,
    Role,
    ThoughtAction,
    ToolDefinition
)

from research_agent.llm.client import AsyncLLMClient, LLMClient
from research_agent.tools.web_search import ToolRegistry, create_default_registry

# PROMPTS ///-----------------------------------

REACT_SYSTEM_PROMPT = """You are a research assistant that helps users find accurate, up-to-date information.

You have access to the following tools:
{tool_descriptions}

## How to respond

For EVERY query that needs external information, you MUST follow this exact format:

Thought: [Your reasoning about what to do next]
Action: [tool_name]
Action Input: {{"param1": "value1", "param2": "value2"}}

After receiving an observation, continue with another Thought, or provide your final answer.

When you have enough information to answer, respond with:

Thought: [Your final reasoning]
Final Answer: [Your comprehensive answer with citations]

## Important Rules

1. ALWAYS use Thought before Action or Final Answer
2. Action Input MUST be valid JSON
3. Only use tools that are listed above
4. If a search returns no results, try rephrasing the query
5. Cite your sources in the final answer
6. If you cannot find the information, say so honestly

## Example

User: What is the capital of France?

Thought: This is a factual question. While I know the answer, let me verify with a search to ensure accuracy.
Action: web_search
Action Input: {{"query": "capital of France"}}

Observation: France's capital is Paris, which has been the capital since 987 CE...

Thought: The search confirms Paris is the capital. I can now provide the final answer.
Final Answer: The capital of France is **Paris**. It has served as the nation's capital since 987 CE and is the country's largest city with a population of over 2 million in the city proper.

Now help the user with their query."""

def format_tool_description(tools: list[ToolDefinition]) -> str:
    """
    Format tool definitions for the system prompt.
    This will be used on tool registry(literal def of tools)
    """
    description = []
    for tool in tools:
        params = []
        for p in tool.parameters:
            req = "(required)" if p.required else "(optional)"
            params.append(f"  - {p.name}: {p.description} {req}")
        
        param_str = "\n".join(params) if params else " (no parameters)"
        description.append(f"### {tool.name}\n{tool.description}\nParameters:\n{param_str}")

    return "\n\n".join(description)

# PARSING UTILITIES ///-----------------------------------------

class ReActParser:
    """
    Parser for ReAct-formatted response.
    - Use r'' (raw strings) for regex to avoid escaping backslashes
    - re.DOTALL makes . match newlines
    - re.IGNORECASE for case-insensitive matching
    
    # Has @classmethod parse
    """
    # Patterns for parsing ReAct output
    THOUGHT_PATTERN = re.compile(r"Thought:\s*(.+?)(?=Action:|Final Answer:|$)", re.DOTALL)
    ACTION_PATTERN = re.compile(r"Action:\s*(\w+)", re.IGNORECASE)
    ACTION_INPUT_PATTERN = re.compile(r"Action Input:\s*(\{.*?\})", re.DOTALL)
    FINAL_ANSWER_PATTERN = re.compile(r"Final Answer:\s*(.+)", re.DOTALL)
    
    @classmethod
    def parse(cls, text: str) -> dict[str, Any]:
        """
        Parse a ReAct-formatted response.
        
        Returns:
            Dictionary with:
            - thought: The reasoning (always present)
            - action: Tool name (if action taken)
            - action_input: Tool parameters (if action taken)
            - final_answer: The answer (if complete)
            - is_complete: Whether agent is done
        """
        result: dict[str, Any] = {
            "thought": None,
            "action": None,
            "action_input": None,
            "final_answer": None,
            "is_complete": False,
        }
        
        # Extract thought
        thought_match = cls.THOUGHT_PATTERN.search(text)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()
            
        # Check for final answer first
        final_match = cls.FINAL_ANSWER_PATTERN.search(text)
        if final_match:
            result["final_answer"] = final_match.group(1).strip()
            result["is_complete"] = True
            return result   
        
        # Extract action
        action_match = cls.ACTION_PATTERN.search(text)
        if action_match:
            result["action"] = action_match.group(1).strip()
            
        # Extract action input (JSON)
        input_match = cls.ACTION_INPUT_PATTERN.search(text)
        if input_match:
            try:
                result["action_input"] = json.loads(input_match.group(1))
            except json.JSONDecodeError:
                # Try to fix common JSON issues
                json_str = input_match.group(1)
                # Replace single quotes with double quotes
                json_str = json_str.replace("'", '"')
                try:
                    result["action_input"] = json.loads(json_str)
                except json.JSONDecodeError:
                    result["action_input"] = {"raw": input_match.group(1)} 
        
        return result       
    
# SYNCHRONOUS AGENT /// ---------------------------------

class ResearchAgent:
    """
    The agent has three main components:
    
    1. LLM Client - for calling Claude
    2. Tool Registry - available tools
    3. Memory - conversation history     
    
    The run() method implements the ReAct loop:
    
        while not done:
            response = llm.complete(messages)
            parsed = parse_react_format(response)
            
            if parsed.is_final_answer:
                return parsed.answer
            
            if parsed.has_action:
                result = execute_tool(parsed.action, parsed.input)
                messages.append(observation(result))
    """     
    def __init__(self, llm_client: LLMClient | None = None,
                 registry: ToolRegistry | None = None,
                 max_iterations: int = 10,
                 verbose: bool = False):
           self._llm = llm_client or LLMClient()
           self._registry = registry or create_default_registry()   
           self._max_iterations = max_iterations     
           self._verbose = verbose
           self._parser = ReActParser()
            
           # Build system prompt with tool descriptions
           tool_descriptions = format_tool_description(self._registry.list_definitions())
           self._system_prompt = REACT_SYSTEM_PROMPT.format(
               tool_descriptions=tool_descriptions
           ) 
    
    def run(self, query: str) -> AgentResponse:
        """
        This is the main entry point. It:
        1. Initializes conversation with the query
        2. Runs the ReAct loop
        3. Returns structured response
        
        Args:
            query: User's question
            
        Returns:
            AgentResponse with answer, confidence, sources, and reasoning
        """   
        # Initial conversation   
        messages = [Message(role = Role.USER, content=query)] 
        reasoning_steps: list[ThoughtAction] = []
        sources: list[str] = []
        
        if self._verbose:
            print(f"Query: {query}")
            
        for iteration in range(self._max_iterations):
            if self._verbose:
                print(f"\n--- Iteration {iteration + 1} ---")
            
            # Get LLM response
            response = self._call_llm(messages)  
            
            # Parse the reponse
            parsed = self._parser.parse(response)     
            
            if self._verbose:
                print(f"thought: {parsed['thought']}")
            
            # Create reasoning step from parsed response from llm
            step = ThoughtAction(
                thought=parsed["thought"] or "",
                action=parsed["action"],
                action_input=parsed["action_input"]
            )

            # Check if we have a final answer
            if parsed["is_complete"]:
                if self._verbose:
                    print(f"\nFinal Answer: {parsed['final_answer']}")

                return AgentResponse(
                    answer=parsed["final_answer"] or "",
                    confidence=0.8,
                    sources=sources,
                    reasoning_steps=reasoning_steps,
                )

            # Execute action if present
            if parsed["action"]:
                if self._verbose:
                    print(f"Action: {parsed['action']}")
                    print(f"Input: {parsed['action_input']}")

                observation = self._execute_tool(
                    parsed["action"],
                    parsed["action_input"] or {},
                )

                step.observation = observation

                if self._verbose:
                    print(f"Observation: {observation[:200]}...")

                # Add to messages for next iteration
                messages.append(Message(role=Role.ASSISTANT, content=response))
                messages.append(Message(
                    role=Role.USER,
                    content=f"Observation: {observation}"
                ))

                # Track sources from search results
                if "url" in str(parsed["action_input"]):
                    sources.append(str(parsed["action_input"].get("url", "")))
            
            reasoning_steps.append(step)          
        
        # Max iterations reached
        return AgentResponse(
            answer="I was unable to find a complete answer within the allowed steps. "
                   "Please try rephrasing your question or breaking it into smaller parts.",
            confidence=0.3,
            sources=sources,
            reasoning_steps=reasoning_steps,
        )    
    
    def _call_llm(self, messages: list[Message]) -> str:
        """Call the LLM and extract text response."""
        # We use the raw API here without tools since we're doing
        # ReAct-style parsing instead of Claude's native tool use
        response = self._llm.complete(messages, system=self._system_prompt)
        
        # Extract text content
        for block in response.get("content", []):
            if block.get("type") == "text":
                return block.get("text", "")
        
        return ""    
    
    def _execute_tool(self, tool_name: str, inputs: dict[str, Any]) -> str:
        
        tool = self._registry.get_tool(tool_name)
        
        if tool is None:
            return f"Error: Unknown tool '{tool_name}'. Available tools: {self._registry.list_names()}"
        
        try:
            # For sync agent, we need to run async tools in event loop
            import asyncio
            
            # Get or create event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            # Run async tool
            result = loop.run_until_complete(tool.execute(**inputs))    
            
            # Convert result to string for observation
            if hasattr(result, "model_dump_json"):
                return result.model_dump_json(indent=2)
            return str(result)
        
        except TypeError as e:
            return f"Error: Invalid inputs for tool '{tool_name}': {e}"
        except Exception as e:
            return f"Error executing tool '{tool_name}': {e}"

# ASYNC AGENT

class AsyncResearchAgent:
    def __init__(
        self,
        llm_client: AsyncLLMClient | None = None,
        registry: ToolRegistry | None = None,
        max_iterations: int = 10,
        verbose: bool = False,
    ):
        self._llm = llm_client or AsyncLLMClient()
        self._registry = registry or create_default_registry()
        self._max_iterations = max_iterations
        self._verbose = verbose
        self._parser = ReActParser()
        
        tool_descriptions = format_tool_description(
            self._registry.list_definitions()
        )
        self._system_prompt = REACT_SYSTEM_PROMPT.format(
            tool_descriptions=tool_descriptions
        )
    
    async def run(self, query: str) -> AgentResponse:
        """
        Run the agent asynchronously.
        
        Usage:
            agent = AsyncResearchAgent()
            response = await agent.run("What is quantum computing?")
        """
        messages = [Message(role=Role.USER, content=query)]
        reasoning_steps: list[ThoughtAction] = []
        sources: list[str] = []
        
        if self._verbose:
            print(f"Query: {query}")
            print(f"Available tools: {self._registry.list_names()}")

        for iteration in range(self._max_iterations):
            if self._verbose:
                print(f"\n--- Iteration {iteration + 1} ---")

            # Get LLM response (async!)
            response = await self._call_llm(messages)

            if self._verbose:
                print(f"\n[RAW LLM RESPONSE]:\n{response[:500]}...")

            # Parse the response
            parsed = self._parser.parse(response)

            if self._verbose:
                # print(f"Thought: {parsed['thought']}")
                print(f"\n[PARSED]:")
                print(f"  Thought: {parsed['thought']}")
                print(f"  Action: {parsed['action']}")
                print(f"  Action Input: {parsed['action_input']}")
                print(f"  Is Complete: {parsed['is_complete']}")
            
            step = ThoughtAction(
                thought=parsed["thought"] or "",
                action=parsed["action"],
                action_input=parsed["action_input"],
            )
            
            if parsed["is_complete"]:
                if self._verbose:
                    print(f"\nFinal Answer: {parsed['final_answer']}")
                
                return AgentResponse(
                    answer=parsed["final_answer"] or "",
                    confidence=0.8,
                    sources=sources,
                    reasoning_steps=reasoning_steps,
                )
            
            if parsed["action"]:
                if self._verbose:
                    # print(f"Action: {parsed['action']}")
                    #print(f"Input: {parsed['action_input']}")
                    print(f"\n[EXECUTING TOOL]: {parsed['action']}")

                # Execute tool (async!)
                observation = await self._execute_tool(
                    parsed["action"],
                    parsed["action_input"] or {},
                )
                
                step.observation = observation
                
                if self._verbose:
                    obs_preview = observation[:200] + "..." if len(observation) > 200 else observation
                    print(f"Observation: {obs_preview}")
                
                messages.append(Message(role=Role.ASSISTANT, content=response))
                messages.append(Message(
                    role=Role.USER, 
                    content=f"Observation: {observation}"
                ))
            
            reasoning_steps.append(step)
        
        return AgentResponse(
            answer="Maximum iterations reached without finding a complete answer.",
            confidence=0.3,
            sources=sources,
            reasoning_steps=reasoning_steps,
        )
    
    async def _call_llm(self, messages: list[Message]) -> str:
        """Call the LLM asynchronously."""
        try:
            response = await self._llm.complete(messages, system=self._system_prompt)

            for block in response.get("content", []):
                if block.get("type") == "text":
                    return block.get("text", "")

            return ""
        except Exception as e:
            # Handle API errors gracefully
            error_msg = str(e)
            if "content filtering" in error_msg.lower():
                return "Thought: The query triggered a content filter. Let me try a different approach.\nFinal Answer: I apologize, but I cannot process this query due to content restrictions. Please try rephrasing your question."
            raise
    
    async def _execute_tool(self, tool_name: str, inputs: dict[str, Any]) -> str:
        """Execute a tool asynchronously."""
        tool = self._registry.get_tool(tool_name)
        
        if tool is None:
            return f"Error: Unknown tool '{tool_name}'. Available tools: {self._registry.list_names()}"
        
        try:
            result = await tool.execute(**inputs)
            
            if hasattr(result, "model_dump_json"):
                return result.model_dump_json(indent=2)
            return str(result)
            
        except TypeError as e:
            return f"Error: Invalid inputs for tool '{tool_name}': {e}"
        except Exception as e:
            return f"Error executing tool '{tool_name}': {e}"
       
# Covenience func to create AGENT
              
def create_agent(api_key: str | None = None,
                verbose: bool = False,
                async_mode: bool = False,) -> ResearchAgent | AsyncResearchAgent:
    
    registry = create_default_registry()
    
    if async_mode:
        client = AsyncLLMClient(api_key = api_key)
        return AsyncResearchAgent(
            llm_client=client,
            registry=registry,
            verbose=verbose,
        )
    else:
        client = LLMClient(api_key = api_key)
        return ResearchAgent(
            llm_client=client,
            registry=registry,
            verbose=verbose,
        )    
