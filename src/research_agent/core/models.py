"""
Core data models for research agent.
"""
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

# BASIC TYPES

class Role (str, Enum):
    """
    Message roles in a conversation
    Role with Enum is like named string(type) const
    
    USER, ASSISTANT, SYSTEM
    """
    USER = "user"
    ASSISTANT = "assitant"
    SYSTEM = "system"
    
class Message(BaseModel):    
    """
    A single message in a converstion
    Pydantic- automatically validating and 
    converting incoming data (e.g., from JSON or HTTP requests).
    BaseModel- BaseModel is the core Pydantic base class you 
    inherit from to define your own models
    
    role, content, model_config
    """
    role: Role
    content: str

    # This is called "model_config" in Pydantic v2
    # Similar to struct tags in Go
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"role": "user", "content": "What is Python"}
            ]
        }
    }
    
# TOOL DEFINITIONs ----------------------------------------

class ToolParameter(BaseModel):
    """Schema for a single tool parameter.
    (name, type, descriptionn, required)
    """
    name: str
    type: str = Field(default="string", description="JSON schema type")
    description: str
    required: bool = True
    # Any is like interface{} in Go - accepts anything
    defualt: Any | None = None
        
class ToolDefinition(BaseModel):
    """ 
    Definition of a tool that the agent can use.
    description, default(if not provided), ge, le, gt, lt
    patterns: Regex validation, min_length, 
    Example:
        age: int = Field(ge=0, le=150, description="Person's age")
        
    name, description, parameters list[ToolParamter],
    return to_anthropic_schema() -> dict[str, ANY]    
    """
    name: str = Field(description="Unique identifier for the tool")
    description: str = Field(description="What the tool does - LLM reads this!")
    parameters: list[ToolParameter] = Field(default_factory=list)
    
    def to_anthropic_schema(self) -> dict[str, Any]:
        """
        Convert to Anthropic's tool format.
        """
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = {
                "type": param.type,
                "desciption": param.description,
            }
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        }
       
# Agent STATE & RESPONSES

class ThoughtAction(BaseModel):
    """
    A single step in the ReAct loop.        
    This represents one iteration of:
    Thought → Action → Observation
    
    thought, action?, action_input?, observation?
    """
    thought: str = Field(description="Agent's reasoning about what to do")
    action: str | None = Field(default=None, description="Tool to call, if any")
    action_input: dict[str, Any] | None = Field(default=None, description="Tool parameters")
    observation: str | None = Field(default=None, description="Result from tool")
    
class AgentResponse(BaseModel):
    """
    Final response from the agent.
    If you try AgentResponse(confidence=1.5), 
    Pydantic raises ValidationError!
    
    answer, confidence, source, resoning_steps, num_steps()
    """
    answer: str
    confidence: float = Field(
        ge=0, 
        le=1, 
        description="How confident the agent is (0-1)"
    )
    source: list[str] = Field(
        default_factory=list,  # Empty list if not provided
        description="URLs or references used"
    )
    
    reasoning_steps: list[ThoughtAction] = Field(
        default_factory=list,
        description="The chain of reasoning that led to this answer"
    )
    
    # Computed property - like a method in Go struct
    @property
    def num_steps(self) -> int:
        """Number of reasoning steps taken."""
        return len(self.reasoning_steps)
    
# SEARCH-SPECIFIC Types

class SearchResult(BaseModel):
    """ 
    SearchResult model for SearchResponse. 
    A single search result from web search. 
    title, url, snippet, content?
    """
    title: str
    url: str
    snippet: str
    content: str | None = None

class SearchResponse(BaseModel):
    """
    Response from a web search operation.
    query, results: [SearchResult], total_results
    """
    query: str
    results: list[SearchResult]
    total_results: int = 0
    
    @property   
    def has_results(self) -> bool:
        """Check if search returned any results."""
        return len(self.results) > 0 
    
    
# Test Instances //////
if __name__ == "__main__":
    """
    For testing models.py
    When `python models.py` is run, this block executes.
    When its imported elsewhere, it doesn't.  
    """  
    # 1- Create a message
    # 2- Create tool definition
    
    # Create a message - Pydantic validates automatically
    msg = Message(role=Role.USER, content="Hello!")
    print(f"Message: {msg}")
    print(f"As JSON: {msg.model_dump_json()}")
    
    # Create a tool definition
    search_tool = ToolDefinition(
        name="web_search",
        description="Search the web for information",
        parameters=[
            ToolParameter(
                name="query",
                type="string",
                description="The search query",
                required= True,
            ),
            ToolParameter(
                name="num_results",
                type="integer",
                description="Number of results to return",
                required=False,
                default=5,
            )
        ])
    print(f"\nTool schema for Anthropic API:")
    import json
    print(json.dumps(search_tool.to_anthropic_schema(), indent=2))
    
    # ValidationError - confidence out of range
    
    try:
        bad_response = AgentResponse(
            answer="test",
            confidence=1.5,
        )
    except Exception as e:
        print(f"\n Validation error (expected): {e}")       