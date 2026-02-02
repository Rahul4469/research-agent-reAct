"""
Research Agent - An AI-powered research assistant.

This package provides:
- ResearchAgent: Synchronous agent for research queries
- AsyncResearchAgent: Async agent for research queries
- LLMClient: Wrapper for Claude API
- Tools: Web search and page fetching
"""

__version__ = "0.1.0"

# Core agent classes
from research_agent.core import (
    ResearchAgent,
    AsyncResearchAgent,
    create_agent,
    ReActParser,
)

# Models
from research_agent.core.models import (
    Role,
    Message,
    ToolParameter,
    ToolDefinition,
    ThoughtAction,
    AgentResponse,
    SearchResult,
    SearchResponse,
)

# LLM client
from research_agent.llm import (
    LLMClient,
    AsyncLLMClient,
    LLMResponse,
    ToolCallRequest,
    MessageBuilder,
)

# Tools
from research_agent.tools import (
    WebClient,
    WebSearchTool,
    FetchPageTool,
    ToolRegistry,
    create_default_registry,
)

__all__ = [
    "__version__",
    # Core
    "ResearchAgent",
    "AsyncResearchAgent",
    "create_agent",
    "ReActParser",
    # Models
    "Role",
    "Message",
    "ToolParameter",
    "ToolDefinition",
    "ThoughtAction",
    "AgentResponse",
    "SearchResult",
    "SearchResponse",
    # LLM
    "LLMClient",
    "AsyncLLMClient",
    "LLMResponse",
    "ToolCallRequest",
    "MessageBuilder",
    # Tools
    "WebClient",
    "WebSearchTool",
    "FetchPageTool",
    "ToolRegistry",
    "create_default_registry",
]
