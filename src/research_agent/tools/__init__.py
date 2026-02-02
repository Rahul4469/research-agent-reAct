"""Tools module."""
from research_agent.tools.web_search import (
    # Tool definitions
    WEB_SEARCH_TOOL,
    FETCH_PAGE_TOOL,
    # Classes
    WebClient,
    WebSearchTool,
    FetchPageTool,
    ToolRegistry,
    create_default_registry,
)

__all__ = [
    # Tool definitions
    "WEB_SEARCH_TOOL",
    "FETCH_PAGE_TOOL",
    # Classes
    "WebClient",
    "WebSearchTool",
    "FetchPageTool",
    "ToolRegistry",
    "create_default_registry",
]
