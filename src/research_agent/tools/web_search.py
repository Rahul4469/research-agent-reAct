"""
Web search and scraping tools for the reasearch agent.
1. tools definitions - web_search, Fetch_page
2. HTTP client with pooling and rate limit
3. HTML parsing utils
4. WEbSearchTool & FetchPageTool classes
5. ToolRegistry
"""
import asyncio
import re
from typing import Any
from urllib.parse import quote_plus, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from research_agent.core.models import SearchResult, SearchResponse, ToolDefinition, ToolParameter


# TOOL Definitions ///

WEB_SEARCH_TOOL = ToolDefinition(
    name="web_search",
    description="""Search the web for information on any topic.
    Use this tool when you need to find current information, facts, or research on a topic.
    Returns a list of relevant search results with titles, URLs, and snippets.""",
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="The search query to look up",
            required=True,
        ),
        ToolParameter(
            name="num_results",
            type="integer",
            description="Number of results to return (1-10)",
            default=5,
        ),
    ],
)

FETCH_PAGE_TOOL = ToolDefinition(
    name="fetch_page",
    description="""Fetch and extract the main content from a web page.
    Use this when you need to read the full content of a specific URL.
    Returns the page title and main text content.""",
    parameters=[
        ToolParameter(
            name="url",
            type="string",
            description="The URL to fetch",
            required=True,
        ),
         ToolParameter(
            name="max_length",
            type="integer",
            description="Maximum characters to return",
            required=False,
            default=5000,
        ),
    ],
)

# HTTP Client with RATE LIMITING ///----------------------

class WebClient:
    """
    HTTP client with connection pooling and rate limiting.
    Python httpx:
        client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
            )
        )
    """
    
    def __init__(self, timeout: float = 30.0, max_connections: int = 100, 
                 requests_per_second: float=2.0,):
        self._timeout = httpx.Timeout(timeout)
        self._limits = httpx.Limits(max_connections=max_connections,
                                max_keepalive_connections=20,
                                )
        
        # Rate limmiting state
        self._rate_limit = requests_per_second
        self._last_request_time: float = 0
        self._rate_lock = asyncio.Lock()
        
        # Lazy inititalize client
        self._client: httpx.AsyncClient | None = None
        
        # Common headers to avoid being blocked
        self._headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        
    async def _get_client(self) -> httpx.AsyncClient:
        """
        Get or create a HTTP client.
        We don't create the client in __init__ because:
        1. Creating async resources in __init__ is problematic
        2. Client might never be used
        3. Allows async setup if needed
        """
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                limits=self._limits,
                headers=self._headers,
                follow_redirects=True,
            )
        return self._client
    
    async def _rate_limit_wait(self) -> None:
        """
        Wait if necessary to respect rate limits.
        """
        async with self._rate_lock:
            import time
            current_time = time.monotonic()
            time_since_last = current_time - self._last_request_time
            min_interval = 1.0 / self._rate_limit

            if time_since_last < min_interval:
                wait_time = min_interval - time_since_last
                await asyncio.sleep(wait_time)

            self._last_request_time = time.monotonic()
        
    async def get(self, url: str) -> httpx.Response:
        """
        Make a rate-limited GET request.

        Args:
            url: URL to fetch

        Returns:
            HTTP response object

        Raises:
            httpx.HTTPError: If request fails
        """
        await self._rate_limit_wait()
        client = await self._get_client()
        return await client.get(url)
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
                    
# HTML Parsing Utilities /// --------------------------

def extract_main_content(html: str, max_length: int = 5000) -> dict[str, Any]:
    """
    Extract the main content from an HTML page.
    
     TEACHING: BeautifulSoup Basics
    =================================
    BeautifulSoup parses HTML into a tree you can query:
    
    soup = BeautifulSoup(html, 'lxml')
    
    Finding elements:
    soup.find('title')           # First <title> tag
    soup.find_all('p')           # All <p> tags
    soup.select('div.content')   # CSS selector
    soup.select_one('#main')     # First match for CSS selector
    
    Getting text:
    element.text                  # All text, including children
    element.get_text(strip=True)  # Cleaned text
    
    Getting attributes:
    element['href']              # Get attribute (raises if missing)
    element.get('href')          # Get attribute (None if missing)
    
    Args:
        html: Raw HTML content
        max_length: Maximum characters to return
        
    Returns:
        Dictionary with title, content, and metadata
    """
    soup = BeautifulSoup(html, "lxml")
    
    # Get title
    title = ""
    if soup.title:
        title = soup.title.get_text(strip=True)
    
    # Remove unwanted elements
    for tag in soup.select("script, style, nav, header, footer, aside, .ad, .sidebar"):
        tag.decompose()  # Remove from tree
    
    """
    🎓 TEACHING: CSS Selectors
    ==========================
    soup.select() uses CSS selector syntax:
    
    'p'              - All <p> tags
    '.class'         - All elements with class="class"
    '#id'            - Element with id="id"
    'div p'          - All <p> inside <div>
    'div > p'        - Direct child <p> of <div>
    'a[href]'        - All <a> with href attribute
    'div.content p'  - <p> inside <div class="content">
    """
    
    # Try to find main content area
    main_content = (
        soup.select_one("article") or
        soup.select_one("main") or
        soup.select_one('[role="main"]') or
        soup.select_one(".content") or
        soup.select_one("#content") or
        soup.body
    )
    
    if not main_content:
        return {"title": title, "content": "", "links": []}
    
    # Extract paragraphs
    paragraphs = []
    for p in main_content.find_all(["p", "h1", "h2", "h3", "li"]):
        text = p.get_text(strip=True)
        if len(text) > 20:  # Skip short/empty paragraphs
            paragraphs.append(text)
    
    content = "\n\n".join(paragraphs)
    
    # Truncate if needed
    if len(content) > max_length:
        content = content[:max_length] + "..."
    
    # Extract links for citations
    links = []
    for a in main_content.select("a[href]")[:10]:
        href = a.get("href", "")
        text = a.get_text(strip=True)
        if href and text and not href.startswith("#"):
            links.append({"text": text, "href": href})
    
    return {
        "title": title,
        "content": content,
        "links": links,
    }


def extract_search_results(html: str, base_url: str) -> list[dict[str, str]]:
    """
    Extract search results from a search engine results page.
    
    Note: This is a simplified extractor. In production, you'd use
    a proper search API (Google Custom Search, Bing, Brave, etc.)
    or a service like SerpAPI.
    
    Args:
        html: Search results page HTML
        base_url: Base URL for resolving relative links
        
    Returns:
        List of results with title, url, snippet
    """
    soup = BeautifulSoup(html, "lxml")
    results = []
    
    # Generic extraction - tries common patterns
    for item in soup.select(".result, .g, .search-result, article"):
        title_elem = item.select_one("h2, h3, .title, a")
        link_elem = item.select_one("a[href]")
        snippet_elem = item.select_one("p, .snippet, .description")
        
        if title_elem and link_elem:
            title = title_elem.get_text(strip=True)
            href = link_elem.get("href", "")
            snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
            
            # Resolve relative URLs
            if href and not href.startswith("http"):
                href = urljoin(base_url, href)
            
            if title and href:
                results.append({
                    "title": title,
                    "url": href,
                    "snippet": snippet,
                })
    
    return results[:10]
                    
                    
# Search Tool Implementation /// -----------------------------------------

class WebSearchTool:
    """
    Each tool in our agent follows this pattern:
    
    1. A ToolDefinition (schema for the LLM)
    2. An implementation class with execute() method
    3. Input validation (via Pydantic)
    4. Proper error handling                    
    """

    def __init__(self, web_client: WebClient | None = None):
        self._client = web_client or WebClient()
        self.definition = WEB_SEARCH_TOOL
    
    async def execute(self, query: str, num_results: int = 5,) -> SearchResponse:
        # mock_results = []
        # for i in range(min(result_results, 5)):
        #    mock_results.append(SearchResults(...))"""
        mock_results = [
            SearchResult(
                title=f"Result {i+1} for: {query}",
                url=f"https://example.com/results/{i+1}",
                snippet=f"This is a search result about {query}. "
                        f"It contains relevant information that would help answer the query.",
            )
            for i in range(min(num_results, 5))
        ]
        # return SearchResponse(
        #     query = query,
        #     results = mock_results,
        #     total_results = len(mock_results),
        # )
        return await self.execute_real(query, num_results)
    
    async def execute_real(self, query: str, num_results: int = 5, search_api_key: str | None = None) -> SearchResponse:
        """
        Execute real web search using SerpAPI.

        Args:
            query: Search query string
            num_results: Number of results to return
            search_api_key: Optional API key (falls back to SERP_API_KEY env var)
        """
        import os

        api_key = search_api_key or os.environ.get("SERP_API_KEY")
        if not api_key:
            raise ValueError("SERP_API_KEY environment variable not set")

        # SerpAPI endpoint
        url = "https://serpapi.com/search"
        params = {
            "engine": "google",
            "q": query,
            "num": num_results,
            "api_key": api_key,  # SerpAPI uses api_key as query param
        }

        client = await self._client._get_client()
        response = await client.get(url, params=params)
        response.raise_for_status()

        data = response.json()

        # SerpAPI returns results in "organic_results" array
        results = [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),  # SerpAPI uses "link" not "url"
                snippet=item.get("snippet", ""),
            )
            for item in data.get("organic_results", [])[:num_results]
        ]

        return SearchResponse(
            query = query,
            results = results,
            total_results = len(results)
        )

class FetchPageTool:
    def __init__(self, web_client: WebClient | None = None):
        self._client = web_client or WebClient()
        self.definition = FETCH_PAGE_TOOL
    
    async def execute(self, url: str, max_length: int = 5000) -> dict[str, Any]:
        """
        Fetch a web page and extract+parse its main content.
        Args:
            url: URL to fetch
            max_length: Maximum content length
            
        Returns:
            Dictionary with title, content, and metadata
        """
        try:
            response = await self._client.get(url)
            response.raise_for_status()

            content = extract_main_content(response.text, max_length)
            content["url"] = url
            content["status"] = "success"
            
            return content
        
        except httpx.HTTPError as e:
            return {
                "url": url,
                "status": "error",
                "error": str(e),
                "title": "",
                "content": "",
            }


# TOOL REGISTRY ///--------------------------

class ToolRegistry:
    """
    Registry of available tools for the agent.
    
    A registry centralizes tool management:
    - Tools register themselves
    - Agent queries registry for available tools
    - Easy to add/remove tools dynamically
    
    init__ tools: dict[str, Any], definitions: dict[str, ToolDefinition]
    """
    def __init__(self):
        self._tools: dict[str, Any] = {}
        self._definitions: dict[str, ToolDefinition] = {}
    
    def register(self, name: str, tool: Any, definition: ToolDefinition) -> None:
        """Register a tool with the registry."""
        self._tools[name] = tool
        self._definitions[name] = definition
    
    def get_tool(self, name: str) -> Any | None:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def get_definition(self, name: str) -> ToolDefinition | None:
        """Get a tool definition by name."""
        return self._definitions.get(name)
    
    def list_definitions(self) -> list[ToolDefinition]:
        """Get all tool definitions (for sending to LLM)."""
        return list(self._definitions.values())
    
    def list_names(self) -> list[str]:
        """Get all registered tool names."""
        return list(self._tools.keys())
    
def create_default_registry() -> ToolRegistry:
       """
       Usage:
       registry = create_default_registry()
       agent = ResearchAgent(registry=registry)
       """
       registry = ToolRegistry()
       
       # Shared web client for connection pooling
       web_client = WebClient()
       
       # Register search tool
       search_tool = WebSearchTool(web_client)
       registry.register("web_search", search_tool, search_tool.definition)
       
       # Register fetch tool
       fetch_tool = FetchPageTool(web_client)
       registry.register("fetch_page", fetch_tool, fetch_tool.definition)
       
       return registry