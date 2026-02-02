"""
Anthropic Claude client wrapper for the research agent.

🎓 MODULE OVERVIEW
==================
This module wraps the Anthropic Python SDK to provide a clean interface
for our research agent. It handles:

1. API authentication and configuration
2. Message formatting and conversion
3. Both synchronous and asynchronous API calls
4. Streaming responses for real-time output
5. Tool/function calling support

🎓 TEACHING: Why Wrap the SDK?
==============================
You might ask: "Why not use the Anthropic SDK directly?"

Reasons for wrapping:
1. ABSTRACTION - Hide SDK complexity from the rest of our code
2. CONSISTENCY - Ensure uniform message formatting throughout
3. TESTABILITY - Easy to mock for unit tests
4. FLEXIBILITY - Can swap LLM providers without changing agent code
5. DEFAULTS - Set sensible defaults for our use case

Compare to Go:
    Go: You'd create a struct with methods that wrap http.Client
    Python: We create classes that wrap the Anthropic client

    // Go approach
    type LLMClient struct {
        client *http.Client
        apiKey string
    }

    # Python approach
    class LLMClient:
        def __init__(self, api_key: str):
            self._client = Anthropic(api_key=api_key)
"""

# import os
# from typing import Any, AsyncIterator, Iterator
# from dataclasses import dataclass, field

# from anthropic import AsyncAnthropic, Anthropic
# from anthropic.types import Message as AnthropicMessage, ContentBlock

# from pydantic import BaseModel, Field

# from research_agent.core.models import Message, Role, ToolDefinition


# =============================================================================
# 🎓 TEACHING: Configuration Constants
# =============================================================================
"""
🎓 TEACHING: Module-Level Constants
===================================
In Python, we define configuration at module level using UPPER_CASE names.
This is a convention (not enforced) that signals "don't modify this."

Compare to Go:
    // Go uses const blocks
    const (
        DefaultModel = "claude-sonnet-4-20250514"
        MaxTokens    = 4096
    )

    # Python uses module-level variables
    DEFAULT_MODEL = "claude-sonnet-4-20250514"
    MAX_TOKENS = 4096

Key differences:
- Go constants are truly immutable (compile-time)
- Python "constants" can technically be changed (runtime)
- Convention is the only enforcement in Python
"""

DEFAULT_MODEL = "claude-sonnet-4-20250514"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.7

# Available Claude models (as of 2024)
CLAUDE_MODELS = {
    "sonnet": "claude-sonnet-4-20250514",     # Fast, good balance
    "opus": "claude-opus-4-20250514",          # Most capable
    "haiku": "claude-3-5-haiku-20241022",      # Fastest, cheapest
}


# =============================================================================
# 🎓 RESPONSE MODELS - Pydantic for Type Safety
# =============================================================================
"""
🎓 TEACHING: Pydantic vs Dataclasses vs NamedTuple
==================================================
Python has several ways to define structured data:

1. Pydantic BaseModel - Full validation, JSON serialization, schemas
   Use for: API boundaries, data that needs validation

2. dataclass - Lightweight, less overhead, no validation
   Use for: Internal data structures, simple DTOs

3. NamedTuple - Immutable, memory efficient
   Use for: Fixed data, dictionary keys

Compare to Go:
    // Go uses structs for everything
    type LLMResponse struct {
        Content    string
        TokensUsed int
        Model      string
    }

    # Python Pydantic
    class LLMResponse(BaseModel):
        content: str
        tokens_used: int
        model: str

    # Python dataclass
    @dataclass
    class LLMResponse:
        content: str
        tokens_used: int
        model: str

We use Pydantic for external data (API responses) and dataclass for internal.
"""


class ToolCallRequest(BaseModel):
    """
    Represents a tool call requested by the LLM.

    🎓 TEACHING: When Claude Wants to Use a Tool
    ============================================
    Claude's tool use follows this flow:

    1. User sends message + available tools
    2. Claude decides to use a tool
    3. Claude returns a "tool_use" content block with:
       - id: Unique identifier for this call
       - name: Which tool to invoke
       - input: JSON arguments for the tool
    4. You execute the tool and send results back
    5. Claude continues with the tool result

    Example tool_use block from API:
    {
        "type": "tool_use",
        "id": "toolu_01A09q90qw90lq917835lk",
        "name": "web_search",
        "input": {"query": "Python async programming"}
    }
    """
    id: str = Field(description="Unique tool call ID from Claude")
    name: str = Field(description="Name of the tool to execute")
    input: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments to pass to the tool"
    )

    def __str__(self) -> str:
        """Human-readable representation."""
        return f"ToolCall({self.name}, args={self.input})"


class LLMResponse(BaseModel):
    """
    Standardized response from LLM API calls.

    🎓 TEACHING: Response Structure
    ===============================
    Claude's raw API response is complex with nested content blocks.
    We flatten it into a predictable structure:

    Raw API response:
    {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Hello!"},
            {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
        ],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 50}
    }

    Our simplified response:
    LLMResponse(
        content="Hello!",
        tool_calls=[ToolCallRequest(...)],
        raw_response={...}  # Keep original for debugging
    )
    """
    content: str = Field(default="", description="Text content from response")
    tool_calls: list[ToolCallRequest] = Field(
        default_factory=list,
        description="Tool calls requested by Claude"
    )
    model: str = Field(default="", description="Model that generated response")
    stop_reason: str | None = Field(
        default=None,
        description="Why generation stopped: end_turn, tool_use, max_tokens"
    )
    input_tokens: int = Field(default=0, description="Tokens in the prompt")
    output_tokens: int = Field(default=0, description="Tokens in response")
    raw_response: dict[str, Any] = Field(
        default_factory=dict,
        description="Original API response for debugging"
    )

    @property
    def total_tokens(self) -> int:
        """Total tokens used in this request."""
        return self.input_tokens + self.output_tokens

    @property
    def has_tool_calls(self) -> bool:
        """Check if Claude wants to use tools."""
        return len(self.tool_calls) > 0

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary format expected by agent.py.

        The agent expects: {"content": [{"type": "text", "text": "..."}]}
        """
        content_blocks = []

        if self.content:
            content_blocks.append({"type": "text", "text": self.content})

        for tool_call in self.tool_calls:
            content_blocks.append({
                "type": "tool_use",
                "id": tool_call.id,
                "name": tool_call.name,
                "input": tool_call.input,
            })

        return {"content": content_blocks}


@dataclass
class StreamChunk:
    """
    A single chunk from a streaming response.

    🎓 TEACHING: Streaming vs Non-Streaming
    =======================================
    Non-streaming: Wait for entire response, then process
        - Simpler code
        - Higher latency (user waits)
        - Lower memory (one response object)

    Streaming: Process chunks as they arrive
        - Complex code (handle partial data)
        - Lower perceived latency (user sees progress)
        - Higher memory (accumulating chunks)

    Compare to Go:
        // Go uses channels for streaming
        func (c *Client) Stream(ctx context.Context) <-chan Chunk {
            ch := make(chan Chunk)
            go func() {
                defer close(ch)
                for chunk := range response.Chunks() {
                    ch <- chunk
                }
            }()
            return ch
        }

        # Python uses generators/async iterators
        async def stream(self) -> AsyncIterator[StreamChunk]:
            async for chunk in response:
                yield StreamChunk(...)

    Key Python concepts:
    - yield: Produces value without ending function
    - AsyncIterator: Async version of iterator protocol
    - async for: Iterate over async iterators
    """
    type: str  # "text", "tool_use", "message_start", "message_delta", "message_stop"
    text: str = ""  # Text content (for type="text")
    tool_call: ToolCallRequest | None = None  # Tool call (for type="tool_use")

    # Metadata
    is_complete: bool = False  # Is this the final chunk?
    input_tokens: int = 0
    output_tokens: int = 0


# =============================================================================
# 🎓 MESSAGE BUILDER - Fluent Interface Pattern
# =============================================================================
"""
🎓 TEACHING: Fluent Interface / Builder Pattern
===============================================
The Builder pattern lets you construct complex objects step-by-step.
A "fluent" interface returns `self` so you can chain method calls.

Compare to Go:
    // Go doesn't have a standard builder pattern
    // Often uses functional options
    client := NewClient(
        WithAPIKey("key"),
        WithTimeout(30),
    )

    # Python fluent builder
    message = (MessageBuilder()
        .system("You are helpful")
        .user("Hello")
        .assistant("Hi there!")
        .user("How are you?")
        .build())

Benefits:
- Readable, self-documenting code
- Enforces required fields
- IDE autocomplete for valid operations
"""


class MessageBuilder:
    """
    Fluent builder for constructing message lists.

    🎓 TEACHING: The Anthropic Message Format
    =========================================
    Claude's API expects messages in this format:

    messages=[
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
        {"role": "user", "content": "How are you?"}
    ]

    Rules:
    1. Messages alternate between user and assistant
    2. First message must be from user
    3. System prompt is passed separately (not in messages)
    4. Content can be string OR list of content blocks

    Content blocks (for multimodal):
    [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image", "source": {"type": "base64", "data": "..."}}
    ]

    Usage:
        builder = MessageBuilder()
        messages = (builder
            .user("What is Python?")
            .assistant("Python is a programming language...")
            .user("Tell me more")
            .build())
    """

    def __init__(self):
        """Initialize empty message list."""
        self._messages: list[dict[str, Any]] = []
        self._system: str | None = None

    def system(self, content: str) -> "MessageBuilder":
        """
        Set the system prompt.

        🎓 NOTE: System prompt is NOT added to messages list.
        It's passed separately to the API.
        """
        self._system = content
        return self

    def user(self, content: str) -> "MessageBuilder":
        """Add a user message."""
        self._messages.append({
            "role": "user",
            "content": content
        })
        return self

    def assistant(self, content: str) -> "MessageBuilder":
        """Add an assistant message."""
        self._messages.append({
            "role": "assistant",
            "content": content
        })
        return self

    def tool_result(self, tool_use_id: str, content: str) -> "MessageBuilder":
        """
        Add a tool result message.

        🎓 TEACHING: Tool Result Flow
        =============================
        After Claude requests a tool, you must send back the result:

        1. Claude: {"type": "tool_use", "id": "abc123", "name": "search", ...}
        2. You execute the tool
        3. You send: {"role": "user", "content": [
               {"type": "tool_result", "tool_use_id": "abc123", "content": "..."}
           ]}
        4. Claude continues with the result
        """
        self._messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": content,
            }]
        })
        return self

    def add_message(self, message: Message) -> "MessageBuilder":
        """Add a Message object from our models."""
        role_str = message.role.value if isinstance(message.role, Role) else message.role
        self._messages.append({
            "role": role_str,
            "content": message.content
        })
        return self

    def add_messages(self, messages: list[Message]) -> "MessageBuilder":
        """Add multiple Message objects."""
        for msg in messages:
            self.add_message(msg)
        return self

    def build(self) -> list[dict[str, Any]]:
        """
        Build and return the message list.

        Returns:
            List of message dictionaries ready for API
        """
        return self._messages.copy()

    def get_system(self) -> str | None:
        """Get the system prompt if set."""
        return self._system

    def clear(self) -> "MessageBuilder":
        """Clear all messages (keep system prompt)."""
        self._messages = []
        return self

    def __len__(self) -> int:
        """Number of messages."""
        return len(self._messages)


# =============================================================================
# 🎓 SYNCHRONOUS LLM CLIENT
# =============================================================================
"""
🎓 TEACHING: Sync vs Async - When to Use Which?
===============================================
Synchronous (blocking):
    - Simpler code, easier to debug
    - Good for: CLI tools, scripts, simple applications
    - Problem: Blocks the thread while waiting for API

Asynchronous (non-blocking):
    - More complex, uses async/await
    - Good for: Web servers, handling many concurrent requests
    - Benefit: Can do other work while waiting for API

Compare to Go:
    // Go has goroutines - lightweight threads
    go func() {
        response := client.Complete(ctx, messages)
        results <- response
    }()

    # Python async uses event loop
    async def main():
        response = await client.complete(messages)

Rule of thumb:
- Building a CLI? Use sync client
- Building a web server? Use async client
- Handling multiple LLM calls? Use async client
"""


class LLMClient:
    """
    Synchronous client for Claude API.

    🎓 TEACHING: Class Design Principles
    ====================================
    This class follows several important principles:

    1. SINGLE RESPONSIBILITY - Only handles LLM communication
    2. DEPENDENCY INJECTION - API key can be injected or from env
    3. SENSIBLE DEFAULTS - Works out of the box with env vars
    4. IMMUTABLE CONFIG - Settings fixed at construction time

    Compare to Go:
        type LLMClient struct {
            client   *anthropic.Client  // unexported = private
            model    string
            maxToks  int
        }

        func NewLLMClient(opts ...Option) *LLMClient {
            // Apply options, validate, return
        }

    Python uses:
        - Leading underscore for "private" (_client)
        - __init__ for construction
        - Default parameter values instead of options pattern

    Usage:
        # With environment variable ANTHROPIC_API_KEY
        client = LLMClient()

        # With explicit API key
        client = LLMClient(api_key="sk-ant-...")

        # With custom settings
        client = LLMClient(
            api_key="sk-ant-...",
            model="claude-opus-4-20250514",
            max_tokens=8192,
        )
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        """
        Initialize the LLM client.

        🎓 TEACHING: Environment Variables
        ==================================
        os.getenv("KEY") returns None if not found
        os.getenv("KEY", "default") returns "default" if not found
        os.environ["KEY"] raises KeyError if not found

        We use `or` to fall back to env var:
            api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

        This works because:
            None or "value" → "value"
            "explicit" or "value" → "explicit"

        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            model: Model to use (default: claude-sonnet-4-20250514)
            max_tokens: Maximum tokens in response (default: 4096)
            temperature: Randomness 0-1 (default: 0.7)
        """
        # Get API key from parameter or environment
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

        if not self._api_key:
            raise ValueError(
                "API key required. Either pass api_key parameter or "
                "set ANTHROPIC_API_KEY environment variable."
            )

        # Store configuration
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

        # Initialize the Anthropic client
        # The SDK handles auth headers, retries, etc.
        self._client = Anthropic(api_key=self._api_key)

    def complete(
        self,
        messages: list[Message] | list[dict[str, Any]],
        system: str | None = None,
        tools: list[ToolDefinition] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """
        Send messages to Claude and get a response.

        🎓 TEACHING: Method Parameter Design
        ====================================
        Notice how parameters can override instance defaults:

            # Use instance default model
            response = client.complete(messages)

            # Override model for this call
            response = client.complete(messages, model="claude-opus-4-20250514")

        Pattern: param or self._param
            max_tokens = max_tokens or self._max_tokens

        This allows:
        - Setting defaults at construction time
        - Overriding per-call when needed
        - None means "use default"

        Args:
            messages: Conversation history (our Message objects or dicts)
            system: System prompt (optional)
            tools: Available tools (optional)
            model: Override default model (optional)
            max_tokens: Override default max tokens (optional)
            temperature: Override default temperature (optional)

        Returns:
            Dictionary with response content (for compatibility with agent.py)
            Format: {"content": [{"type": "text", "text": "..."}]}
        """
        # Convert our Message objects to API format
        formatted_messages = self._format_messages(messages)

        # Convert tools to Anthropic format
        formatted_tools = None
        if tools:
            formatted_tools = [tool.to_anthropic_schema() for tool in tools]

        # Build API call parameters
        # 🎓 NOTE: **kwargs unpacking would be cleaner but less explicit
        api_params: dict[str, Any] = {
            "model": model or self._model,
            "max_tokens": max_tokens or self._max_tokens,
            "messages": formatted_messages,
        }

        # Only include optional parameters if provided
        # 🎓 TEACHING: API Design - Don't send None values
        # Some APIs treat missing vs null differently
        if system:
            api_params["system"] = system

        if formatted_tools:
            api_params["tools"] = formatted_tools

        # Temperature is special - 0 is valid, so check for None explicitly
        temp = temperature if temperature is not None else self._temperature
        if temp is not None:
            api_params["temperature"] = temp

        # Make the API call
        # 🎓 TEACHING: Exception Handling
        # We let exceptions propagate - the caller should handle them
        # Common exceptions:
        # - anthropic.APIConnectionError: Network issues
        # - anthropic.RateLimitError: Too many requests
        # - anthropic.APIStatusError: Invalid request
        response = self._client.messages.create(**api_params)

        # Parse and return response
        return self._parse_response(response).to_dict()

    def complete_structured(
        self,
        messages: list[Message] | list[dict[str, Any]],
        system: str | None = None,
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Complete with structured LLMResponse return type.

        🎓 TEACHING: Method Overloading in Python
        =========================================
        Python doesn't have true method overloading like Java/C++.
        Instead, we use different method names or **kwargs.

        Options:
        1. Different methods: complete() vs complete_structured()
        2. Union return type: -> dict | LLMResponse
        3. Parameter flag: complete(structured=True)

        We chose option 1 for clarity and type safety.

        Returns:
            LLMResponse object with parsed content
        """
        formatted_messages = self._format_messages(messages)
        formatted_tools = [t.to_anthropic_schema() for t in tools] if tools else None

        api_params: dict[str, Any] = {
            "model": kwargs.get("model", self._model),
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
            "messages": formatted_messages,
        }

        if system:
            api_params["system"] = system
        if formatted_tools:
            api_params["tools"] = formatted_tools

        response = self._client.messages.create(**api_params)
        return self._parse_response(response)

    def stream(
        self,
        messages: list[Message] | list[dict[str, Any]],
        system: str | None = None,
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        """
        Stream responses from Claude.

        🎓 TEACHING: Generators in Python
        =================================
        A generator is a function that uses 'yield' instead of 'return'.
        It produces values one at a time, pausing between yields.

        Compare to Go:
            // Go uses channels for streaming
            func (c *Client) Stream() <-chan Chunk {
                ch := make(chan Chunk)
                go func() {
                    defer close(ch)
                    for chunk := range stream {
                        ch <- chunk
                    }
                }()
                return ch
            }

            # Python uses generators
            def stream(self) -> Iterator[Chunk]:
                for chunk in stream:
                    yield chunk

        Usage:
            for chunk in client.stream(messages):
                print(chunk.text, end="", flush=True)

        Benefits:
        - Memory efficient (one chunk at a time)
        - Lower perceived latency (user sees progress)
        - Can stop early (break from loop)

        Yields:
            StreamChunk objects with partial content
        """
        formatted_messages = self._format_messages(messages)
        formatted_tools = [t.to_anthropic_schema() for t in tools] if tools else None

        api_params: dict[str, Any] = {
            "model": kwargs.get("model", self._model),
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
            "messages": formatted_messages,
        }

        if system:
            api_params["system"] = system
        if formatted_tools:
            api_params["tools"] = formatted_tools

        # Use streaming API
        with self._client.messages.stream(**api_params) as stream:
            for event in stream:
                chunk = self._parse_stream_event(event)
                if chunk:
                    yield chunk

    def _format_messages(
        self,
        messages: list[Message] | list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Convert messages to Anthropic API format.

        🎓 TEACHING: Duck Typing
        ========================
        "If it walks like a duck and quacks like a duck, it's a duck."

        We check for attributes rather than types:
            if hasattr(msg, "role") → It's a Message-like object
            else → It's already a dict

        This makes the method flexible - it accepts:
        - List of our Message objects
        - List of dicts (already formatted)
        - Mixed lists

        Compare to Go:
            // Go uses interfaces for this
            type MessageLike interface {
                GetRole() string
                GetContent() string
            }
        """
        formatted = []

        for msg in messages:
            if hasattr(msg, "role") and hasattr(msg, "content"):
                # It's a Message object
                role_str = msg.role.value if isinstance(msg.role, Role) else str(msg.role)
                # Fix the typo in Role enum ("assitant" -> "assistant")
                if role_str == "assitant":
                    role_str = "assistant"
                formatted.append({
                    "role": role_str,
                    "content": msg.content
                })
            else:
                # Already a dict
                formatted.append(msg)

        return formatted

    def _parse_response(self, response: AnthropicMessage) -> LLMResponse:
        """
        Parse Anthropic response into our LLMResponse model.

        🎓 TEACHING: Defensive Parsing
        ==============================
        When parsing external data (API responses), be defensive:

        1. Use .get() with defaults for dict access
        2. Check types before operations
        3. Handle missing/unexpected fields gracefully

        Example:
            # Bad - raises KeyError if missing
            text = response["content"]["text"]

            # Good - returns empty string if missing
            text = response.get("content", {}).get("text", "")
        """
        text_content = ""
        tool_calls: list[ToolCallRequest] = []

        # Extract content from response
        for block in response.content:
            if block.type == "text":
                text_content += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCallRequest(
                    id=block.id,
                    name=block.name,
                    input=block.input if isinstance(block.input, dict) else {},
                ))

        return LLMResponse(
            content=text_content,
            tool_calls=tool_calls,
            model=response.model,
            stop_reason=response.stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            raw_response=response.model_dump(),
        )

    def _parse_stream_event(self, event: Any) -> StreamChunk | None:
        """
        Parse a streaming event into a StreamChunk.

        🎓 TEACHING: Event Types in Claude Streaming
        ============================================
        Claude streaming sends different event types:

        1. message_start - Initial message metadata
        2. content_block_start - Beginning of a content block
        3. content_block_delta - Incremental content (the actual text!)
        4. content_block_stop - End of a content block
        5. message_delta - Final message metadata (stop reason, tokens)
        6. message_stop - Stream complete

        Most useful is content_block_delta which contains the text.
        """
        event_type = getattr(event, "type", None)

        if event_type == "content_block_delta":
            delta = getattr(event, "delta", None)
            if delta and getattr(delta, "type", None) == "text_delta":
                return StreamChunk(
                    type="text",
                    text=getattr(delta, "text", ""),
                )
        elif event_type == "message_delta":
            usage = getattr(event, "usage", None)
            return StreamChunk(
                type="message_delta",
                is_complete=True,
                output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
            )
        elif event_type == "message_stop":
            return StreamChunk(type="message_stop", is_complete=True)

        return None


# =============================================================================
# 🎓 ASYNCHRONOUS LLM CLIENT
# =============================================================================
"""
🎓 TEACHING: Async/Await in Python
==================================
Python's async/await is similar to JavaScript's Promise-based async.

Key concepts:
1. async def - Defines a coroutine (async function)
2. await - Pause until async operation completes
3. asyncio - The event loop that runs coroutines

Compare to Go:
    // Go uses goroutines (concurrent, parallel)
    go func() {
        result := <-channel  // Block until result
    }()

    # Python uses async/await (concurrent, usually not parallel)
    async def main():
        result = await some_async_function()

Key differences from Go:
- Python async is single-threaded (uses event loop)
- Go goroutines are multi-threaded (true parallelism)
- Python async is for I/O-bound tasks (network, file)
- Go goroutines work for both I/O and CPU-bound tasks

When to use async in Python:
- Web servers handling many requests
- Making multiple API calls concurrently
- Any I/O-heavy operations

When NOT to use async:
- CPU-intensive computations (use multiprocessing)
- Simple scripts (adds complexity)
- When sync code is clearer
"""


class AsyncLLMClient:
    """
    Asynchronous client for Claude API.

    🎓 TEACHING: Async Client Design
    ================================
    The async client mirrors the sync client but uses:
    - AsyncAnthropic instead of Anthropic
    - async def instead of def
    - await for API calls
    - AsyncIterator for streaming

    This duplication (sync + async) is common in Python libraries.
    Some libraries use code generation to avoid it.

    Usage:
        async def main():
            client = AsyncLLMClient()
            response = await client.complete(messages)

            # Or with streaming
            async for chunk in client.stream(messages):
                print(chunk.text, end="")

        # Run the async function
        asyncio.run(main())
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        """
        Initialize the async LLM client.

        🎓 NOTE: __init__ cannot be async!
        ==================================
        Python's __init__ must be synchronous. If you need async setup:

        Option 1: Factory function (recommended)
            async def create_client():
                client = AsyncLLMClient()
                await client.initialize()  # Async setup
                return client

        Option 2: Lazy initialization
            async def complete(self):
                if not self._initialized:
                    await self._initialize()
                ...

        For this client, we don't need async init since
        the Anthropic SDK handles connection pooling lazily.
        """
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

        if not self._api_key:
            raise ValueError(
                "API key required. Either pass api_key parameter or "
                "set ANTHROPIC_API_KEY environment variable."
            )

        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

        # Use AsyncAnthropic for async operations
        self._client = AsyncAnthropic(api_key=self._api_key)

    async def complete(
        self,
        messages: list[Message] | list[dict[str, Any]],
        system: str | None = None,
        tools: list[ToolDefinition] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """
        Send messages to Claude asynchronously.

        🎓 TEACHING: Async Method Signature
        ===================================
        Notice: async def complete(...) -> dict[str, Any]

        The return type is dict, not Awaitable[dict]!

        When you call an async function, you get a coroutine:
            coro = client.complete(messages)  # Returns coroutine
            result = await coro               # Returns dict

        The type hint shows the final value type (dict),
        not the intermediate coroutine type.

        Returns:
            Dictionary with response content
        """
        formatted_messages = self._format_messages(messages)
        formatted_tools = [t.to_anthropic_schema() for t in tools] if tools else None

        api_params: dict[str, Any] = {
            "model": model or self._model,
            "max_tokens": max_tokens or self._max_tokens,
            "messages": formatted_messages,
        }

        if system:
            api_params["system"] = system
        if formatted_tools:
            api_params["tools"] = formatted_tools

        temp = temperature if temperature is not None else self._temperature
        if temp is not None:
            api_params["temperature"] = temp

        # await the API call - this is the key difference from sync
        response = await self._client.messages.create(**api_params)

        return self._parse_response(response).to_dict()

    async def complete_structured(
        self,
        messages: list[Message] | list[dict[str, Any]],
        system: str | None = None,
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Complete with structured LLMResponse return type.

        Returns:
            LLMResponse object with parsed content
        """
        formatted_messages = self._format_messages(messages)
        formatted_tools = [t.to_anthropic_schema() for t in tools] if tools else None

        api_params: dict[str, Any] = {
            "model": kwargs.get("model", self._model),
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
            "messages": formatted_messages,
        }

        if system:
            api_params["system"] = system
        if formatted_tools:
            api_params["tools"] = formatted_tools

        response = await self._client.messages.create(**api_params)
        return self._parse_response(response)

    async def stream(
        self,
        messages: list[Message] | list[dict[str, Any]],
        system: str | None = None,
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream responses from Claude asynchronously.

        🎓 TEACHING: Async Generators
        =============================
        An async generator uses both 'async def' and 'yield':

            async def stream() -> AsyncIterator[Chunk]:
                async for event in api_stream:
                    yield Chunk(event)

        Consume with 'async for':
            async for chunk in client.stream(messages):
                print(chunk.text)

        Compare to sync generator:
            # Sync
            for chunk in client.stream(messages):
                print(chunk.text)

            # Async
            async for chunk in client.stream(messages):
                print(chunk.text)

        The only difference is 'async for' instead of 'for'.

        Yields:
            StreamChunk objects with partial content
        """
        formatted_messages = self._format_messages(messages)
        formatted_tools = [t.to_anthropic_schema() for t in tools] if tools else None

        api_params: dict[str, Any] = {
            "model": kwargs.get("model", self._model),
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
            "messages": formatted_messages,
        }

        if system:
            api_params["system"] = system
        if formatted_tools:
            api_params["tools"] = formatted_tools

        # Use async streaming API
        async with self._client.messages.stream(**api_params) as stream:
            async for event in stream:
                chunk = self._parse_stream_event(event)
                if chunk:
                    yield chunk

    def _format_messages(
        self,
        messages: list[Message] | list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Convert messages to Anthropic API format.

        🎓 NOTE: This is NOT async
        ==========================
        Not every method in an async class needs to be async!

        Use async only for I/O operations (network, file, database).
        Pure computation should stay synchronous.

        This method just transforms data in memory - no I/O needed.
        """
        formatted = []

        for msg in messages:
            if hasattr(msg, "role") and hasattr(msg, "content"):
                role_str = msg.role.value if isinstance(msg.role, Role) else str(msg.role)
                if role_str == "assitant":
                    role_str = "assistant"
                formatted.append({
                    "role": role_str,
                    "content": msg.content
                })
            else:
                formatted.append(msg)

        return formatted

    def _parse_response(self, response: AnthropicMessage) -> LLMResponse:
        """Parse Anthropic response into our LLMResponse model."""
        text_content = ""
        tool_calls: list[ToolCallRequest] = []

        for block in response.content:
            if block.type == "text":
                text_content += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCallRequest(
                    id=block.id,
                    name=block.name,
                    input=block.input if isinstance(block.input, dict) else {},
                ))

        return LLMResponse(
            content=text_content,
            tool_calls=tool_calls,
            model=response.model,
            stop_reason=response.stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            raw_response=response.model_dump(),
        )

    def _parse_stream_event(self, event: Any) -> StreamChunk | None:
        """Parse a streaming event into a StreamChunk.
        Streaming Concept:
        ==================================================================
        Instead of waiting for the full response, streaming gives you pieces as they're generated:


        ┌─────────┐     ┌─────────┐     ┌─────────┐
        │ "Hello" │ --> │ " world"│ --> │  [DONE] │
        └─────────┘     └─────────┘     └─────────┘
           Event 1        Event 2         Event 3
        Code Walkthrough

        event_type = getattr(event, "type", None)  # Get event.type safely
        Then handle 3 types of events:

        Event Type	        What It Means	            What You Do
        content_block_delta	New text chunk arrived	    Extract the text
        message_delta	    Metadata update (token count)	Note completion status
        message_stop	    Stream finished	            Mark as complete

        """
        event_type = getattr(event, "type", None)

        if event_type == "content_block_delta":
            delta = getattr(event, "delta", None)
            if delta and getattr(delta, "type", None) == "text_delta":
                return StreamChunk(
                    type="text",
                    text=getattr(delta, "text", ""),
                )
        elif event_type == "message_delta":
            usage = getattr(event, "usage", None)
            return StreamChunk(
                type="message_delta",
                is_complete=True,
                output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
            )
        elif event_type == "message_stop":
            return StreamChunk(type="message_stop", is_complete=True)

        return None


# =============================================================================
# 🎓 CONVENIENCE ALIASES
# =============================================================================
"""
🎓 TEACHING: Type Aliases for Flexibility
=========================================
We create aliases so the codebase can use generic names:

    from research_agent.llm.client import LLMClient, AsyncLLMClient

These could point to different implementations:
- AnthropicClient (Claude)
- OpenAIClient (GPT-4)
- LocalClient (Ollama)

This decouples the agent from the specific LLM provider.
"""

# These aliases are used by agent.py
# If we add support for other LLMs, we just change what these point to
AnthropicClient = LLMClient
AsyncAnthropicClient = AsyncLLMClient


# =============================================================================
# 🎓 MODULE TEST / EXAMPLE USAGE
# =============================================================================
"""
🎓 TEACHING: if __name__ == "__main__"
======================================
This block runs only when the file is executed directly:
    python client.py  # Runs the block

Not when imported:
    from research_agent.llm.client import LLMClient  # Skips the block

Compare to Go:
    // Go checks package name
    func main() {  // Only runs if package main
        ...
    }

Use this for:
- Quick testing during development
- Example usage documentation
- Sanity checks
"""

if __name__ == "__main__":
    import asyncio

    print("=" * 60)
    print("LLM Client Module - Example Usage")
    print("=" * 60)

    # Check if API key is available
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n⚠️  ANTHROPIC_API_KEY not set in environment")
        print("Set it to test the client:")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        print("\nShowing structure examples instead:\n")

    # Example: MessageBuilder
    print("📝 MessageBuilder Example:")
    builder = MessageBuilder()
    messages = (builder
        .system("You are a helpful assistant")
        .user("What is Python?")
        .assistant("Python is a programming language...")
        .user("Tell me more about async")
        .build())

    print(f"  Built {len(messages)} messages:")
    for msg in messages:
        preview = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
        print(f"    [{msg['role']}]: {preview}")

    # Example: ToolCallRequest
    print("\n🔧 ToolCallRequest Example:")
    tool_call = ToolCallRequest(
        id="toolu_01ABC",
        name="web_search",
        input={"query": "Python async programming"}
    )
    print(f"  {tool_call}")
    print(f"  JSON: {tool_call.model_dump_json()}")

    # Example: LLMResponse
    print("\n📨 LLMResponse Example:")
    response = LLMResponse(
        content="Here are the search results...",
        tool_calls=[tool_call],
        model="claude-sonnet-4-20250514",
        stop_reason="end_turn",
        input_tokens=150,
        output_tokens=300,
    )
    print(f"  Content: {response.content[:50]}...")
    print(f"  Has tool calls: {response.has_tool_calls}")
    print(f"  Total tokens: {response.total_tokens}")

    # Example: Sync client (if API key available)
    if api_key:
        print("\n🤖 Testing Sync Client:")
        try:
            client = LLMClient()
            from research_agent.core.models import Message, Role

            test_messages = [
                Message(role=Role.USER, content="Say 'Hello, World!' and nothing else.")
            ]
            result = client.complete(test_messages)
            print(f"  Response: {result}")
        except Exception as e:
            print(f"  Error: {e}")

        # Example: Async client
        print("\n🚀 Testing Async Client:")

        async def test_async():
            client = AsyncLLMClient()
            test_messages = [
                Message(role=Role.USER, content="Say 'Async Hello!' and nothing else.")
            ]
            result = await client.complete(test_messages)
            print(f"  Response: {result}")

        try:
            asyncio.run(test_async())
        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "=" * 60)
    print("✅ Client module loaded successfully!")
    print("=" * 60)
