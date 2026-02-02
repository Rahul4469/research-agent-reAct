"""Anthropic Claude client wrapper for the Research Agent."""

import os 
from typing import Any, AsyncIterator, Iterator
from dataclasses import dataclass, field

from anthropic import AsyncAnthropic, Anthropic
from anthropic.types import Message as AnthropicMessage, ContentBlock

from pydantic import BaseModel, Field
from dotenv import load_dotenv

from research_agent.core.models import Message, Role, ToolDefinition

load_dotenv()

DEFAULT_MODEL = "claude-sonnet-4-20250514"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.7

# Available Claude models (as of 2024)
CLAUDE_MODELS = {
    "sonnet": "claude-sonnet-4-20250514",     # Fast, good balance
    "opus": "claude-opus-4-20250514",          # Most capable
    "haiku": "claude-3-5-haiku-20241022",      # Fastest, cheapest
}

class ToolCallRequest(BaseModel):
    """ tool call requested by the LLM.
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
        description="Argumen to pass to the tool"
    )
    
    def __str__(self) -> str:
        """Human-readable representation"""
        return f"ToolCall({self.name}, args={self.input})"
    
class LLMResponse(BaseModel):
    """
    Standardized response from LLM API calls.
    Claude's raw API response is complex with nested content blocks.
    We flatten it into a predictable structure
    ---
    content, tool_calls, model, stop_reason, input_tokens, outpu_tokens, raw_reponse
    """
    content: str = Field(default="", description="Text content from response")
    tool_calls: list[ToolCallRequest] = Field(
        default_factory=list,
        description="Tool calls requested by Claude"
    )
    model: str =Field(default="", description="Model that generated response")
    stop_reason: str | None = Field(default=None,
                                    description="Why generation stopped: end_turn, tool_use, max_tokens")
    input_tokens: int = Field(default=0, description="Token in the prompt")
    output_tokens: int = Field(default=0, description="Token in response")
    raw_response: dict[str, Any] = Field(
        default_factory=dict,
        description="Original API reponse for debugging"
    )
    
    @property
    def total_tokens(self) -> int:
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
    type: str
    text: str =""
    tool_call: ToolCallRequest | None = None
        
        # Metadata
    is_complete: bool = False  # Is this the final chunk?
    input_tokens: int = 0
    output_tokens: int = 0
        
# MESSAGE BUILDER - Fluent Interface Pattern ///--------------------------

class MessageBuilder:
    """ 
    Fluent builder for constructing message lists.
    """
    def __init__(self):
        """Initialize empty message list."""
        self._messages: list[dict[str, Any]] = []
        self._system: str | None = None
    
    def system(self, content: str) -> "MessageBuilder":
        """
        Set the system prompt.
        NOTE: System prompt is NOT added to messages list.
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
        """Add a tool result message"""
        self._messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": content
            }]
        })
        return self

    def add_message(self, message: Message) -> "MessageBuilder":
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
        """Build and return the message list. message dict for API"""
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

# SYNCHRONOUS LLM CLIENT ///-----------------------------------

class LLMClient:
    def __init__(self,
                 api_key: str | None = None,
                 model: str = DEFAULT_MODEL,
                 max_tokens: int = DEFAULT_MAX_TOKENS,
                 temperature: float = DEFAULT_TEMPERATURE):
        # Get API key from parameter or environment
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        
        if not self._api_key:
            raise ValueError(
                "API key required. Either pass api_key parameter or "
                "set ANTHROPIC_API_KEY environment variable."
            ) 
            
        # Store configs
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        
        # Initialize the Anthropic client
        # The SDK handles auth headers, retries, etc.
        self._client = Anthropic(api_key=self._api_key) 
    
    def complete(self, message: list[Message] | list[dict[str, Any]],
                system: str | None = None,
                tools: list[ToolDefinition] | None = None,
                model: str | None = None,
                max_tokens: int | None = None,
                temperature: float | None = None,) -> dict[str, Any]:
        """Send messages to Claude and get a response. 
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
        formatted_messages = self._format_messages(message) 
        
        # Convert tools to Anthropic format
        formatted_tools = None
        if tools:
            formatted_tools = [tool.to_anthropic_schema() for tool in tools]
            
        # Build API call parameters
        #NOTE: **kwargs unpacking would be cleaner but less explicit
        api_params: dict[str, Any] = {
            "model": model or self._model,
            "max_tokens": max_tokens or self._max_tokens,
            "messages": formatted_messages,
        }
        
        # Only include optional parameters if provided
        # API Design - Don't send None values
        # Some APIs treat missing vs null differently
        if system:
            api_params["system"] = system

        if formatted_tools:
            api_params["tools"] = formatted_tools

        # Temperature is special - 0 is valid, so check for None explicitly
        temp = temperature if temperature is not None else self._temperature
        if temp is not None:
            api_params["temperature"] = temp   
            
        # MAKE THE API CALL
        response = self._client.messages.create(**api_params)   
        
        # Parse and return response
        return self._parse_response(response).to_dict()                    
    
    def complete_structured(self,
                            message: list[Message] | list[dict[str, Any]],
                            system: str | None = None,
                            tools: list[ToolDefinition] | None = None,
                            **kwargs: Any,
                            ) -> LLMResponse:
        formatted_messages = self._format_messages(message)
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
    
    def stream(self,
               messages: list[Message] | list[dict[str, Any]],
               system: str | None = None,
               tools: list[ToolDefinition] | None = None,
               **kwargs: Any) -> Iterator[StreamChunk]:
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
                    
    def _format_messages(self,
                        messages: list[Message] | list[dict[str, Any]]
                        ) -> list[dict[str, Any]]:
        """Convert messages to Anthropic API format."""
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
        """ Parse Anthropic response into our LLMResponse model."""
        text_content = ""
        tool_calls: list[ToolCallRequest] =[]
        
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
        """Parse a streaming event into a StreamChunk."""
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

# ASYNC LLM CLIENT ///-----------------------------------

class AsyncLLMClient:
    """Async version of LLMClient for use with async/await."""

    def __init__(self,
                 api_key: str | None = None,
                 model: str = DEFAULT_MODEL,
                 max_tokens: int = DEFAULT_MAX_TOKENS,
                 temperature: float = DEFAULT_TEMPERATURE):
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

        if not self._api_key:
            raise ValueError(
                "API key required. Either pass api_key parameter or "
                "set ANTHROPIC_API_KEY environment variable."
            )

        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = AsyncAnthropic(api_key=self._api_key)

    async def complete(self,
                       messages: list[Message] | list[dict[str, Any]],
                       system: str | None = None,
                       tools: list[ToolDefinition] | None = None,
                       model: str | None = None,
                       max_tokens: int | None = None,
                       temperature: float | None = None) -> dict[str, Any]:
        """Send messages to Claude asynchronously."""
        formatted_messages = self._format_messages(messages)
        formatted_tools = [tool.to_anthropic_schema() for tool in tools] if tools else None

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

        response = await self._client.messages.create(**api_params)
        return self._parse_response(response).to_dict()

    async def stream(self,
                     messages: list[Message] | list[dict[str, Any]],
                     system: str | None = None,
                     tools: list[ToolDefinition] | None = None,
                     **kwargs: Any) -> AsyncIterator[StreamChunk]:
        """Stream responses asynchronously."""
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

        async with self._client.messages.stream(**api_params) as stream:
            async for event in stream:
                chunk = self._parse_stream_event(event)
                if chunk:
                    yield chunk

    def _format_messages(self,
                         messages: list[Message] | list[dict[str, Any]]
                         ) -> list[dict[str, Any]]:
        """Convert messages to Anthropic API format."""
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
        """Parse Anthropic response into LLMResponse."""
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
        """Parse a streaming event into a StreamChunk."""
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

# CONVENIENCE ALIASES ////-----------------------------------
"""This decouples the agent from the specific LLM provider."""

# These aliases are used by agent.py
# If we add support for other LLMs, we just change what these point to
AnthropicClient = LLMClient
AsyncAnthropicClient = AsyncLLMClient 

if __name__ == "__main__":
    import asyncio

    print("LLM Client Module - Example Usage")
    

    # Check if API key is available
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n ANTHROPIC_API_KEY not set in environment")
        print("Set it to test the client:")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        print("\nShowing structure examples instead:\n")

    # Example: MessageBuilder
    print("MessageBuilder Example:")
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
    print("\nToolCallRequest Example:")
    tool_call = ToolCallRequest(
        id="toolu_01ABC",
        name="web_search",
        input={"query": "Python async programming"}
    )
    print(f"  {tool_call}")
    print(f"  JSON: {tool_call.model_dump_json()}")

    # Example: LLMResponse
    print("\nLLMResponse Example:")
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
        print("\nTesting Sync Client:")
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
        print("\nTesting Async Client:")

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

    print("Client module loaded successfully!")     