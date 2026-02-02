"""
LLM client module.

🎓 TEACHING: __init__.py Files
==============================
The __init__.py file makes a directory a Python "package".
It controls what gets exported when you import from the package.

Example:
    # Without __init__.py exports:
    from research_agent.llm.client import LLMClient  # Need full path

    # With __init__.py exports:
    from research_agent.llm import LLMClient  # Cleaner!

The __all__ list controls what's exported with `from package import *`
(though we rarely use * imports in practice).

Compare to Go:
    // Go uses package-level exports (capitalized names)
    package llm

    type LLMClient struct { ... }  // Exported (capital L)
    type helper struct { ... }     // Not exported (lowercase)

    # Python uses __all__ and __init__.py
    __all__ = ["LLMClient"]  # Explicitly exported
"""

from research_agent.llm.client import (
    # Main clients
    LLMClient,
    AsyncLLMClient,

    # Aliases for explicit Anthropic naming
    AnthropicClient,
    AsyncAnthropicClient,

    # Response models
    LLMResponse,
    ToolCallRequest,
    StreamChunk,

    # Utilities
    MessageBuilder,

    # Constants
    DEFAULT_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    CLAUDE_MODELS,
)

__all__ = [
    # Main clients (generic names)
    "LLMClient",
    "AsyncLLMClient",

    # Anthropic-specific aliases
    "AnthropicClient",
    "AsyncAnthropicClient",

    # Response models
    "LLMResponse",
    "ToolCallRequest",
    "StreamChunk",

    # Utilities
    "MessageBuilder",

    # Constants
    "DEFAULT_MODEL",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
    "CLAUDE_MODELS",
]
