"""
Utility functions and helpers for the research agent.

🎓 TEACHING: Utils Package Structure
====================================
This package contains general-purpose utilities:

    utils/
    ├── __init__.py      # This file - exports public API
    └── helpers.py       # Implementation

Why separate files?
1. __init__.py stays clean (just imports)
2. helpers.py can be long without cluttering imports
3. Easy to add more modules (retry.py, cache.py, etc.)

Usage:
    # Import from package (preferred)
    from research_agent.utils import configure_logging, retry

    # Or import module directly
    from research_agent.utils.helpers import RateLimiter

Compare to Go:
    // Go would have separate packages
    import (
        "myapp/pkg/logging"
        "myapp/pkg/retry"
    )

    # Python groups related utils in one package
    from research_agent.utils import configure_logging, retry
"""

from research_agent.utils.helpers import (
    # Logging
    configure_logging,
    get_logger,
    ColoredFormatter,
    JsonFormatter,

    # Retry
    retry,
    async_retry,

    # Rate limiting
    RateLimiter,

    # Text utilities
    truncate_text,
    estimate_tokens,
    clean_whitespace,
    extract_code_blocks,

    # JSON utilities
    safe_json_loads,
    safe_json_dumps,

    # Environment utilities
    get_env,
    get_env_bool,
    get_env_int,

    # Timing utilities
    Timer,
    timed,
    async_timed,

    # Miscellaneous
    chunk_list,
    flatten_dict,
    generate_id,
)

__all__ = [
    # Logging
    "configure_logging",
    "get_logger",
    "ColoredFormatter",
    "JsonFormatter",

    # Retry
    "retry",
    "async_retry",

    # Rate limiting
    "RateLimiter",

    # Text utilities
    "truncate_text",
    "estimate_tokens",
    "clean_whitespace",
    "extract_code_blocks",

    # JSON utilities
    "safe_json_loads",
    "safe_json_dumps",

    # Environment utilities
    "get_env",
    "get_env_bool",
    "get_env_int",

    # Timing utilities
    "Timer",
    "timed",
    "async_timed",

    # Miscellaneous
    "chunk_list",
    "flatten_dict",
    "generate_id",
]
