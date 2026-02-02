"""
Utility functions and helpers for the research agent.

🎓 MODULE OVERVIEW
==================
This module contains general-purpose utilities used throughout the codebase:

1. Logging configuration
2. Retry decorators (sync and async)
3. Rate limiting
4. Text manipulation
5. JSON helpers
6. Environment variable utilities
7. Timing utilities

🎓 TEACHING: Utility Modules
============================
Every project needs a "utils" or "helpers" module. Design principles:

1. SINGLE PURPOSE - Each function does one thing well
2. NO SIDE EFFECTS - Pure functions when possible
3. WELL TESTED - Utils are used everywhere, bugs spread fast
4. DOCUMENTED - Clear docstrings with examples

Compare to Go:
    // Go puts utilities in internal/pkg/
    internal/
        pkg/
            retry/
            logging/
            strings/

    # Python uses a single utils module or package
    utils/
        __init__.py
        helpers.py
        retry.py
"""

import asyncio
import functools
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Callable, TypeVar, ParamSpec, Awaitable

# Type variables for generic decorators
T = TypeVar("T")
P = ParamSpec("P")


# =============================================================================
# 🎓 LOGGING CONFIGURATION
# =============================================================================
"""
🎓 TEACHING: Python Logging
===========================
Python's logging module is powerful but confusing at first.

Key concepts:
1. LOGGER - The object you call .info(), .error() on
2. HANDLER - Where logs go (console, file, network)
3. FORMATTER - How logs look (timestamp, level, message)
4. LEVEL - What gets logged (DEBUG < INFO < WARNING < ERROR)

Hierarchy:
    root logger
    └── research_agent
        ├── research_agent.llm
        ├── research_agent.tools
        └── research_agent.core

Setting level on parent affects children!

Compare to Go:
    // Go uses structured logging (slog, zerolog, zap)
    logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
    logger.Info("message", "key", "value")

    # Python traditional logging
    logger = logging.getLogger(__name__)
    logger.info("message", extra={"key": "value"})

    # Python structlog (similar to Go)
    import structlog
    logger = structlog.get_logger()
    logger.info("message", key="value")
"""


def configure_logging(
    level: str = "INFO",
    format_string: str | None = None,
    log_file: str | None = None,
    json_format: bool = False,
) -> logging.Logger:
    """
    Configure logging for the application.

    🎓 TEACHING: Logging Best Practices
    ====================================
    1. Use __name__ for logger names (creates hierarchy)
    2. Configure at application entry point only
    3. Libraries should NEVER configure logging (let app decide)
    4. Use appropriate levels:
       - DEBUG: Detailed debugging (development only)
       - INFO: General operational messages
       - WARNING: Something unexpected but handled
       - ERROR: Something failed
       - CRITICAL: Application cannot continue

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string (optional)
        log_file: Path to log file (optional, logs to console if None)
        json_format: If True, output JSON logs (good for production)

    Returns:
        Configured root logger for the application

    Example:
        # Basic usage
        configure_logging(level="DEBUG")

        # With file output
        configure_logging(level="INFO", log_file="app.log")

        # JSON format for production
        configure_logging(level="INFO", json_format=True)
    """
    # Get the root logger for our application
    logger = logging.getLogger("research_agent")

    # Convert string level to logging constant
    # 🎓 NOTE: getattr is safer than eval for dynamic attribute access
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Clear existing handlers (avoid duplicate logs on reconfigure)
    logger.handlers.clear()

    # Create formatter
    if json_format:
        # JSON format for production/log aggregation
        formatter = JsonFormatter()
    elif format_string:
        formatter = logging.Formatter(format_string)
    else:
        # Default format with colors for console
        formatter = ColoredFormatter(
            "%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
            datefmt="%H:%M:%S",
        )

    # Console handler (always)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        # Always use plain format for files (no colors)
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Don't propagate to root logger (avoid duplicate logs)
    logger.propagate = False

    logger.debug(f"Logging configured: level={level}, file={log_file}")
    return logger


class ColoredFormatter(logging.Formatter):
    """
    Formatter that adds ANSI colors to log levels.

    🎓 TEACHING: ANSI Escape Codes
    ==============================
    Terminal colors use escape sequences:
        \033[XXm  where XX is a color code

    Common codes:
        31 = Red, 32 = Green, 33 = Yellow
        34 = Blue, 35 = Magenta, 36 = Cyan
        0 = Reset (back to normal)

    Example:
        print("\033[31mThis is red\033[0m")

    Note: Windows CMD needs special handling (or use colorama library)
    Modern Windows Terminal and PowerShell support ANSI natively.
    """

    # Color codes for each level
    COLORS = {
        logging.DEBUG: "\033[36m",     # Cyan
        logging.INFO: "\033[32m",      # Green
        logging.WARNING: "\033[33m",   # Yellow
        logging.ERROR: "\033[31m",     # Red
        logging.CRITICAL: "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format with colors if terminal supports it."""
        # Check if output is a terminal (not redirected to file)
        if hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
            color = self.COLORS.get(record.levelno, "")
            record.levelname = f"{color}{record.levelname}{self.RESET}"

        return super().format(record)


class JsonFormatter(logging.Formatter):
    """
    Formatter that outputs JSON logs.

    🎓 TEACHING: Structured Logging
    ===============================
    JSON logs are machine-parseable, great for:
    - Log aggregation (ELK stack, Splunk, CloudWatch)
    - Searching and filtering
    - Metrics extraction

    Example output:
    {"timestamp": "2024-01-15T10:30:00", "level": "INFO", "message": "User logged in", "user_id": 123}

    Compare to plain text:
    2024-01-15 10:30:00 INFO User logged in user_id=123

    JSON is harder to read but easier to process.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "exc_info", "exc_text", "thread", "threadName",
                "message",
            ):
                log_data[key] = value

        return json.dumps(log_data)


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Get a logger instance.

    🎓 TEACHING: Logger Naming Convention
    =====================================
    Always use __name__ for logger names:

        logger = get_logger(__name__)

    This creates hierarchical loggers:
        research_agent.llm.client
        research_agent.tools.search

    Benefits:
    - Filter logs by module
    - Set different levels per module
    - Clear source of log messages

    Args:
        name: Logger name (use __name__)

    Returns:
        Logger instance
    """
    if name is None:
        return logging.getLogger("research_agent")
    return logging.getLogger(f"research_agent.{name}")


# =============================================================================
# 🎓 RETRY DECORATORS
# =============================================================================
"""
🎓 TEACHING: Retry Pattern
==========================
External services fail. Networks timeout. APIs rate-limit.
The retry pattern handles transient failures gracefully.

Key decisions:
1. MAX RETRIES - How many times to try (3-5 typical)
2. BACKOFF - How long to wait between retries
   - Fixed: Always wait same time
   - Linear: Wait increases (1s, 2s, 3s)
   - Exponential: Wait doubles (1s, 2s, 4s, 8s)
3. JITTER - Random variation to avoid thundering herd
4. EXCEPTIONS - Which errors to retry (not all!)

Compare to Go:
    // Go uses explicit loops
    for i := 0; i < maxRetries; i++ {
        result, err := doSomething()
        if err == nil {
            return result
        }
        time.Sleep(backoff)
    }

    # Python uses decorators for cleaner code
    @retry(max_attempts=3)
    def do_something():
        ...
"""


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator that retries a function on failure.

    🎓 TEACHING: Decorator with Arguments
    =====================================
    This is a "decorator factory" - a function that returns a decorator.

    Why? Decorators without arguments:
        @simple_decorator
        def func(): ...

    Decorators WITH arguments need an extra layer:
        @decorator_with_args(arg1, arg2)
        def func(): ...

    The layers:
    1. retry(max_attempts=3) - Returns the actual decorator
    2. decorator(func) - Wraps the function
    3. wrapper(*args) - Called when function is invoked

    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry

    Returns:
        Decorated function that retries on failure

    Example:
        @retry(max_attempts=3, delay=1.0, backoff=2.0)
        def fetch_data():
            return requests.get("https://api.example.com")

        # Will try up to 3 times with delays: 1s, 2s, 4s
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)  # Preserves function metadata
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Exception | None = None
            current_delay = delay

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        logger = get_logger("retry")
                        logger.warning(
                            f"Attempt {attempt}/{max_attempts} failed: {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger = get_logger("retry")
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}"
                        )

            # All retries exhausted
            raise last_exception  # type: ignore

        return wrapper

    return decorator


def async_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """
    Async version of retry decorator.

    🎓 TEACHING: Async Decorators
    =============================
    Async decorators are tricky because:
    1. The wrapper must be async
    2. Must use await inside wrapper
    3. Must use asyncio.sleep not time.sleep

    Common mistake:
        # WRONG - blocks event loop!
        time.sleep(1)

        # CORRECT - yields to event loop
        await asyncio.sleep(1)

    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry

    Returns:
        Decorated async function that retries on failure

    Example:
        @async_retry(max_attempts=3)
        async def fetch_data():
            async with httpx.AsyncClient() as client:
                return await client.get("https://api.example.com")
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Exception | None = None
            current_delay = delay

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        logger = get_logger("retry")
                        logger.warning(
                            f"Attempt {attempt}/{max_attempts} failed: {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger = get_logger("retry")
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}"
                        )

            raise last_exception  # type: ignore

        return wrapper

    return decorator


# =============================================================================
# 🎓 RATE LIMITING
# =============================================================================
"""
🎓 TEACHING: Rate Limiting
==========================
APIs have rate limits. Exceeding them gets you blocked.
Rate limiting ensures we stay within allowed request rates.

Common algorithms:
1. TOKEN BUCKET - Tokens refill over time, each request uses a token
2. SLIDING WINDOW - Count requests in rolling time window
3. FIXED WINDOW - Count requests in fixed time periods

We implement a simple token bucket for its simplicity.

Compare to Go:
    // Go's rate package has built-in limiter
    limiter := rate.NewLimiter(rate.Every(time.Second), 10)
    limiter.Wait(ctx)  // Blocks until allowed

    # Python needs manual implementation or aiolimiter package
"""


class RateLimiter:
    """
    Simple token bucket rate limiter.

    🎓 TEACHING: Token Bucket Algorithm
    ===================================
    Imagine a bucket that:
    - Holds max N tokens
    - Tokens are added at rate R per second
    - Each request takes 1 token
    - If empty, request must wait

    Example: 10 requests/second
    - Bucket holds 10 tokens
    - 10 tokens added per second
    - Can burst up to 10 requests
    - Then limited to 10/second

    This allows short bursts while maintaining average rate.

    Usage:
        limiter = RateLimiter(requests_per_second=10)

        # Sync
        limiter.wait()  # Blocks if rate exceeded
        make_request()

        # Async
        await limiter.wait_async()
        await make_request()
    """

    def __init__(self, requests_per_second: float, burst: int | None = None):
        """
        Initialize rate limiter.

        Args:
            requests_per_second: Maximum sustained request rate
            burst: Maximum burst size (defaults to requests_per_second)
        """
        self.rate = requests_per_second
        self.burst = burst or int(requests_per_second)
        self.tokens = float(self.burst)
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_update
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        self.last_update = now

    def wait(self) -> None:
        """
        Wait until a request is allowed (sync version).

        🎓 NOTE: This blocks the thread. For async code, use wait_async().
        """
        while True:
            self._refill()
            if self.tokens >= 1:
                self.tokens -= 1
                return
            # Calculate sleep time
            sleep_time = (1 - self.tokens) / self.rate
            time.sleep(sleep_time)

    async def wait_async(self) -> None:
        """
        Wait until a request is allowed (async version).

        🎓 NOTE: Uses asyncio.sleep to not block the event loop.
        """
        async with self._lock:  # Ensure thread safety
            while True:
                self._refill()
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                sleep_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(sleep_time)

    def try_acquire(self) -> bool:
        """
        Try to acquire a token without waiting.

        Returns:
            True if token acquired, False if rate limited
        """
        self._refill()
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False


# =============================================================================
# 🎓 TEXT UTILITIES
# =============================================================================
"""
🎓 TEACHING: String Handling
============================
LLMs work with text, so text utilities are essential.
Common operations:
- Truncating long text (with ellipsis)
- Counting tokens (approximately)
- Cleaning whitespace
- Extracting snippets
"""


def truncate_text(
    text: str,
    max_length: int = 1000,
    suffix: str = "...",
    word_boundary: bool = True,
) -> str:
    """
    Truncate text to maximum length.

    🎓 TEACHING: Why Truncate?
    ==========================
    1. LLMs have context limits
    2. Logs should be readable
    3. UI has display constraints
    4. API responses have size limits

    The word_boundary option avoids cutting mid-word:
        "Hello wor..."  →  "Hello..."

    Args:
        text: Text to truncate
        max_length: Maximum length including suffix
        suffix: String to append when truncated
        word_boundary: If True, cut at word boundary

    Returns:
        Truncated text

    Example:
        >>> truncate_text("Hello world!", 8)
        'Hello...'
        >>> truncate_text("Hello world!", 8, word_boundary=False)
        'Hello...'
    """
    if len(text) <= max_length:
        return text

    # Account for suffix in max length
    target_length = max_length - len(suffix)

    if target_length <= 0:
        return suffix[:max_length]

    truncated = text[:target_length]

    if word_boundary:
        # Find last space to avoid cutting mid-word
        last_space = truncated.rfind(" ")
        if last_space > target_length // 2:  # Only if reasonable
            truncated = truncated[:last_space]

    return truncated.rstrip() + suffix


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.

    🎓 TEACHING: Token Estimation
    =============================
    LLMs don't process characters - they process tokens.
    Tokens are roughly:
    - 1 token ≈ 4 characters (English)
    - 1 token ≈ 0.75 words

    This is a rough estimate! For exact counts, use:
    - anthropic's count_tokens() method
    - tiktoken library (OpenAI's tokenizer)

    Why estimate?
    - Check if text fits in context window
    - Calculate costs
    - Decide when to truncate

    Args:
        text: Text to estimate

    Returns:
        Estimated token count
    """
    # Simple heuristic: ~4 chars per token for English
    return len(text) // 4


def clean_whitespace(text: str) -> str:
    """
    Clean and normalize whitespace in text.

    🎓 TEACHING: Whitespace Cleaning
    ================================
    Common issues with text:
    - Multiple spaces: "hello    world"
    - Mixed newlines: "hello\r\nworld"
    - Leading/trailing whitespace
    - Tabs vs spaces inconsistency

    This function normalizes all of these.

    Args:
        text: Text to clean

    Returns:
        Cleaned text
    """
    import re

    # Replace multiple whitespace with single space
    text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing
    return text.strip()


def extract_code_blocks(text: str) -> list[dict[str, str]]:
    """
    Extract code blocks from markdown text.

    🎓 TEACHING: Parsing LLM Output
    ===============================
    LLMs often return markdown with code blocks:

    ```python
    print("hello")
    ```

    Extracting these programmatically is useful for:
    - Executing generated code
    - Syntax highlighting
    - Code analysis

    Args:
        text: Markdown text with code blocks

    Returns:
        List of dicts with 'language' and 'code' keys

    Example:
        >>> text = '''Here's code:
        ... ```python
        ... print("hi")
        ... ```
        ... '''
        >>> extract_code_blocks(text)
        [{'language': 'python', 'code': 'print("hi")'}]
    """
    import re

    pattern = r"```(\w*)\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)

    return [
        {"language": lang or "text", "code": code.strip()}
        for lang, code in matches
    ]


# =============================================================================
# 🎓 JSON UTILITIES
# =============================================================================
"""
🎓 TEACHING: Safe JSON Handling
===============================
JSON parsing can fail. External data is unpredictable.
Always handle JSON errors gracefully.
"""


def safe_json_loads(
    text: str,
    default: Any = None,
    fix_common_errors: bool = True,
) -> Any:
    """
    Safely parse JSON with error handling.

    🎓 TEACHING: Common JSON Errors from LLMs
    =========================================
    LLMs sometimes generate malformed JSON:

    1. Single quotes instead of double:
       {'key': 'value'}  →  {"key": "value"}

    2. Trailing commas:
       {"a": 1,}  →  {"a": 1}

    3. Unquoted keys:
       {key: "value"}  →  {"key": "value"}

    4. Comments (not valid JSON):
       {"a": 1 // comment}  →  {"a": 1}

    This function attempts to fix common errors.

    Args:
        text: JSON string to parse
        default: Value to return on parse failure
        fix_common_errors: Attempt to fix malformed JSON

    Returns:
        Parsed JSON or default value

    Example:
        >>> safe_json_loads('{"key": "value"}')
        {'key': 'value'}
        >>> safe_json_loads("{'key': 'value'}")  # Single quotes
        {'key': 'value'}
        >>> safe_json_loads("invalid", default={})
        {}
    """
    if not text or not text.strip():
        return default

    # Try parsing as-is first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if not fix_common_errors:
            return default

    # Try fixing common errors
    fixed = text

    # Replace single quotes with double quotes
    # 🎓 NOTE: This is naive - fails for strings containing quotes
    fixed = fixed.replace("'", '"')

    # Remove trailing commas before } or ]
    import re
    fixed = re.sub(r",\s*([\]}])", r"\1", fixed)

    # Remove JavaScript-style comments
    fixed = re.sub(r"//.*$", "", fixed, flags=re.MULTILINE)
    fixed = re.sub(r"/\*.*?\*/", "", fixed, flags=re.DOTALL)

    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        return default


def safe_json_dumps(
    obj: Any,
    default: str = "{}",
    pretty: bool = False,
) -> str:
    """
    Safely serialize object to JSON.

    🎓 TEACHING: JSON Serialization Issues
    ======================================
    Not everything is JSON serializable:
    - datetime objects
    - custom classes
    - bytes
    - sets

    This function handles common cases.

    Args:
        obj: Object to serialize
        default: String to return on failure
        pretty: If True, format with indentation

    Returns:
        JSON string
    """

    def json_serializer(o: Any) -> Any:
        """Custom serializer for non-standard types."""
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, bytes):
            return o.decode("utf-8", errors="replace")
        if isinstance(o, set):
            return list(o)
        if hasattr(o, "model_dump"):  # Pydantic model
            return o.model_dump()
        if hasattr(o, "__dict__"):
            return o.__dict__
        return str(o)

    try:
        return json.dumps(
            obj,
            default=json_serializer,
            indent=2 if pretty else None,
            ensure_ascii=False,
        )
    except (TypeError, ValueError):
        return default


# =============================================================================
# 🎓 ENVIRONMENT UTILITIES
# =============================================================================
"""
🎓 TEACHING: Environment Variables
==================================
Environment variables configure applications without code changes.
Critical for:
- Secrets (API keys)
- Environment-specific config (dev vs prod)
- Feature flags
"""


def get_env(
    key: str,
    default: str | None = None,
    required: bool = False,
) -> str | None:
    """
    Get environment variable with validation.

    🎓 TEACHING: Environment Variable Best Practices
    ================================================
    1. Never hardcode secrets
    2. Fail fast if required vars missing
    3. Document required variables
    4. Provide sensible defaults when possible

    Args:
        key: Environment variable name
        default: Default value if not set
        required: If True, raise error when missing

    Returns:
        Environment variable value or default

    Raises:
        ValueError: If required and not set

    Example:
        # Required variable
        api_key = get_env("ANTHROPIC_API_KEY", required=True)

        # Optional with default
        log_level = get_env("LOG_LEVEL", default="INFO")
    """
    value = os.environ.get(key, default)

    if required and value is None:
        raise ValueError(
            f"Required environment variable '{key}' is not set. "
            f"Set it with: export {key}=<value>"
        )

    return value


def get_env_bool(key: str, default: bool = False) -> bool:
    """
    Get environment variable as boolean.

    🎓 TEACHING: Boolean Environment Variables
    ==========================================
    There's no standard for boolean env vars. Common conventions:
    - "true", "1", "yes", "on" → True
    - "false", "0", "no", "off" → False

    We support all common conventions for flexibility.

    Args:
        key: Environment variable name
        default: Default value if not set

    Returns:
        Boolean value
    """
    value = os.environ.get(key, "").lower()

    if not value:
        return default

    if value in ("true", "1", "yes", "on", "enabled"):
        return True
    if value in ("false", "0", "no", "off", "disabled"):
        return False

    return default


def get_env_int(key: str, default: int = 0) -> int:
    """
    Get environment variable as integer.

    Args:
        key: Environment variable name
        default: Default value if not set or invalid

    Returns:
        Integer value
    """
    value = os.environ.get(key)

    if value is None:
        return default

    try:
        return int(value)
    except ValueError:
        return default


# =============================================================================
# 🎓 TIMING UTILITIES
# =============================================================================
"""
🎓 TEACHING: Performance Measurement
====================================
Measuring execution time helps:
- Find bottlenecks
- Monitor production
- Compare implementations
"""


class Timer:
    """
    Context manager for timing code blocks.

    🎓 TEACHING: Context Managers
    =============================
    Context managers use __enter__ and __exit__ for setup/cleanup:

        with open("file.txt") as f:  # __enter__ opens file
            data = f.read()
        # __exit__ closes file automatically

    Our Timer:
        with Timer("API call"):  # __enter__ starts timer
            response = api.call()
        # __exit__ logs duration

    Compare to Go:
        // Go uses defer
        start := time.Now()
        defer func() {
            log.Printf("Duration: %v", time.Since(start))
        }()

    Usage:
        with Timer("Processing"):
            do_work()
        # Logs: "Processing completed in 1.23s"

        # Or get elapsed time
        with Timer("Processing") as t:
            do_work()
        print(f"Took {t.elapsed:.2f}s")
    """

    def __init__(self, name: str = "Operation", log: bool = True):
        """
        Initialize timer.

        Args:
            name: Name for logging
            log: If True, log duration on exit
        """
        self.name = name
        self.log = log
        self.start_time: float = 0
        self.end_time: float = 0
        self.elapsed: float = 0

    def __enter__(self) -> "Timer":
        """Start the timer."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        """Stop the timer and optionally log."""
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time

        if self.log:
            logger = get_logger("timer")
            logger.debug(f"{self.name} completed in {self.elapsed:.3f}s")


def timed(func: Callable[P, T]) -> Callable[P, T]:
    """
    Decorator that logs function execution time.

    🎓 TEACHING: Simple Decorator
    =============================
    This is a "simple" decorator (no arguments).
    Compare to retry() which is a decorator factory.

    @timed
    def slow_function():
        ...

    vs

    @retry(max_attempts=3)  # needs ()
    def flaky_function():
        ...

    Usage:
        @timed
        def process_data(data):
            # ... processing
            return result

        # Logs: "process_data completed in 0.123s"
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        with Timer(func.__name__):
            return func(*args, **kwargs)

    return wrapper


def async_timed(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
    """
    Async version of timed decorator.

    Usage:
        @async_timed
        async def fetch_data():
            return await api.get()
    """

    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start = time.perf_counter()
        try:
            return await func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            logger = get_logger("timer")
            logger.debug(f"{func.__name__} completed in {elapsed:.3f}s")

    return wrapper


# =============================================================================
# 🎓 MISCELLANEOUS UTILITIES
# =============================================================================


def chunk_list(lst: list[T], chunk_size: int) -> list[list[T]]:
    """
    Split a list into chunks of specified size.

    🎓 TEACHING: Chunking Data
    ==========================
    Why chunk?
    - Batch processing (API limits)
    - Parallel processing
    - Memory management
    - Progress reporting

    Args:
        lst: List to split
        chunk_size: Maximum size of each chunk

    Returns:
        List of chunks

    Example:
        >>> chunk_list([1, 2, 3, 4, 5], 2)
        [[1, 2], [3, 4], [5]]
    """
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_dict(
    d: dict[str, Any],
    parent_key: str = "",
    separator: str = ".",
) -> dict[str, Any]:
    """
    Flatten a nested dictionary.

    🎓 TEACHING: Dictionary Flattening
    ==================================
    Nested dict:
        {"a": {"b": {"c": 1}}}

    Flattened:
        {"a.b.c": 1}

    Useful for:
    - Logging nested objects
    - Configuration merging
    - Database storage

    Args:
        d: Dictionary to flatten
        parent_key: Prefix for keys (used in recursion)
        separator: String to join nested keys

    Returns:
        Flattened dictionary

    Example:
        >>> flatten_dict({"user": {"name": "John", "age": 30}})
        {'user.name': 'John', 'user.age': 30}
    """
    items: list[tuple[str, Any]] = []

    for key, value in d.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key

        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, separator).items())
        else:
            items.append((new_key, value))

    return dict(items)


def generate_id(prefix: str = "", length: int = 8) -> str:
    """
    Generate a random ID.

    🎓 TEACHING: ID Generation
    ==========================
    Options for unique IDs:
    1. UUID - Guaranteed unique, but long (36 chars)
    2. Random string - Shorter, tiny collision risk
    3. Sequential - Predictable, reveals count
    4. Timestamp-based - Sortable, reveals timing

    We use random for simplicity. For high-volume systems,
    consider UUID or snowflake IDs.

    Args:
        prefix: Optional prefix for the ID
        length: Length of random portion

    Returns:
        Generated ID string

    Example:
        >>> generate_id("msg_")
        'msg_a1b2c3d4'
    """
    import secrets
    import string

    alphabet = string.ascii_lowercase + string.digits
    random_part = "".join(secrets.choice(alphabet) for _ in range(length))

    return f"{prefix}{random_part}"


# =============================================================================
# 🎓 MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Utils Module - Examples")
    print("=" * 60)

    # Logging example
    print("\n1. Logging Configuration:")
    logger = configure_logging(level="DEBUG")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")

    # Timer example
    print("\n2. Timer Example:")
    with Timer("Sleep test") as t:
        time.sleep(0.1)
    print(f"   Elapsed: {t.elapsed:.3f}s")

    # Rate limiter example
    print("\n3. Rate Limiter Example:")
    limiter = RateLimiter(requests_per_second=10)
    print(f"   Can acquire: {limiter.try_acquire()}")
    print(f"   Can acquire: {limiter.try_acquire()}")

    # Text utilities
    print("\n4. Text Utilities:")
    long_text = "This is a very long text that needs to be truncated for display purposes."
    print(f"   Original: {long_text}")
    print(f"   Truncated: {truncate_text(long_text, 30)}")
    print(f"   Estimated tokens: {estimate_tokens(long_text)}")

    # JSON utilities
    print("\n5. JSON Utilities:")
    bad_json = "{'key': 'value', 'number': 42,}"  # Single quotes and trailing comma
    result = safe_json_loads(bad_json)
    print(f"   Fixed JSON: {result}")

    # Generate ID
    print("\n6. ID Generation:")
    print(f"   Random ID: {generate_id()}")
    print(f"   With prefix: {generate_id('user_')}")

    print("\n" + "=" * 60)
    print("Utils module loaded successfully!")
    print("=" * 60)
