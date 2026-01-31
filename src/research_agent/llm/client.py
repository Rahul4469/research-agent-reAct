"""
Anthropic Claude client wrapper for the research agent.
"""

import os 
from typing import Any, AsyncIterator
from anthropic import AsyncAnthropic, Anthropic
from research_agent.core.models import Message, Rule, ToolDefinition