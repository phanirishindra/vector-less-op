"""LLM Client for local llama.cpp server.

Uses OpenAI SDK configured for local inference.
DO NOT ping the real OpenAI API - this is strictly for local LLM.
"""

import asyncio
import re
from collections.abc import AsyncGenerator
from typing import Any, Literal

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from vnull.core.config import settings
from vnull.core.logging import get_logger

logger = get_logger(__name__)


class LLMResponse(BaseModel):
    """Structured response from LLM."""

    content: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class LLMClient:
    """Client for interacting with local LLM via OpenAI-compatible API.

    Configured strictly for local llama.cpp server inference.
    Uses the OpenAI SDK with custom base_url pointing to localhost.

    Example:
        >>> client = LLMClient()
        >>> response = await client.complete("Translate this HTML to Markdown: <h1>Hello</h1>")
        >>> print(response.content)
        # Hello
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> None:
        """Initialize LLM client.

        Args:
            base_url: LLM server URL. Defaults to settings.llm_base_url.
            api_key: API key. Defaults to settings.llm_api_key.
            model: Model identifier. Defaults to settings.llm_model.
            temperature: Generation temperature. Defaults to settings.llm_temperature.
            max_tokens: Max response tokens. Defaults to settings.llm_max_tokens.
            timeout: Request timeout in seconds. Defaults to settings.llm_timeout.
        """
        self.base_url = base_url or settings.llm_base_url
        self.api_key = api_key or settings.llm_api_key
        self.model = model or settings.llm_model
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self.timeout = timeout or settings.llm_timeout

        # Async client for most operations
        self._async_client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

        # Sync client for blocking operations
        self._sync_client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

        logger.info(
            "LLM client initialized",
            base_url=self.base_url,
            model=self.model,
        )

    @retry(
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        """Generate a completion from the LLM.

        Args:
            prompt: User prompt/message.
            system_prompt: Optional system prompt for context.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.
            stop: Stop sequences.

        Returns:
            LLMResponse with content and token usage.
        """
        messages: list[dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        logger.debug(
            "Sending completion request",
            prompt_length=len(prompt),
            has_system=bool(system_prompt),
        )

        response = await self._async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            stop=stop,
        )

        content = response.choices[0].message.content or ""
        usage = response.usage

        logger.debug(
            "Completion received",
            content_length=len(content),
            finish_reason=response.choices[0].finish_reason,
            tokens=usage.total_tokens if usage else None,
        )

        return LLMResponse(
            content=content,
            finish_reason=response.choices[0].finish_reason,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
        )

    async def stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        hide_think_tags: bool = True,
    ) -> AsyncGenerator[str, None]:
        """Stream completion tokens from the LLM.

        Implements streaming generator that can intercept and hide
        <think>...</think> tags from the output.

        Args:
            prompt: User prompt/message.
            system_prompt: Optional system prompt.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.
            stop: Stop sequences.
            hide_think_tags: If True, filter out <think>...</think> content.

        Yields:
            String tokens as they arrive.
        """
        messages: list[dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        stream = await self._async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            stop=stop,
            stream=True,
        )

        if hide_think_tags:
            async for token in self._filter_think_tags(stream):
                yield token
        else:
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

    async def _filter_think_tags(
        self,
        stream: Any,
    ) -> AsyncGenerator[str, None]:
        """Filter out <think>...</think> tags from streaming output.

        Implements a state machine to track when we're inside think tags
        and buffer content appropriately.
        """
        buffer = ""
        in_think_block = False
        think_start = "<think>"
        think_end = "</think>"

        async for chunk in stream:
            if not chunk.choices or not chunk.choices[0].delta.content:
                continue

            token = chunk.choices[0].delta.content
            buffer += token

            while True:
                if in_think_block:
                    # Look for end of think block
                    end_idx = buffer.find(think_end)
                    if end_idx != -1:
                        # Found end, discard think content
                        buffer = buffer[end_idx + len(think_end) :]
                        in_think_block = False
                    else:
                        # Still in think block, check if we might have partial end tag
                        if len(buffer) > len(think_end):
                            # Keep only potential partial tag
                            buffer = buffer[-(len(think_end) - 1) :]
                        break
                else:
                    # Look for start of think block
                    start_idx = buffer.find(think_start)
                    if start_idx != -1:
                        # Yield content before think block
                        if start_idx > 0:
                            yield buffer[:start_idx]
                        buffer = buffer[start_idx + len(think_start) :]
                        in_think_block = True
                    else:
                        # Check for partial start tag at end
                        safe_end = 0
                        for i in range(1, len(think_start)):
                            if buffer.endswith(think_start[:i]):
                                safe_end = i
                                break

                        if safe_end > 0:
                            # Yield everything except potential partial tag
                            yield buffer[: -safe_end]
                            buffer = buffer[-safe_end:]
                        else:
                            # No partial tag, yield everything
                            yield buffer
                            buffer = ""
                        break

        # Yield any remaining content not in think block
        if buffer and not in_think_block:
            yield buffer

    async def complete_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any] | list[Any]:
        """Generate a JSON response from the LLM.

        Attempts to parse the response as JSON, with fallback extraction.

        Args:
            prompt: User prompt requesting JSON output.
            system_prompt: Optional system prompt.
            temperature: Override default temperature.

        Returns:
            Parsed JSON as dict or list.

        Raises:
            ValueError: If response cannot be parsed as JSON.
        """
        import json

        response = await self.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature or 0.0,  # Lower temp for structured output
        )

        content = response.content.strip()

        # Try direct parse first
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try to find JSON array or object
        for pattern in [r"(\[.*\])", r"(\{.*\})"]:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue

        raise ValueError(f"Could not parse JSON from response: {content[:200]}...")

    def complete_sync(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Synchronous completion for non-async contexts.

        Args:
            prompt: User prompt/message.
            system_prompt: Optional system prompt.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.

        Returns:
            LLMResponse with content and token usage.
        """
        messages: list[dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = self._sync_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens or self.max_tokens,
        )

        content = response.choices[0].message.content or ""
        usage = response.usage

        return LLMResponse(
            content=content,
            finish_reason=response.choices[0].finish_reason,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
        )

    async def flush_kv_cache(self) -> None:
        """Attempt to flush KV cache on the LLM server.

        This is server-specific and may not be supported by all backends.
        For llama.cpp, we send a minimal request to help clear context.
        """
        try:
            # Send a minimal completion to help rotate KV cache
            await self._async_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "."}],
                max_tokens=1,
                temperature=0,
            )
            logger.debug("KV cache flush attempted")
        except Exception as e:
            logger.warning("KV cache flush failed", error=str(e))

    async def health_check(self) -> bool:
        """Check if the LLM server is responsive.

        Returns:
            True if server responds, False otherwise.
        """
        try:
            response = await self.complete(
                prompt="Say 'ok'",
                max_tokens=5,
                temperature=0,
            )
            return bool(response.content)
        except Exception as e:
            logger.error("LLM health check failed", error=str(e))
            return False

    async def __aenter__(self) -> "LLMClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self._async_client.close()

    def __enter__(self) -> "LLMClient":
        """Sync context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Sync context manager exit."""
        self._sync_client.close()
