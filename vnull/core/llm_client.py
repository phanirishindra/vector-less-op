"""LLM Client for local llama.cpp server.

Uses OpenAI SDK configured for local inference.
DO NOT ping the real OpenAI API - this is strictly for local LLM.
"""

import asyncio
import json
import re
from collections.abc import AsyncGenerator
from typing import Any

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
    """Client for interacting with local LLM via OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> None:
        self.base_url = base_url or settings.llm_base_url
        self.api_key = api_key or settings.llm_api_key
        self.model = model or settings.llm_model
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self.timeout = timeout or settings.llm_timeout

        self._async_client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

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

    def _extract_content_and_usage(self, response: Any) -> tuple[str, str | None, Any]:
        """Normalize OpenAI-compatible responses robustly."""
        content = ""
        finish_reason = None
        usage = getattr(response, "usage", None)

        try:
            choices = getattr(response, "choices", None) or []
            if choices:
                c0 = choices[0]
                finish_reason = getattr(c0, "finish_reason", None)

                # Chat-completions shape
                msg = getattr(c0, "message", None)
                if msg is not None:
                    msg_content = getattr(msg, "content", None)
                    if isinstance(msg_content, str):
                        content = msg_content
                    elif isinstance(msg_content, list):
                        # Some providers return typed blocks
                        parts: list[str] = []
                        for part in msg_content:
                            if isinstance(part, dict):
                                txt = part.get("text")
                                if isinstance(txt, str):
                                    parts.append(txt)
                        content = "".join(parts)

                # Legacy/text fallback
                if not content:
                    txt = getattr(c0, "text", None)
                    if isinstance(txt, str):
                        content = txt
        except Exception as e:
            logger.warning("Failed to normalize LLM response", error=str(e))

        if content is None:
            content = ""

        return content, finish_reason, usage

    @retry(
        retry=retry_if_exception_type((TimeoutError, ConnectionError, asyncio.TimeoutError)),
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

        content, finish_reason, usage = self._extract_content_and_usage(response)

        logger.debug(
            "Completion received",
            content_length=len(content),
            finish_reason=finish_reason,
            tokens=getattr(usage, "total_tokens", None) if usage else None,
        )

        return LLMResponse(
            content=content or "",
            finish_reason=finish_reason,
            prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
            total_tokens=getattr(usage, "total_tokens", None) if usage else None,
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
                if token:
                    yield token
        else:
            async for chunk in stream:
                try:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                except Exception:
                    continue

    async def _filter_think_tags(self, stream: Any) -> AsyncGenerator[str, None]:
        buffer = ""
        in_think_block = False
        think_start = "<think>"
        think_end = "</think>"

        async for chunk in stream:
            try:
                if not chunk.choices or not chunk.choices[0].delta.content:
                    continue
                token = chunk.choices[0].delta.content
            except Exception:
                continue

            buffer += token

            while True:
                if in_think_block:
                    end_idx = buffer.find(think_end)
                    if end_idx != -1:
                        buffer = buffer[end_idx + len(think_end) :]
                        in_think_block = False
                    else:
                        if len(buffer) > len(think_end):
                            buffer = buffer[-(len(think_end) - 1) :]
                        break
                else:
                    start_idx = buffer.find(think_start)
                    if start_idx != -1:
                        if start_idx > 0:
                            yield buffer[:start_idx]
                        buffer = buffer[start_idx + len(think_start) :]
                        in_think_block = True
                    else:
                        safe_end = 0
                        for i in range(1, len(think_start)):
                            if buffer.endswith(think_start[:i]):
                                safe_end = i
                                break

                        if safe_end > 0:
                            yield buffer[:-safe_end]
                            buffer = buffer[-safe_end:]
                        else:
                            yield buffer
                            buffer = ""
                        break

        if buffer and not in_think_block:
            yield buffer

    async def complete_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any] | list[Any]:
        response = await self.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature if temperature is not None else 0.0,
        )

        content = (response.content or "").strip()

        # 1) direct parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # 2) code fence parse
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # 3) greedy object/array extraction fallback
        for pattern in [r"(\{[\s\S]*\})", r"(\[[\s\S]*\])"]:
            match = re.search(pattern, content)
            if match:
                candidate = match.group(1).strip()
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue

        raise ValueError(f"Could not parse JSON from response: {content[:300]}...")

    def complete_sync(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
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

        content, finish_reason, usage = self._extract_content_and_usage(response)

        return LLMResponse(
            content=content or "",
            finish_reason=finish_reason,
            prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
            total_tokens=getattr(usage, "total_tokens", None) if usage else None,
        )

    async def flush_kv_cache(self) -> None:
        try:
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
        try:
            response = await self.complete(
                prompt="Say 'ok'",
                max_tokens=5,
                temperature=0,
            )
            return bool((response.content or "").strip())
        except Exception as e:
            logger.error("LLM health check failed", error=str(e))
            return False

    async def __aenter__(self) -> "LLMClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._async_client.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self._sync_client.close()
