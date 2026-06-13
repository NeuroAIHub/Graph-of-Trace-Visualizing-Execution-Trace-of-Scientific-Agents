from __future__ import annotations

from typing import Any, Dict, List

import httpx

from .base_adapter import ChatAdapter, extract_content, post_with_retry


class AnthropicAdapter(ChatAdapter):
    """
    Adapter for the Anthropic Messages API.

    Unlike the OpenAI-style providers, Anthropic uses a different endpoint,
    auth header, and response schema:
    - endpoint: POST {api_base}/messages
    - headers:  x-api-key, anthropic-version
    - body:     requires max_tokens; an optional top-level `system` string is
                separate from the messages list
    - response: content[0].text

    It expects the following keys in `self.config`:
    - api_base:        base URL of the API (default: https://api.anthropic.com/v1)
    - api_key:         API key string
    - model:           model name to use, e.g. claude-opus-4-8
    - max_tokens:      max output tokens (default: 1024)
    - anthropic_version: API version header (default: 2023-06-01)
    - timeout:         optional request timeout in seconds (default: 30)
    """

    def _extract_system(self, messages: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]]]:
        """Split out any system messages; Anthropic takes `system` at the top level."""
        system_parts: List[str] = []
        chat_messages: List[Dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content")
                if isinstance(content, str) and content:
                    system_parts.append(content)
            else:
                chat_messages.append(msg)
        return "\n\n".join(system_parts), chat_messages

    async def chat(self, prompt: str, **kwargs: Any) -> str:
        api_base = (self.config.get("api_base") or "https://api.anthropic.com/v1").rstrip("/")
        api_key = self.config["api_key"]
        model = self.config["model"]
        max_tokens = int(self.config.get("max_tokens", 1024))
        anthropic_version = self.config.get("anthropic_version") or "2023-06-01"
        timeout = float(self.config.get("timeout", 30))
        max_retries = int(self.config.get("max_retries", 2))

        messages = kwargs.get(
            "messages",
            [{"role": "user", "content": prompt}],
        )
        system, chat_messages = self._extract_system(messages)

        payload: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": chat_messages,
        }
        if system:
            payload["system"] = system

        async with httpx.AsyncClient(base_url=api_base, timeout=timeout, trust_env=False) as client:
            response = await post_with_retry(
                lambda: client.post(
                    "/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": anthropic_version,
                        "Content-Type": "application/json",
                    },
                    json=payload,
                ),
                max_retries=max_retries,
                label="anthropic chat",
            )
            data = response.json()
            return extract_content(
                data,
                lambda d: d["content"][0]["text"],
                "content[0].text",
            )
