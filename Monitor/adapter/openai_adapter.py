from __future__ import annotations

from typing import Any, Dict

import httpx

from .base_adapter import ChatAdapter


class OpenAIAdapter(ChatAdapter):
    """
    Adapter for OpenAI-style chat completion APIs.

    It expects the following keys in `self.config`:
    - api_base: base URL of the API, e.g. https://api.openai.com/v1
    - api_key: API key string
    - model:   model name to use
    - timeout: optional request timeout in seconds (default: 30)
    """

    async def chat(self, prompt: str, **kwargs: Any) -> str:
        api_base = self.config["api_base"]
        api_key = self.config["api_key"]
        model = self.config["model"]
        timeout = float(self.config.get("timeout", 30))

        messages = kwargs.get(
            "messages",
            [{"role": "user", "content": prompt}],
        )

        async with httpx.AsyncClient(base_url=api_base, timeout=timeout, trust_env=False) as client:
            response = await client.post(
                "/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": messages,
                },
            )
            response.raise_for_status()
            data = response.json()
            # Adjust this according to the actual API schema if necessary
            return data["choices"][0]["message"]["content"]


