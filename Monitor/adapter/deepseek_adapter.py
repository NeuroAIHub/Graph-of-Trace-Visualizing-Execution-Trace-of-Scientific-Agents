from __future__ import annotations

from typing import Any, Dict

import httpx

from .base_adapter import ChatAdapter, extract_content, post_with_retry


class DeepSeekAdapter(ChatAdapter):
    """
    Adapter for DeepSeek-style chat completion APIs.

    It expects the following keys in `self.config`:
    - api_base: base URL of the API
    - api_key: API key string
    - model:   model name to use
    - timeout: optional request timeout in seconds (default: 30)
    """

    async def chat(self, prompt: str, **kwargs: Any) -> str:
        api_base = self.config["api_base"]
        api_key = self.config["api_key"]
        model = self.config["model"]
        timeout = float(self.config.get("timeout", 30))
        max_retries = int(self.config.get("max_retries", 2))

        messages = kwargs.get(
            "messages",
            [{"role": "user", "content": prompt}],
        )

        async with httpx.AsyncClient(base_url=api_base, timeout=timeout, trust_env=False) as client:
            response = await post_with_retry(
                lambda: client.post(
                    "/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": messages,
                    },
                ),
                max_retries=max_retries,
                label="deepseek chat",
            )
            data = response.json()
            return extract_content(
                data,
                lambda d: d["choices"][0]["message"]["content"],
                "choices[0].message.content",
            )


