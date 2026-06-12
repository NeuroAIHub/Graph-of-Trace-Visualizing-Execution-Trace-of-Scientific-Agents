from __future__ import annotations

import logging
from typing import Any

import httpx

from .base_adapter import ChatAdapter

log = logging.getLogger("mcp_tools.monitor")


class AzureOpenAIAdapter(ChatAdapter):
    """
    Adapter for Azure OpenAI chat completions.

    """

    async def chat(self, prompt: str, **kwargs: Any) -> str:
        """
        Call the Azure OpenAI Chat Completions API.

        For easier troubleshooting, this outputs some debug info (without printing
        prompt content), including the final request's base_url, path, api_version,
        deployment, etc.
        """
        raw_api_base = self.config["api_base"]
        # Handle cases where the user includes /openai or /openai/v1 in api_base to avoid duplicate path.
        api_base = raw_api_base.rstrip("/")
        api_base = api_base.replace("/openai/v1", "").replace("/openai", "")

        api_key = self.config["api_key"]
        deployment = self.config["deployment"]
        api_version = self.config.get("api_version") or "2024-02-15-preview"
        timeout = float(self.config.get("timeout", 30))

        messages = kwargs.get(
            "messages",
            [{"role": "user", "content": prompt}],
        )

        # Azure OpenAI path format:
        #   POST {api_base}/openai/deployments/{deployment}/chat/completions?api-version=xxx
        path = f"/openai/deployments/{deployment}/chat/completions"
        params = {"api-version": api_version}

        # ---- Debug info (excludes user content) ----
        log.debug("[AzureOpenAIAdapter] base_url=%s", api_base)
        log.debug("[AzureOpenAIAdapter] path=%s", path)
        log.debug("[AzureOpenAIAdapter] params=%s", params)
        log.debug("[AzureOpenAIAdapter] deployment=%s", deployment)
        log.debug("[AzureOpenAIAdapter] timeout=%s", timeout)

        async with httpx.AsyncClient(base_url=api_base, timeout=timeout, trust_env=False) as client:
            try:
                response = await client.post(
                    path,
                    params=params,
                    headers={
                        "api-key": api_key,
                        "Content-Type": "application/json",
                    },
                    json={
                        "messages": messages,
                    },
                )
                # Print response status code for debugging
                log.debug("[AzureOpenAIAdapter] status_code=%s", response.status_code)
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                # Enhance error message with URL and first part of response text
                text_snippet = exc.response.text[:500]
                raise httpx.HTTPStatusError(
                    f"Azure OpenAI request failed: {exc} "
                    f"(url={exc.request.url}, response_snippet={text_snippet})",
                    request=exc.request,
                    response=exc.response,
                ) from exc

            data = response.json()
            # OpenAI-compatible response structure: choices[0].message.content
            return data["choices"][0]["message"]["content"]


