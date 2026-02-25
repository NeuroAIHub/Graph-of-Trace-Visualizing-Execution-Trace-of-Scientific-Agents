from __future__ import annotations

from typing import Any

import httpx

from .base_adapter import ChatAdapter


class AzureOpenAIAdapter(ChatAdapter):
    """
    Adapter for Azure OpenAI chat completions.

    期望 config 中包含以下字段：
    - api_base:   Azure 资源地址，例如 https://<resource>.openai.azure.com
    - api_key:    Azure OpenAI API Key
    - deployment: 部署名称（deployment name）
    - api_version: API 版本号，例如 2024-02-15-preview
    - timeout:    可选，请求超时时间（秒）
    """

    async def chat(self, prompt: str, **kwargs: Any) -> str:
        """
        调用 Azure OpenAI Chat Completions 接口。

        为了方便排查问题，这里会输出部分调试信息（不会打印 prompt 内容），
        包括最终请求的 base_url、path、api_version、deployment 等。
        """
        raw_api_base = self.config["api_base"]
        # 兼容用户把 /openai 或 /openai/v1 写进 api_base 的情况，避免重复路径。
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

        # Azure OpenAI 路径格式：
        #   POST {api_base}/openai/deployments/{deployment}/chat/completions?api-version=xxx
        path = f"/openai/deployments/{deployment}/chat/completions"
        params = {"api-version": api_version}

        # ---- 调试信息（不包含用户内容）----
        print("[AzureOpenAIAdapter] base_url:", api_base)
        print("[AzureOpenAIAdapter] path:", path)
        print("[AzureOpenAIAdapter] params:", params)
        print("[AzureOpenAIAdapter] deployment:", deployment)
        print("[AzureOpenAIAdapter] timeout:", timeout)

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
                # 打印响应状态码，便于调试
                print("[AzureOpenAIAdapter] status_code:", response.status_code)
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                # 增强错误信息，包含 URL 和响应文本前一部分
                text_snippet = exc.response.text[:500]
                raise httpx.HTTPStatusError(
                    f"Azure OpenAI request failed: {exc} "
                    f"(url={exc.request.url}, response_snippet={text_snippet})",
                    request=exc.request,
                    response=exc.response,
                ) from exc

            data = response.json()
            # 与 OpenAI 兼容的返回结构：choices[0].message.content
            return data["choices"][0]["message"]["content"]


