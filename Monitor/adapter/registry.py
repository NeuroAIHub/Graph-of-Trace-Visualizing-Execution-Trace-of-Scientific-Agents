from __future__ import annotations

from typing import Dict, Optional, Type

from .base_adapter import ChatAdapter
from .openai_adapter import OpenAIAdapter
from .deepseek_adapter import DeepSeekAdapter
from .azure_adapter import AzureOpenAIAdapter
from ..config.parser import (
    load_config,
    get_active_provider,
    get_active_api,
    get_provider_api_config,
)


ADAPTERS: Dict[str, Type[ChatAdapter]] = {
    "openai": OpenAIAdapter,
    "deepseek": DeepSeekAdapter,
    "azure": AzureOpenAIAdapter,
    # Add new providers here, e.g. "zhipu": ZhipuAdapter,
}


def get_chat_adapter(
    provider: Optional[str] = None,
    api_name: Optional[str] = None,
) -> ChatAdapter:
    """
    Unified entrypoint for creating chat adapters.

    - If provider/api_name are not given, use runtime.active_provider / runtime.active_api
      from the merged configuration.
    - Otherwise, use the explicitly provided provider/api_name.
    """
    cfg = load_config()

    provider_name = (provider or get_active_provider(cfg)).lower()
    api_config_name = api_name or get_active_api(cfg)

    if provider_name not in ADAPTERS:
        raise ValueError(f"Unsupported provider: {provider_name}")

    api_cfg = get_provider_api_config(cfg, provider_name, api_config_name)
    adapter_cls = ADAPTERS[provider_name]
    return adapter_cls(api_cfg)


