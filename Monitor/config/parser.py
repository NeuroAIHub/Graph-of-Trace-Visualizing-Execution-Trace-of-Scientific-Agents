from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import os

import yaml


BASE_DIR = Path(__file__).resolve().parent


def load_yaml(path: Path) -> Dict[str, Any]:
    """Safely load a YAML file, returning an empty dict if it does not exist."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries, with values from override taking precedence."""
    result: Dict[str, Any] = base.copy()
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _replace_env(value: Any) -> Any:
    """
    Replace strings of the form ${ENV_VAR} with the corresponding environment variable.
    Non-string values are returned unchanged.
    """
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_name = value[2:-1]
        return os.getenv(env_name, "")
    if isinstance(value, dict):
        return {k: _replace_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_replace_env(v) for v in value]
    return value


def load_config() -> Dict[str, Any]:
    """
    Load and merge config.yaml and local.yaml, then apply environment variable substitution.
    local.yaml is optional and overrides values from config.yaml when present.
    """
    base_cfg = load_yaml(BASE_DIR / "config.yaml")
    local_cfg = load_yaml(BASE_DIR / "local.yaml")
    merged = deep_merge(base_cfg, local_cfg) if local_cfg else base_cfg
    return _replace_env(merged)


def get_active_provider(cfg: Dict[str, Any]) -> str:
    """Return the currently active provider, falling back to 'openai'."""
    return cfg.get("runtime", {}).get("active_provider", "openai")


def get_active_api(cfg: Dict[str, Any]) -> str:
    """Return the currently active API configuration name, falling back to 'default'."""
    return cfg.get("runtime", {}).get("active_api", "default")


def get_provider_api_config(
    cfg: Dict[str, Any],
    provider: str,
    api_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build the effective configuration for a given provider and api_name.

    It merges provider-level common settings (e.g. api_key, timeout)
    with a specific entry from providers.<provider>.apis.<api_name>.
    """
    provider = provider.lower()
    providers = cfg.get("providers", {})
    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}")

    provider_cfg = providers[provider]
    api_name = api_name or "default"
    apis = provider_cfg.get("apis", {})
    if api_name not in apis:
        raise ValueError(f"Unknown api '{api_name}' for provider '{provider}'")

    api_cfg = apis[api_name]

    # Merge provider-level settings (except 'apis') with api-level config
    merged: Dict[str, Any] = {k: v for k, v in provider_cfg.items() if k != "apis"}
    merged.update(api_cfg)
    return merged


