from __future__ import annotations

import asyncio
import logging
import random
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Dict

import httpx

log = logging.getLogger("mcp_tools.monitor")

# Cap on how much of an error response body we surface in exceptions/logs.
_ERR_SNIPPET_CAP = 500

# HTTP statuses worth retrying: rate limit + transient server-side errors.
# Auth (401/403), not-found (404), and bad-request (400) are NOT retried —
# retrying them only wastes time, so they surface immediately.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

# Defaults; overridable per provider via config (max_retries / retry_base_delay).
_DEFAULT_MAX_RETRIES = 2          # 2 retries => up to 3 attempts total
_DEFAULT_RETRY_BASE_DELAY = 0.5   # seconds; grows exponentially per attempt
_RETRY_DELAY_CAP = 8.0            # seconds; ceiling on a single backoff wait


def _is_retryable(exc: Exception) -> bool:
    """True for transient failures that a retry might recover from."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS
    # Timeouts and connection/transport errors are transient by nature.
    return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))


async def post_with_retry(
    send: Callable[[], Awaitable[httpx.Response]],
    *,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    base_delay: float = _DEFAULT_RETRY_BASE_DELAY,
    label: str = "chat",
) -> httpx.Response:
    """Send a request with bounded exponential backoff + jitter on transient errors.

    `send` performs one HTTP request and returns its Response. This calls it,
    runs raise_for_status_verbose, and on a retryable failure waits and retries
    up to `max_retries` times. Non-retryable errors (auth, 404, 400) and the
    final attempt's failure propagate to the caller unchanged.

    Backoff uses asyncio.sleep so it never blocks the event loop. Jitter spreads
    out concurrent retries so many agents hitting the same provider's rate limit
    don't re-collide in lockstep.
    """
    attempts = max(1, max_retries + 1)
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = await send()
            raise_for_status_verbose(response)
            return response
        except Exception as exc:  # noqa: BLE001 - re-raised below if not retryable
            last_exc = exc
            if attempt >= attempts or not _is_retryable(exc):
                raise
            delay = min(base_delay * (2 ** (attempt - 1)), _RETRY_DELAY_CAP)
            delay += random.uniform(0, delay)  # full jitter on the computed delay
            log.warning(
                "%s transient failure (attempt %d/%d), retrying in %.2fs: %s",
                label, attempt, attempts, delay, exc,
            )
            await asyncio.sleep(delay)
    # Unreachable: the loop either returns or raises, but keep the type checker happy.
    assert last_exc is not None
    raise last_exc


def raise_for_status_verbose(response: httpx.Response) -> None:
    """Like response.raise_for_status(), but include the URL and response body.

    Providers almost always put the real reason ("model not found", "invalid
    api key", quota exhausted) in the response body, while the default
    HTTPStatusError message carries only the status code. This re-raises with
    that body (truncated) attached so server-side logs are actionable.

    The enhanced message only reaches server logs: build_trace's top-level
    handler logs it and returns a fixed message to the agent, so the response
    body is never forwarded to the agent or frontend.
    """
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        snippet = (exc.response.text or "")[:_ERR_SNIPPET_CAP]
        raise httpx.HTTPStatusError(
            f"{exc} (url={exc.request.url}, response_snippet={snippet})",
            request=exc.request,
            response=exc.response,
        ) from exc


def extract_content(data: Any, getter: Callable[[Any], str], path_desc: str) -> str:
    """Pull the text answer out of a parsed JSON response.

    Some gateways return HTTP 200 with an error-shaped body (e.g. {"error": ...})
    instead of the expected schema. Indexing that blindly raises a bare
    KeyError/IndexError that hides the real cause. This wraps the access and,
    on failure, raises a ValueError carrying a snippet of the actual payload.
    """
    try:
        content = getter(data)
    except (KeyError, IndexError, TypeError) as exc:
        snippet = repr(data)[:_ERR_SNIPPET_CAP]
        raise ValueError(
            f"Unexpected chat response shape (expected {path_desc}); "
            f"body_snippet={snippet}"
        ) from exc
    if content is None:
        snippet = repr(data)[:_ERR_SNIPPET_CAP]
        raise ValueError(
            f"Chat response had null content (expected {path_desc}); "
            f"body_snippet={snippet}"
        )
    return content


class ChatAdapter(ABC):
    """
    Abstract base class for chat adapters.

    Each provider-specific adapter should implement the `chat` method and use
    the configuration dictionary passed in the constructor to perform API calls.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    async def chat(self, prompt: str, **kwargs: Any) -> str:
        """
        Send a prompt to the underlying model and return its textual response.

        Provider-specific adapters can accept additional keyword arguments,
        but they should all return a plain string as the main answer.
        """
        raise NotImplementedError


