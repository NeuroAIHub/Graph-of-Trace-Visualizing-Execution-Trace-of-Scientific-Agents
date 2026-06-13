import httpx
import pytest
import asyncio

from Monitor.adapter.base_adapter import (
    extract_content,
    post_with_retry,
    raise_for_status_verbose,
)
from Monitor.adapter.registry import ADAPTERS
from Monitor.adapter.anthropic_adapter import AnthropicAdapter


def test_anthropic_registered():
    assert ADAPTERS["anthropic"] is AnthropicAdapter


def test_extract_content_ok():
    data = {"choices": [{"message": {"content": "hello"}}]}
    assert extract_content(data, lambda d: d["choices"][0]["message"]["content"], "x") == "hello"


def test_extract_content_error_body_surfaced():
    # HTTP 200 with an error-shaped body -> ValueError carrying the payload, not KeyError
    data = {"error": {"message": "model not found"}}
    with pytest.raises(ValueError) as ei:
        extract_content(data, lambda d: d["choices"][0]["message"]["content"], "choices[0].message.content")
    assert "model not found" in str(ei.value)


def test_raise_for_status_verbose_includes_body():
    req = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
    resp = httpx.Response(404, request=req, text='{"error":"no such model xyz"}')
    with pytest.raises(httpx.HTTPStatusError) as ei:
        raise_for_status_verbose(resp)
    msg = str(ei.value)
    assert "no such model xyz" in msg
    assert "api.example.com" in msg


def test_anthropic_extract_system():
    a = AnthropicAdapter({"api_key": "k", "model": "m"})
    system, msgs = a._extract_system(
        [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hi"},
        ]
    )
    assert system == "you are helpful"
    assert msgs == [{"role": "user", "content": "hi"}]


def _resp(status: int) -> httpx.Response:
    req = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
    return httpx.Response(status, request=req, text="{}")


def test_retry_succeeds_after_transient_429():
    calls = {"n": 0}

    async def send():
        calls["n"] += 1
        return _resp(429) if calls["n"] < 3 else _resp(200)

    resp = asyncio.run(post_with_retry(send, max_retries=2, base_delay=0.0))
    assert resp.status_code == 200
    assert calls["n"] == 3  # two retries then success


def test_retry_exhausts_and_raises():
    calls = {"n": 0}

    async def send():
        calls["n"] += 1
        return _resp(503)

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(post_with_retry(send, max_retries=2, base_delay=0.0))
    assert calls["n"] == 3  # 1 + 2 retries, then gives up


def test_no_retry_on_auth_error():
    calls = {"n": 0}

    async def send():
        calls["n"] += 1
        return _resp(401)

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(post_with_retry(send, max_retries=2, base_delay=0.0))
    assert calls["n"] == 1  # 401 is not retryable


def test_retry_on_timeout_then_success():
    calls = {"n": 0}

    async def send():
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectTimeout("boom")
        return _resp(200)

    resp = asyncio.run(post_with_retry(send, max_retries=2, base_delay=0.0))
    assert resp.status_code == 200
    assert calls["n"] == 2

