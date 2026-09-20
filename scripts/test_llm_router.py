"""Offline checks for the browser-supplied LLM pool."""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.llm_router import (
    LLMRouter,
    LLMRouterError,
    OpenAICompatibleProvider,
    PoolEndpoint,
    ProviderRequestError,
    parse_pool,
)


def endpoint(name: str, model: str = "same-model", priority: int = 1) -> PoolEndpoint:
    return PoolEndpoint(
        name=name, base_url="https://example.test/v1", model=model,
        api_key=f"secret-{name}", endpoint_id=name, priority=priority,
    )


def test_pool_validation_and_redaction():
    items = parse_pool([{
        "name": "primary", "base_url": "https://example.test/v1",
        "model": "demo", "api_key": "secret-value",
    }])
    assert len(items) == 1
    assert "secret-value" not in repr(items[0])
    assert "secret-value" not in str(items[0].safe_info())


def test_same_model_failover():
    router = LLMRouter([endpoint("first"), endpoint("second")])
    failure = ProviderRequestError("rate limited", retryable=True, status_code=429)
    with patch.object(OpenAICompatibleProvider, "chat", side_effect=[failure, "ok"]):
        assert router.chat([{"role": "user", "content": "hi"}]) == "ok"
    assert router.last_used["name"] == "second"


def test_fallback_model_group():
    router = LLMRouter([endpoint("primary", "model-a"), endpoint("fallback", "model-b", 2)])
    failure = ProviderRequestError("upstream unavailable", retryable=True, status_code=503)
    with patch.object(OpenAICompatibleProvider, "chat", side_effect=[failure, "fallback-ok"]):
        assert router.chat([{"role": "user", "content": "hi"}]) == "fallback-ok"


def test_all_cooling_is_safe_error():
    router = LLMRouter([endpoint("only")])
    failure = ProviderRequestError("rate limited", retryable=True, status_code=429)
    with patch.object(OpenAICompatibleProvider, "chat", side_effect=failure):
        try:
            router.chat([{"role": "user", "content": "hi"}])
        except LLMRouterError:
            pass
        else:
            raise AssertionError("expected router error")
    with patch.object(OpenAICompatibleProvider, "chat") as chat:
        try:
            router.chat([{"role": "user", "content": "hi"}])
        except LLMRouterError as exc:
            assert "冷却" in str(exc)
        else:
            raise AssertionError("expected cooldown error")
        chat.assert_not_called()


def test_reasoning_model_parameter_order():
    variants = OpenAICompatibleProvider._variants(endpoint("reasoning", "gpt-5-mini"), [], 8, False)
    assert "max_completion_tokens" in variants[0]
    assert "temperature" not in variants[0]


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"LLM router tests: {len(tests)}/{len(tests)} passed")
