"""Offline checks for the browser-supplied LLM pool."""
import sys
from pathlib import Path
from types import SimpleNamespace
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


def test_nonretryable_error_still_uses_fallback_model():
    router = LLMRouter([endpoint("invalid", "model-a"), endpoint("fallback", "model-b", 2)])
    failure = ProviderRequestError("invalid model", retryable=False, status_code=404)
    with patch.object(OpenAICompatibleProvider, "chat", side_effect=[failure, "fallback-ok"]):
        assert router.chat([{"role": "user", "content": "hi"}]) == "fallback-ok"


def test_lower_priority_key_is_not_rotated_ahead_of_primary():
    router = LLMRouter([
        endpoint("primary", "same-model", 1),
        endpoint("secondary", "same-model", 2),
    ])

    def reply(selected, messages, max_tokens):
        return selected.name

    with patch.object(OpenAICompatibleProvider, "chat", side_effect=reply):
        assert router.chat([{"role": "user", "content": "one"}]) == "primary"
        assert router.chat([{"role": "user", "content": "two"}]) == "primary"


def test_duplicate_ids_and_string_booleans_are_validated():
    base = {
        "name": "one", "base_url": "https://example.test/v1",
        "model": "demo", "api_key": "secret", "id": "duplicate",
    }
    try:
        parse_pool([base, {**base, "name": "two"}])
    except ValueError as exc:
        assert "ID" in str(exc)
    else:
        raise AssertionError("duplicate endpoint IDs must be rejected")

    try:
        parse_pool([{**base, "id": "disabled", "enabled": "false"}])
    except ValueError as exc:
        assert "启用" in str(exc)
    else:
        raise AssertionError("string false must disable the endpoint")


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


def test_deepseek_reasoner_uses_completion_tokens_first():
    """deepseek-reasoner 曾被漏判为非推理模型，首发 max_tokens 导致答案为空。

    实测该模型传 max_tokens 时思维链与答案共用预算，思维链吃光后
    finish_reason=length 且 content 为空串；传 max_completion_tokens 则正常。
    """
    from app.llm_router import _reasoning_model

    # 各家推理模型都要被识别出来，避免硬编码列表漏判
    for model in (
        "deepseek-reasoner", "deepseek-r1", "deepseek-r1-0528",
        "deepseek-v3-reasoner", "deepseek/deepseek-reasoner",
        "qwq-32b", "qwen-qwq", "gpt-5", "o1-mini", "o3",
    ):
        assert _reasoning_model(model), f"{model} 应判定为推理模型"

    # 普通模型不能被误判，否则会丢掉 temperature 影响生成质量
    for model in ("deepseek-chat", "deepseek-v3", "gpt-4o", "glm-4-plus"):
        assert not _reasoning_model(model), f"{model} 不应判定为推理模型"

    # 推理模型首发必须是 max_completion_tokens（决定实际请求成败）
    variants = OpenAICompatibleProvider._variants(
        endpoint("ds", "deepseek-reasoner"), [], 3000, True,
    )
    assert "max_completion_tokens" in variants[0]
    assert "max_tokens" not in variants[0]

    # 空内容错误要允许换参数变体重试，否则一次失败就冷却整个端点
    from app.llm_router import EmptyProviderResponseError, _is_parameter_error

    assert _is_parameter_error(EmptyProviderResponseError("模型流式响应为空（finish_reason=length）"))
    # 但不该把普通网络/限流错误当成参数问题
    assert not _is_parameter_error(Exception("Connection error."))
    assert not _is_parameter_error(Exception("Rate limit exceeded"))


def test_content_parts_are_joined():
    content = [
        {"type": "text", "text": "hello "},
        SimpleNamespace(type="text", text="world"),
    ]
    assert OpenAICompatibleProvider._content_text(content) == "hello world"


def test_empty_completion_is_not_success():
    response = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="", model_extra={"reasoning_content": "draft"}),
        finish_reason="length",
    )])
    client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=lambda **params: response),
    ))
    selected = endpoint("empty")
    with patch.object(selected, "client", return_value=client):
        try:
            OpenAICompatibleProvider.chat(selected, [{"role": "user", "content": "hi"}], 8)
        except Exception as exc:
            message = str(exc)
            assert "finish_reason=length" in message
            assert "reasoning_content" in message
        else:
            raise AssertionError("empty completion must not be treated as success")


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"LLM router tests: {len(tests)}/{len(tests)} passed")
