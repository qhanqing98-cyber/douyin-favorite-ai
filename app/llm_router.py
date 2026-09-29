"""Web supplied LLM pool and OpenAI-compatible routing.

The browser owns the durable configuration.  This module only keeps a copy in
memory for the lifetime of a request/Agent run; credentials are never part of
the persisted Agent state.
"""
from __future__ import annotations

import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable
from urllib.parse import urlparse

from openai import OpenAI


MAX_POOL_SIZE = 10
POOL_TTL_SECONDS = 30 * 60
RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


class LLMRouterError(RuntimeError):
    """A safe, user-facing routing error."""


class ProviderRequestError(LLMRouterError):
    def __init__(self, message: str, *, retryable: bool, status_code: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


class EmptyProviderResponseError(RuntimeError):
    """Provider 接受了请求但没有给出可消费的正文。"""


def _status_code(exc: Exception) -> int | None:
    value = getattr(exc, "status_code", None)
    return value if isinstance(value, int) else None


def _error_text(exc: Exception) -> str:
    text = str(exc).strip().replace("\n", " ")
    return text[:280] or exc.__class__.__name__


def _is_parameter_error(exc: Exception) -> bool:
    """是否是「这个参数服务端不认」的错误——只有这类错误才值得换参数变体重试。

    用「拒绝语气」短语判断，而不是裸参数名：错误文案里出现 max_tokens 这个
    词不代表它是参数不兼容（例如“模型流式响应为空（finish_reason=length）”
    也可能被上游包装进含 max_tokens 的说明），按裸词匹配会导致该降级时不降级。
    """
    if isinstance(exc, EmptyProviderResponseError):
        # 空内容不是参数问题，但值得换参数变体重试（正是 max_tokens 预算被
        # 思维链吃光的典型症状），所以这里显式放行给 _variants 继续下一个变体。
        return True
    text = _error_text(exc).lower()
    return any(token in text for token in (
        "unsupported parameter", "unknown parameter", "unrecognized request argument",
        "unrecognized parameter", "invalid parameter", "not supported",
        "does not support", "unsupported value",
    ))


def classify_provider_error(exc: Exception) -> ProviderRequestError:
    status = _status_code(exc)
    retryable = status in RETRYABLE_STATUS_CODES or status is None
    if _is_parameter_error(exc):
        retryable = False
    return ProviderRequestError(_error_text(exc), retryable=retryable, status_code=status)


def _reasoning_model(model: str) -> bool:
    """判断模型是否为「思维链与答案分预算」的推理模型。

    推理模型必须优先用 max_completion_tokens：实测 deepseek-reasoner 传
    max_tokens 时思维链会吃光整个预算，finish_reason=length 且 content 为空串，
    表现为“模型返回空内容”。传 max_completion_tokens 时两者独立计费，答案稳定输出。

    用模式匹配而非硬编码列表，避免新推理模型（r1/qwq 等）漏判后静默退化。
    """
    normalized = model.lower().split("/")[-1].strip()
    if normalized.startswith(("gpt-5", "o1", "o3", "o4")):
        return True
    # deepseek-reasoner / deepseek-r1 / deepseek-r1-0528 / deepseek-v3-reasoner
    if normalized.startswith("deepseek") and (
        "reasoner" in normalized or re.search(r"-r1(?:[-.]|$)", normalized)
    ):
        return True
    # qwq / qwq-32b / qwen-qwq、以及常见 reasoner/thinking 命名
    if "qwq" in normalized or "reasoner" in normalized or "-thinking" in normalized:
        return True
    return False


def _enabled_value(value) -> bool:
    """只接受明确的布尔值/常见布尔字符串，避免 bool("false") == True。"""
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError("enabled 必须是布尔值")


@dataclass
class PoolEndpoint:
    name: str
    base_url: str
    model: str
    api_key: str = field(repr=False)
    endpoint_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    enabled: bool = True
    priority: int = 1
    weight: int = 1
    cooldown_until: float = 0.0
    failures: int = 0
    _client: OpenAI | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_payload(cls, payload: dict) -> "PoolEndpoint":
        name = str(payload.get("name") or "未命名模型").strip()
        base_url = str(payload.get("base_url") or "").strip().rstrip("/")
        model = str(payload.get("model") or "").strip()
        api_key = str(payload.get("api_key") or "").strip()
        parsed = urlparse(base_url)
        if not name or len(name) > 80:
            raise ValueError("模型配置名称长度必须在 1 到 80 个字符之间")
        if not api_key or len(api_key) > 500:
            raise ValueError("API Key 不能为空且不能超过 500 个字符")
        if not model or len(model) > 160:
            raise ValueError("模型名称不能为空且不能超过 160 个字符")
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("API 地址必须是有效的 HTTP(S) 地址，且不能包含账号或密码")
        if parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("远程 API 地址必须使用 HTTPS")
        try:
            priority = max(1, min(999, int(payload.get("priority", 1))))
            weight = max(1, min(100, int(payload.get("weight", 1))))
        except (TypeError, ValueError) as exc:
            raise ValueError("优先级和权重必须是数字") from exc
        endpoint_id = str(payload.get("id") or payload.get("endpoint_id") or uuid.uuid4().hex)
        if len(endpoint_id) > 80:
            raise ValueError("模型配置 ID 过长")
        return cls(
            name=name, base_url=base_url, model=model, api_key=api_key,
            endpoint_id=endpoint_id, enabled=_enabled_value(payload.get("enabled")),
            priority=priority, weight=weight,
        )

    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def safe_info(self) -> dict:
        return {
            "id": self.endpoint_id, "name": self.name, "model": self.model,
            "base_url": self.base_url, "enabled": self.enabled,
            "priority": self.priority, "weight": self.weight,
            "key_last4": self.api_key[-4:],
            "cooldown_until": self.cooldown_until,
            "failures": self.failures,
        }


def parse_pool(payload: list[dict] | None) -> list[PoolEndpoint]:
    if not isinstance(payload, list) or not payload:
        raise ValueError("请至少配置一个可用的模型 API")
    if len(payload) > MAX_POOL_SIZE:
        raise ValueError(f"模型池最多支持 {MAX_POOL_SIZE} 个配置")
    endpoints = [PoolEndpoint.from_payload(item) for item in payload]
    endpoint_ids = [item.endpoint_id for item in endpoints]
    if len(endpoint_ids) != len(set(endpoint_ids)):
        raise ValueError("模型池中的配置 ID 不能重复")
    if not any(item.enabled for item in endpoints):
        raise ValueError("模型池中至少要启用一个配置")
    return endpoints


class OpenAICompatibleProvider:
    """OpenAI Chat Completions compatible provider with parameter fallback."""

    @classmethod
    def _content_text(cls, value) -> str:
        """兼容字符串和部分网关返回的 content parts 数组。"""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "".join(cls._content_text(item) for item in value)
        if isinstance(value, dict):
            for key in ("text", "content", "output_text"):
                if key in value:
                    return cls._content_text(value[key])
            return ""
        for attr in ("text", "content", "output_text"):
            nested = getattr(value, attr, None)
            if nested is not None and nested is not value:
                return cls._content_text(nested)
        return ""

    @staticmethod
    def _empty_message(choice, message) -> str:
        finish_reason = getattr(choice, "finish_reason", None) or "unknown"
        extra = getattr(message, "model_extra", None) or {}
        extra_fields = sorted(
            str(key) for key, value in extra.items()
            if value not in (None, "", [], {})
        )
        hint = f"，附加字段={','.join(extra_fields)}" if extra_fields else ""
        return f"模型返回空内容（finish_reason={finish_reason}{hint}）"

    @staticmethod
    def _variants(endpoint: PoolEndpoint, messages: list[dict], max_tokens: int,
                  stream: bool) -> list[dict]:
        """按「最可能成功」排序的参数变体，逐个尝试直到有一个被服务端接受。

        两组预算字段的区别（实测 deepseek-reasoner）：
        - max_completion_tokens：思维链与答案**分开**计费，答案稳定输出 → 推理模型首选
        - max_tokens：思维链与答案**共用**预算，思维链吃光后 finish_reason=length
          且 content 为空串 → 推理模型必须靠后，仅作兼容性兜底
        """
        base = {"model": endpoint.model, "messages": messages, "stream": stream}
        memory_cap = {**base, "max_completion_tokens": max_tokens}
        token_cap = {**base, "max_tokens": max_tokens}
        token_cap_no_temp = {**base, "max_tokens": max_tokens, "temperature": 0.3}
        if _reasoning_model(endpoint.model):
            # 推理模型：正确的预算字段优先，再退到只换字段名，最后才试旧字段。
            return [memory_cap, {**base, "max_completion_tokens": max_tokens, "temperature": 1}, token_cap]
        return [token_cap_no_temp, memory_cap, token_cap]

    @classmethod
    def chat(cls, endpoint: PoolEndpoint, messages: list[dict], max_tokens: int) -> str:
        last: Exception | None = None
        for params in cls._variants(endpoint, messages, max_tokens, False):
            try:
                response = endpoint.client().chat.completions.create(**params)
                if not response.choices:
                    raise EmptyProviderResponseError("模型返回空 choices")
                choice = response.choices[0]
                message = choice.message
                content = cls._content_text(getattr(message, "content", None)).strip()
                if not content:
                    raise EmptyProviderResponseError(cls._empty_message(choice, message))
                return content
            except Exception as exc:
                last = exc
                if not _is_parameter_error(exc):
                    raise
        assert last is not None
        raise last

    @classmethod
    def chat_stream(cls, endpoint: PoolEndpoint, messages: list[dict], max_tokens: int,
                    on_delta: Callable[[str], None]) -> str:
        last: Exception | None = None
        for params in cls._variants(endpoint, messages, max_tokens, True):
            parts: list[str] = []
            finish_reason = None
            try:
                stream = endpoint.client().chat.completions.create(**params)
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    finish_reason = getattr(choice, "finish_reason", None) or finish_reason
                    delta = cls._content_text(getattr(choice.delta, "content", None))
                    if delta:
                        parts.append(delta)
                        on_delta(delta)
                result = "".join(parts).strip()
                if not result:
                    raise EmptyProviderResponseError(
                        f"模型流式响应为空（finish_reason={finish_reason or 'unknown'}）"
                    )
                return result
            except Exception as exc:
                last = exc
                # Once text was emitted, retrying would duplicate user-visible text.
                if parts:
                    setattr(exc, "_llm_stream_started", True)
                if parts or not _is_parameter_error(exc):
                    raise
        assert last is not None
        raise last


class LLMRouter:
    """Priority-first pool router with per-endpoint cooldown and round-robin."""

    def __init__(self, endpoints: list[PoolEndpoint], ttl_seconds: float = POOL_TTL_SECONDS):
        self.endpoints = endpoints
        self.created_at = time.monotonic()
        self.expires_at = self.created_at + ttl_seconds
        self._cursor: dict[tuple[str, str], int] = {}
        self._lock = threading.RLock()
        self.last_used: dict | None = None

    def expired(self) -> bool:
        return time.monotonic() >= self.expires_at

    def _groups(self) -> list[list[PoolEndpoint]]:
        groups: dict[str, list[PoolEndpoint]] = {}
        for endpoint in self.endpoints:
            if endpoint.enabled:
                groups.setdefault(endpoint.model, []).append(endpoint)
        return [
            sorted(items, key=lambda item: (item.priority, item.endpoint_id))
            for items in sorted(
                groups.values(),
                key=lambda items: (min(item.priority for item in items), items[0].model),
            )
        ]

    def _ordered(self, group: list[PoolEndpoint]) -> list[PoolEndpoint]:
        result: list[PoolEndpoint] = []
        # 优先级是硬边界：只有当前优先级全部失败/冷却后，才尝试下一层。
        # 权重轮询只发生在同模型、同优先级的 Key 之间。
        priorities = sorted({endpoint.priority for endpoint in group})
        for priority in priorities:
            tier = [endpoint for endpoint in group if endpoint.priority == priority]
            weighted = [endpoint for endpoint in tier for _ in range(endpoint.weight)]
            if not weighted:
                continue
            key = (group[0].model, str(priority))
            with self._lock:
                start = self._cursor.get(key, 0) % len(weighted)
                self._cursor[key] = (start + 1) % len(weighted)
            seen: set[str] = set()
            for offset in range(len(weighted)):
                endpoint = weighted[(start + offset) % len(weighted)]
                if endpoint.endpoint_id not in seen:
                    result.append(endpoint)
                    seen.add(endpoint.endpoint_id)
        return result

    @staticmethod
    def _available(endpoint: PoolEndpoint) -> bool:
        return endpoint.cooldown_until <= time.monotonic()

    def _mark_failure(self, endpoint: PoolEndpoint, error: ProviderRequestError) -> None:
        with self._lock:
            endpoint.failures += 1
            if error.retryable:
                endpoint.cooldown_until = time.monotonic() + min(60.0, 2 ** min(endpoint.failures, 5))
            else:
                endpoint.cooldown_until = time.monotonic() + 300.0

    def _mark_success(self, endpoint: PoolEndpoint) -> None:
        with self._lock:
            endpoint.failures = 0
            endpoint.cooldown_until = 0.0
            self.last_used = endpoint.safe_info()

    def _call(self, endpoint: PoolEndpoint, messages: list[dict], max_tokens: int,
              on_delta: Callable[[str], None] | None) -> str:
        if on_delta is None:
            return OpenAICompatibleProvider.chat(endpoint, messages, max_tokens)
        return OpenAICompatibleProvider.chat_stream(endpoint, messages, max_tokens, on_delta)

    def _run(self, messages: list[dict], max_tokens: int,
             on_delta: Callable[[str], None] | None) -> str:
        if self.expired():
            raise LLMRouterError("本次任务的模型配置已过期，请重新提交模型池")
        errors: list[str] = []
        for group in self._groups():
            group_errors = []
            for endpoint in self._ordered(group):
                if not self._available(endpoint):
                    continue
                try:
                    result = self._call(endpoint, messages, max_tokens, on_delta)
                    self._mark_success(endpoint)
                    return result
                except Exception as exc:
                    error = classify_provider_error(exc)
                    safe_message = str(error).replace(endpoint.api_key, "***")
                    if safe_message != str(error):
                        error = ProviderRequestError(
                            safe_message, retryable=error.retryable, status_code=error.status_code,
                        )
                    self._mark_failure(endpoint, error)
                    group_errors.append(f"{endpoint.name}: {error}")
                    if on_delta is not None and getattr(exc, "_llm_stream_started", False):
                        raise error from exc
            if group_errors:
                errors.extend(group_errors)
        if errors:
            raise LLMRouterError("所有可用模型配置均失败：" + "；".join(errors)[:900])
        raise LLMRouterError("模型池中的配置都在冷却中，请稍后重试")

    def chat(self, messages: list[dict], max_tokens: int = 1500) -> str:
        return self._run(messages, max_tokens, None)

    def chat_stream(self, messages: list[dict], on_delta: Callable[[str], None],
                    max_tokens: int = 1500) -> str:
        return self._run(messages, max_tokens, on_delta)


__all__ = [
    "LLMRouter", "LLMRouterError", "PoolEndpoint", "ProviderRequestError",
    "MAX_POOL_SIZE", "POOL_TTL_SECONDS", "parse_pool",
]
