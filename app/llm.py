"""LLM 封装（OpenAI 兼容协议）→ 概要生成 + 检索问答。

配置全部来自 .env（项目根目录），绝不硬编码密钥：
    LLM_API_KEY=你的密钥
    LLM_BASE_URL=https://api.deepseek.com/v1
    LLM_MODEL=deepseek-chat
兼容任何 OpenAI 协议的服务（混元/Ollama/Qwen），改 .env 三个值即可换。
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from openai import OpenAI

PROJECT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT / ".env"

_client = None  # 进程内复用


@dataclass
class LLMSettings:
    """一次 Web 会话提供的模型配置；密钥只保存在内存，不参与 repr。"""
    api_key: str = field(repr=False)
    base_url: str
    model: str
    _client: OpenAI | None = field(default=None, init=False, repr=False)

    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client


def _load_env() -> None:
    """极简 .env 解析：KEY=VALUE 逐行，# 开头是注释，不覆盖已有环境变量。"""
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _load_env()
        api_key = os.environ.get("LLM_API_KEY")
        if not api_key:
            raise RuntimeError(
                "缺少 LLM_API_KEY：请在项目根目录 .env 文件里配置"
                "（参考 .env.example）"
            )
        _client = OpenAI(
            api_key=api_key,
            base_url=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1"),
        )
    return _client


def _model(settings: LLMSettings | None = None) -> str:
    if settings is not None:
        return settings.model
    _load_env()
    return os.environ.get("LLM_MODEL", "deepseek-chat")


def _chat(messages: list[dict], max_tokens: int = 1500,
          settings: LLMSettings | None = None) -> str:
    client = settings.client() if settings is not None else _get_client()
    resp = client.chat.completions.create(
        model=_model(settings), messages=messages, max_tokens=max_tokens, temperature=0.3
    )
    # 推理类模型或部分兼容服务可能返回 content=None，直接 .strip() 会抛 AttributeError。
    content = resp.choices[0].message.content or ""
    return content.strip()


def _chat_stream(messages: list[dict], on_delta: Callable[[str], None],
                 max_tokens: int = 1500,
                 settings: LLMSettings | None = None) -> str:
    """流式读取 OpenAI 兼容接口，同时返回完整文本供上层解析与持久化。"""
    client = settings.client() if settings is not None else _get_client()
    stream = client.chat.completions.create(
        model=_model(settings), messages=messages, max_tokens=max_tokens,
        temperature=0.3, stream=True,
    )
    parts: list[str] = []
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content or ""
        if not delta:
            continue
        parts.append(delta)
        on_delta(delta)
    return "".join(parts).strip()


def summarize(title: str, author: str, transcript: str,
              settings: LLMSettings | None = None) -> str:
    """把一条视频的转写压成概要。截断 8000 字防超上下文。"""
    prompt = (
        "请为下面这个抖音视频的语音转写内容生成概要，要求：\n"
        "1. 用 3-5 句话概括核心内容\n"
        "2. 如果是知识类视频，列出讲到的关键要点（用短横线列表）\n"
        "3. 不要编造转写里没有的信息\n\n"
        f"标题：{title}\n作者：{author}\n转写内容：\n{transcript[:8000]}"
    )
    return _chat([{"role": "user", "content": prompt}], settings=settings)


def answer(question: str, contexts: list[dict],
           settings: LLMSettings | None = None) -> str:
    """检索问答：把命中的视频转写作为上下文，让模型"看着原文"回答。

    contexts: [{title, author, aweme_id, transcript, excerpt}]。
    其中 excerpt 是检索层给出的「命中片段」（相关窗口/概要），优先使用；
    没有 excerpt 时退回转写开头 3000 字。
    """
    blocks = []
    for i, c in enumerate(contexts, 1):
        body = (c.get("excerpt") or "").strip() or (c["transcript"] or "")[:3000]
        blocks.append(
            f"[来源{i}] 标题:{c['title']} 作者:{c['author']} id:{c['aweme_id']}\n{body}"
        )
    system = (
        "你是用户的抖音收藏知识库助手。根据提供的视频内容摘录回答问题，"
        "回答末尾用 [来源N] 标注引用了哪些视频。"
        "如果摘录内容不足以回答，直接说明，不要编造。"
    )
    user = f"以下是相关视频的内容摘录：\n\n" + "\n\n".join(blocks) + f"\n\n问题：{question}"
    return _chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=2000,
        settings=settings,
    )
