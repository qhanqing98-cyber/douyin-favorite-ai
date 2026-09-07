"""LLM 封装（OpenAI 兼容协议）→ 概要生成 + 检索问答。

配置全部来自 .env（项目根目录），绝不硬编码密钥：
    LLM_API_KEY=你的密钥
    LLM_BASE_URL=https://api.deepseek.com/v1
    LLM_MODEL=deepseek-chat
兼容任何 OpenAI 协议的服务（混元/Ollama/Qwen），改 .env 三个值即可换。
"""
import os
from pathlib import Path

from openai import OpenAI

PROJECT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT / ".env"

_client = None  # 进程内复用


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


def _model() -> str:
    _load_env()
    return os.environ.get("LLM_MODEL", "deepseek-chat")


def _chat(messages: list[dict], max_tokens: int = 1500) -> str:
    resp = _get_client().chat.completions.create(
        model=_model(), messages=messages, max_tokens=max_tokens, temperature=0.3
    )
    return resp.choices[0].message.content.strip()


def summarize(title: str, author: str, transcript: str) -> str:
    """把一条视频的转写压成概要。截断 8000 字防超上下文。"""
    prompt = (
        "请为下面这个抖音视频的语音转写内容生成概要，要求：\n"
        "1. 用 3-5 句话概括核心内容\n"
        "2. 如果是知识类视频，列出讲到的关键要点（用短横线列表）\n"
        "3. 不要编造转写里没有的信息\n\n"
        f"标题：{title}\n作者：{author}\n转写内容：\n{transcript[:8000]}"
    )
    return _chat([{"role": "user", "content": prompt}])


def answer(question: str, contexts: list[dict]) -> str:
    """检索问答：把命中的视频转写作为上下文，让模型"看着原文"回答。

    contexts: [{title, author, aweme_id, transcript}]，转写各截 3000 字防爆上下文。
    """
    blocks = []
    for i, c in enumerate(contexts, 1):
        blocks.append(
            f"[来源{i}] 标题:{c['title']} 作者:{c['author']} id:{c['aweme_id']}\n"
            f"{(c['transcript'] or '')[:3000]}"
        )
    system = (
        "你是用户的抖音收藏知识库助手。根据提供的视频转写内容回答问题，"
        "回答末尾用 [来源N] 标注引用了哪些视频。"
        "如果转写内容不足以回答，直接说明，不要编造。"
    )
    user = f"以下是相关视频的转写内容：\n\n" + "\n\n".join(blocks) + f"\n\n问题：{question}"
    return _chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=2000,
    )
