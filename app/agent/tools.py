"""Agent 工具层：把已有能力包装成可校验、可枚举、可审计的工具。

数据流：Agent 原始参数 → Pydantic 校验 → ToolRegistry.call()
      → 具体业务函数 → 统一 ToolResult。

本文件只负责工具协议和适配，不负责 Agent 规划；下一步的 runtime
会根据 side_effect 决定工具是否需要用户确认。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, TypeVar

from pydantic import BaseModel, Field

from app import db

SideEffect = Literal["none", "write"]
ArgsModel = TypeVar("ArgsModel", bound=BaseModel)


class SearchArgs(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=10, ge=1, le=20)


class SemanticSearchArgs(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=10, ge=1, le=20)


class VideoArgs(BaseModel):
    aweme_id: str = Field(min_length=1, max_length=64)


class VideoIdsArgs(BaseModel):
    ids: list[str] = Field(min_length=1, max_length=20)


@dataclass(frozen=True)
class ToolSpec:
    """一个工具的公开契约。"""

    name: str
    description: str
    args_model: type[BaseModel]
    side_effect: SideEffect
    handler: Callable[[BaseModel], dict]

    def schema(self) -> dict:
        """转成 OpenAI-compatible function tool schema。"""
        if hasattr(self.args_model, "model_json_schema"):
            parameters = self.args_model.model_json_schema()
        else:  # Pydantic v1 compatibility
            parameters = self.args_model.schema()
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }


def _result(data: Any = None, *, sources: list[dict] | None = None,
            affected_ids: list[str] | None = None) -> dict:
    """构造成功结果；所有工具都返回相同的外层结构。"""
    return {
        "ok": True,
        "data": data,
        "error": None,
        "affected_ids": affected_ids or [],
        "sources": sources or [],
    }


def _error(code: str, message: str) -> dict:
    return {
        "ok": False,
        "data": None,
        "error": {"code": code, "message": message},
        "affected_ids": [],
        "sources": [],
    }


def _tags(value: str | None) -> list:
    try:
        import json

        parsed = json.loads(value or "[]")
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def _video_dict(row) -> dict:
    """SQLite Row → 稳定的 Agent 数据对象，避免把数据库对象泄露给模型。"""
    return {
        "aweme_id": row["aweme_id"],
        "title": row["title"] or "",
        "tags": _tags(row["tags"]),
        "author": row["author"] or "",
        "share_url": row["share_url"] or "",
        "transcript": row["transcript"] or "",
        "summary": row["summary"] if "summary" in row.keys() else None,
        "category": row["category"] if "category" in row.keys() else None,
    }


def _search(args: SearchArgs) -> dict:
    rows = db.search(args.query, limit=args.limit)
    items = [_video_dict(db.get_video(row["aweme_id"])) for row in rows]
    return _result({"items": items, "count": len(items)}, sources=items)


def _semantic_search(args: SemanticSearchArgs) -> dict:
    from app import retriever

    hits = retriever.hybrid_search(args.question, k_videos=args.limit)
    sources = [
        {
            "aweme_id": h["aweme_id"],
            "title": h["title"],
            "matched_by": h["matched_by"],
        }
        for h in hits
    ]
    return _result({"items": hits, "count": len(hits)}, sources=sources)


def _get_video(args: VideoArgs) -> dict:
    row = db.get_video(args.aweme_id)
    if not row:
        return _error("not_found", f"视频不存在：{args.aweme_id}")
    item = _video_dict(row)
    return _result(item, sources=[item])


def _transcribe(args: VideoIdsArgs) -> dict:
    from app import transcribe

    available = db.get_untranscribed(10**9, ids=args.ids)
    ids = [row["aweme_id"] for row in available]
    if not ids:
        return _result({"processed": 0, "message": "没有待转写的视频"})
    processed = transcribe.run(limit=len(ids), ids=ids)
    return _result(
        {"processed": processed, "requested": len(ids)},
        affected_ids=ids,
    )


def _summarize(args: VideoIdsArgs) -> dict:
    from app import llm

    processed = 0
    skipped: list[dict] = []
    affected: list[str] = []
    for aweme_id in args.ids:
        row = db.get_video(aweme_id)
        if not row:
            skipped.append({"aweme_id": aweme_id, "reason": "not_found"})
            continue
        transcript = row["transcript"] or ""
        if not transcript or transcript.startswith("【"):
            skipped.append({"aweme_id": aweme_id, "reason": "no_transcript"})
            continue
        if row["summary"]:
            skipped.append({"aweme_id": aweme_id, "reason": "already_summarized"})
            continue
        summary = llm.summarize(row["title"], row["author"], transcript)
        db.set_summary(aweme_id, summary)
        processed += 1
        affected.append(aweme_id)
    return _result(
        {"processed": processed, "skipped": skipped},
        affected_ids=affected,
    )


def _compare(args: VideoIdsArgs) -> dict:
    items = []
    sources = []
    for aweme_id in args.ids:
        row = db.get_video(aweme_id)
        if not row:
            continue
        item = _video_dict(row)
        # 比较工具只准备证据，不在这里调用 LLM，避免工具职责和最终回答混在一起。
        body = item["transcript"] or item["summary"] or item["title"]
        items.append({
            "aweme_id": item["aweme_id"],
            "title": item["title"],
            "author": item["author"],
            "evidence": body[:5000],
        })
        sources.append({"aweme_id": item["aweme_id"], "title": item["title"]})
    return _result({"items": items, "count": len(items)}, sources=sources)


class ToolRegistry:
    """工具注册表：统一列举工具、生成模型 schema、校验并执行调用。"""

    def __init__(self, specs: list[ToolSpec]):
        self._specs = {spec.name: spec for spec in specs}

    def list(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def schemas(self) -> list[dict]:
        return [spec.schema() for spec in self._specs.values()]

    def call(self, name: str, raw_args: dict, *, allow_write: bool = False) -> dict:
        """校验并调用工具；写工具必须显式传入 allow_write=True。"""
        spec = self.get(name)
        if spec is None:
            return _error("unknown_tool", f"未注册的工具：{name}")
        if spec.side_effect == "write" and not allow_write:
            return {
                **_error("approval_required", f"工具 {name} 需要用户批准"),
                "tool": name,
                "side_effect": spec.side_effect,
            }
        if not isinstance(raw_args, dict):
            return _error("invalid_arguments", "工具参数必须是 JSON 对象")
        try:
            args = spec.args_model.model_validate(raw_args)
        except AttributeError:  # Pydantic v1 compatibility
            try:
                args = spec.args_model.parse_obj(raw_args)
            except Exception as exc:
                return _error("invalid_arguments", str(exc)[:300])
        except Exception as exc:
            return _error("invalid_arguments", str(exc)[:300])
        try:
            result = spec.handler(args)
            result["tool"] = name
            result["side_effect"] = spec.side_effect
            return result
        except Exception as exc:
            return {
                **_error("execution_error", str(exc)[:300]),
                "tool": name,
                "side_effect": spec.side_effect,
            }


TOOLS = ToolRegistry([
    ToolSpec("search_favorites", "搜索收藏视频的标题、标签、作者和转写内容。",
             SearchArgs, "none", _search),
    ToolSpec("search_semantic", "用语义和关键词混合检索相关收藏视频。",
             SemanticSearchArgs, "none", _semantic_search),
    ToolSpec("get_video", "读取一个收藏视频的完整元数据、转写和概要。",
             VideoArgs, "none", _get_video),
    ToolSpec("transcribe_videos", "转写指定视频；这是会修改本地数据的操作。",
             VideoIdsArgs, "write", _transcribe),
    ToolSpec("summarize_videos", "为指定视频生成概要；这是会修改本地数据的操作。",
             VideoIdsArgs, "write", _summarize),
    ToolSpec("compare_videos", "读取多个视频的证据，供 Agent 比较观点。",
             VideoIdsArgs, "none", _compare),
])


__all__ = ["TOOLS", "ToolRegistry", "ToolSpec"]
