"""离线评估 Agent 的关键行为。

用法：python scripts/eval_agent.py
该脚本只使用假模型和假工具，不调用网络，也不修改正式数据库。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent.runtime import AgentRuntime
from app.agent.tools import ToolRegistry, ToolSpec, VideoIdsArgs


class EmptyArgs(BaseModel):
    """测试工具不需要参数。"""


def scripted_chat(responses: list[str]):
    index = 0

    def chat(messages, max_tokens=2000):
        nonlocal index
        if index >= len(responses):
            raise AssertionError("scripted model ran out of responses")
        response = responses[index]
        index += 1
        return response

    return chat


def registry_for(handler, side_effect="none") -> ToolRegistry:
    return ToolRegistry([
        ToolSpec("lookup", "offline test tool", EmptyArgs, side_effect, handler),
    ])


def test_plan_dependency() -> None:
    registry = registry_for(lambda args: {"ok": True, "data": {}, "sources": []})
    chat = scripted_chat([
        '{"type":"plan","tasks":[{"id":"collect","title":"collect","depends_on":[]},{"id":"compare","title":"compare","depends_on":["collect"]}]}',
        '{"type":"tool_call","tool":"lookup","task_id":"compare","args":{}}',
        '{"type":"tool_call","tool":"lookup","task_id":"collect","args":{}}',
        '{"type":"tool_call","tool":"lookup","task_id":"compare","args":{}}',
        '{"type":"final","answer":"done","citations":[]}',
    ])
    result = AgentRuntime(chat=chat, registry=registry).run("compare")
    assert result.status == "completed"
    assert [task["status"] for task in result.plan] == ["completed", "completed"]
    assert result.steps[0]["type"] == "plan"
    assert result.steps[1]["status"] == "rejected"


def test_retry() -> None:
    attempts = []

    def flaky(args):
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("temporary")
        return {"ok": True, "data": {}, "sources": []}

    registry = registry_for(flaky)
    result = AgentRuntime(
        chat=scripted_chat([
            '{"type":"tool_call","tool":"lookup","args":{}}',
            '{"type":"final","answer":"done","citations":[]}',
        ]),
        registry=registry,
    ).run("retry")
    assert result.status == "completed"
    assert len(attempts) == 3
    assert result.steps[0]["retries"] == 2


def test_replan() -> None:
    attempts = []

    def failing_once(args):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("temporary")
        return {"ok": True, "data": {}, "sources": []}

    registry = registry_for(failing_once)
    result = AgentRuntime(
        chat=scripted_chat([
            '{"type":"plan","tasks":[{"id":"old","title":"old route","depends_on":[]}]}',
            '{"type":"tool_call","tool":"lookup","task_id":"old","args":{}}',
            '{"type":"plan","reason":"old route failed","tasks":[{"id":"new","title":"fallback route","depends_on":[]}]}',
            '{"type":"tool_call","tool":"lookup","task_id":"new","args":{}}',
            '{"type":"final","answer":"done","citations":[]}',
        ]),
        registry=registry,
        max_tool_retries=0,
    ).run("replan")
    assert result.status == "completed"
    assert [step["type"] for step in result.steps] == ["plan", "tool_call", "replan", "tool_call", "final"]
    assert result.plan[0]["id"] == "new"


def test_approval_gate() -> None:
    writes = []

    def write(args):
        writes.append(1)
        return {"ok": True, "data": {}, "sources": []}

    registry = registry_for(write, side_effect="write")
    execution = AgentRuntime(
        chat=scripted_chat([
            '{"type":"tool_call","tool":"lookup","args":{}}',
            '{"type":"final","answer":"done","citations":[]}',
        ]),
        registry=registry,
    ).start("write")
    waiting = execution.advance()
    assert waiting.status == "waiting_approval"
    assert writes == []
    completed = execution.approve(True)
    assert completed.status == "completed"
    assert len(writes) == 1


def test_timeout() -> None:
    registry = registry_for(lambda args: {"ok": True, "data": {}, "sources": []})

    def slow_chat(messages, max_tokens=2000):
        time.sleep(0.01)
        return '{"type":"final","answer":"done","citations":[]}'

    result = AgentRuntime(
        chat=slow_chat,
        registry=registry,
        timeout_seconds=0.001,
    ).run("timeout")
    assert result.status == "paused"
    assert result.error


def test_streaming_answer() -> None:
    chunks = [
        '{"type":"final","answer":"你',
        '好\\n世',
        '界","citations":[]}',
    ]

    def stream_chat(messages, on_delta, max_tokens=2000):
        for chunk in chunks:
            on_delta(chunk)
        return "".join(chunks)

    deltas = []
    execution = AgentRuntime(
        stream_chat=stream_chat,
        registry=registry_for(lambda args: {"ok": True, "data": {}, "sources": []}),
    ).start("stream")
    execution.on_text_delta = lambda delta, answer: deltas.append(delta)
    result = execution.advance()
    assert result.status == "completed"
    assert result.answer == "你好\n世界"
    assert "".join(deltas) == result.answer


def test_args_alias() -> None:
    seen = {}

    def pick(args):
        seen["ids"] = args.ids
        return {"ok": True, "data": {}, "sources": []}

    registry = ToolRegistry([
        ToolSpec("pick", "offline test tool", VideoIdsArgs, "none", pick),
    ])
    result = registry.call("pick", {"aweme_ids": ["a", "b"]})
    assert result["ok"], result["error"]
    assert seen["ids"] == ["a", "b"]


CASES = {
    "plan": test_plan_dependency,
    "retry": test_retry,
    "replan": test_replan,
    "approval": test_approval_gate,
    "timeout": test_timeout,
    "stream": test_streaming_answer,
    "args_alias": test_args_alias,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="离线评估 Agent Runtime")
    parser.add_argument("--case", choices=["all", *CASES], default="all")
    args = parser.parse_args()
    cases = CASES if args.case == "all" else {args.case: CASES[args.case]}
    passed = 0
    for name, case in cases.items():
        try:
            case()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as exc:
            print(f"[FAIL] {name}: {exc}")
    print(f"Agent eval: {passed}/{len(cases)} passed")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
