"""离线评估 Agent 的关键行为。

用法：python scripts/eval_agent.py
该脚本只使用假模型和假工具，不调用网络，也不修改正式数据库。
"""
from __future__ import annotations

import argparse
import sys
import threading
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


def test_stuck_plan_replan() -> None:
    """计划无可执行任务时，重规划必须被接受，否则会空转到步数耗尽。"""
    def handler(args):
        return {"ok": True, "data": {}, "sources": []}

    registry = registry_for(handler)
    result = AgentRuntime(
        chat=scripted_chat([
            # 计划 b 依赖 a，但模型先去调用依赖未满足的 b
            '{"type":"plan","tasks":['
            '{"id":"a","title":"first","depends_on":[]},'
            '{"id":"b","title":"second","depends_on":["a"]}]}',
            # 触发“依赖未满足”拒绝分支
            '{"type":"tool_call","tool":"lookup","task_id":"b","args":{}}',
            # 模型按提示重规划（此时计划并未卡死，因为有可执行的 a）
            '{"type":"plan","reason":"reorder","tasks":['
            '{"id":"a","title":"first","depends_on":[]},'
            '{"id":"b","title":"second","depends_on":["a"]}]}',
            '{"type":"tool_call","tool":"lookup","task_id":"a","args":{}}',
            '{"type":"final","answer":"done","citations":[]}',
        ]),
        registry=registry,
        max_tool_retries=0,
    ).run("stuck")

    assert result.status == "completed", result.error
    rejected = [s for s in result.steps if s.get("status") == "rejected"]
    assert rejected, "应记录一次被拒绝的调用"
    # 重规划发生在计划仍可推进时：第一次提示后模型应改调可执行任务。
    assert result.plan[0]["status"] == "completed"


def test_plan_stuck_allows_replan() -> None:
    """依赖成环导致计划卡死时，_apply_plan 必须放行重规划。"""
    from app.agent.runtime import AgentExecution, AgentRuntime as _RT, PlanDecision, PlanItem

    execution = AgentExecution(_RT(), "stuck")
    execution.plan = [
        {"id": "a", "title": "A", "depends_on": ["b"], "status": "pending"},
        {"id": "b", "title": "B", "depends_on": ["a"], "status": "pending"},
    ]
    assert execution._plan_is_stuck()

    decision = PlanDecision(
        type="plan", reason="break cycle",
        tasks=[PlanItem(id="fresh", title="fresh", depends_on=[])],
    )
    assert execution._apply_plan(decision) is None
    assert [task["id"] for task in execution.plan] == ["fresh"]


def test_replan_rejections_round_trip() -> None:
    """连续重复规划的计数必须可持久化，避免快照恢复后阈值漂移。"""
    from app.agent.runtime import AgentExecution, AgentRuntime as _RT

    execution = AgentExecution(_RT(), "q")
    assert execution.replan_rejections == 0
    state = execution.to_state()
    assert "replan_rejections" in state
    restored = AgentExecution.from_state(_RT(), {**state, "replan_rejections": 5})
    assert restored.replan_rejections == 5


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

    rejected_execution = AgentRuntime(
        chat=scripted_chat(['{"type":"tool_call","tool":"lookup","args":{}}']),
        registry=registry,
    ).start("reject")
    assert rejected_execution.advance().status == "waiting_approval"
    rejected = rejected_execution.approve(False)
    assert rejected.status == "cancelled"
    assert rejected_execution.steps[-1]["status"] == "rejected"


def test_deferred_approval_handoff() -> None:
    """Web 可以先返回 running，再在后台执行耗时写工具。"""
    writes = []

    def write(args):
        writes.append(1)
        return {"ok": True, "data": {}, "sources": []}

    execution = AgentRuntime(
        chat=scripted_chat([
            '{"type":"tool_call","tool":"lookup","args":{}}',
            '{"type":"final","answer":"done","citations":[]}',
        ]),
        registry=registry_for(write, side_effect="write"),
    ).start("write later")
    waiting = execution.advance()
    assert waiting.status == "waiting_approval"

    pending = execution.begin_approval()
    assert execution.status == "running"
    assert execution.pending_tool is None
    assert execution.steps[-1]["status"] == "running"
    assert writes == []

    completed = execution.execute_approved_tool(pending)
    assert completed.status == "completed"
    assert writes == [1]


def test_tool_runtime_context() -> None:
    """工具按声明接收本次 Agent 的模型函数和取消信号。"""
    seen = {}

    def contextual(args, *, llm_chat=None, should_stop=None):
        seen["reply"] = llm_chat([{"role": "user", "content": "ping"}])
        seen["cancelled"] = should_stop()
        return {"ok": True, "data": {}, "sources": []}

    registry = registry_for(contextual)
    result = registry.call(
        "lookup", {},
        llm_chat=lambda messages, max_tokens=2000: "web-provider",
        should_stop=lambda: False,
    )
    assert result["ok"]
    assert seen == {"reply": "web-provider", "cancelled": False}


def test_cancel_during_approved_tool() -> None:
    """取消信号应传进耗时工具，且工具返回后不能复活 Agent。"""
    started = threading.Event()
    holder = {}

    def write(args, *, should_stop=None):
        started.set()
        while not should_stop():
            time.sleep(0.005)
        return {"ok": False, "data": None, "error": {"code": "cancelled"}, "sources": []}

    execution = AgentRuntime(
        chat=scripted_chat([
            '{"type":"tool_call","tool":"lookup","args":{}}',
            '{"type":"final","answer":"must-not-run","citations":[]}',
        ]),
        registry=registry_for(write, side_effect="write"),
    ).start("cancel write")
    assert execution.advance().status == "waiting_approval"
    pending = execution.begin_approval()

    def run_tool():
        holder["result"] = execution.execute_approved_tool(pending)

    worker = threading.Thread(target=run_tool)
    worker.start()
    assert started.wait(timeout=1)
    execution.cancel()
    worker.join(timeout=1)
    assert not worker.is_alive()
    assert holder["result"].status == "cancelled"
    assert execution.steps[-1]["status"] == "cancelled"


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


def test_step_limit_can_be_extended_on_resume() -> None:
    execution = AgentRuntime(
        chat=scripted_chat([
            '{"type":"plan","tasks":[{"id":"one","title":"one","depends_on":[]}]}',
            '{"type":"final","answer":"continued","citations":[]}',
        ]),
        registry=registry_for(lambda args: {"ok": True, "data": {}, "sources": []}),
        max_steps=1,
    ).start("continue after limit")
    paused = execution.advance()
    assert paused.status == "paused"
    assert execution.step_no == 1
    assert execution.step_limit == 1

    completed = execution.resume()
    assert completed.status == "completed"
    assert completed.answer == "continued"
    assert execution.step_no == 2
    assert execution.step_limit == 2


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
    "deferred_approval": test_deferred_approval_handoff,
    "tool_context": test_tool_runtime_context,
    "cancel_approved": test_cancel_during_approved_tool,
    "timeout": test_timeout,
    "resume_budget": test_step_limit_can_be_extended_on_resume,
    "stream": test_streaming_answer,
    "args_alias": test_args_alias,
    "stuck_plan": test_stuck_plan_replan,
    "replan_stuck": test_plan_stuck_allows_replan,
    "replan_state": test_replan_rejections_round_trip,
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
