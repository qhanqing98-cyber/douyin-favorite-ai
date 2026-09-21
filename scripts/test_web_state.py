"""Web Agent 持久化恢复的离线回归测试；不读取或修改正式数据库。"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import db
from app.agent.runtime import AgentExecution, AgentResult, AgentRuntime


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        db.DB_PATH = Path(directory) / "state-test.db"
        # web 导入时会初始化数据库；DB_PATH 已先指向临时目录。
        from app import web

        session_id = "session"
        waiting_id = "waiting"
        execution = AgentExecution(AgentRuntime(chat=lambda messages, max_tokens=2000: ""), "q")
        execution.status = "waiting_approval"
        execution.pending_tool = {
            "step": 1, "type": "tool_call", "tool": "transcribe_videos",
            "args": {"ids": ["1"]}, "side_effect": "write",
        }
        execution.steps = [{**execution.pending_tool, "status": "waiting_approval"}]
        db.create_agent_session(session_id, "test")
        db.create_agent_run(waiting_id, session_id, "q", execution.to_state())
        db.update_agent_run(waiting_id, status="waiting_approval", state=execution.to_state())

        waiting = web._load_agent_record(waiting_id)
        assert waiting is not None
        assert waiting["needs_credentials"] is False
        assert web._agent_payload(waiting_id, waiting)["status"] == "waiting_approval"

        interrupted_id = "interrupted"
        execution.status = "running"
        execution.pending_tool = None
        stale = AgentResult(
            status="waiting_approval", pending_tool={"tool": "transcribe_videos"},
        )
        db.create_agent_run(interrupted_id, session_id, "q2", execution.to_state())
        db.update_agent_run(
            interrupted_id, status="interrupted", state=execution.to_state(),
            result=stale.as_dict(), error="service restarted",
        )
        interrupted = web._load_agent_record(interrupted_id)
        assert interrupted is not None
        assert interrupted["result"] is None
        assert web._agent_payload(interrupted_id, interrupted)["status"] == "interrupted"

        # A history cleanup must not erase a task that still owns live progress.
        running_job = db.create_job("running")
        assert db.delete_job(running_job) is False
        db.clear_jobs()
        assert any(item["id"] == running_job for item in db.recent_jobs())
        db.finish_job(running_job, "completed")
        assert db.delete_job(running_job) is True

    print("Web state tests: 3/3 passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
