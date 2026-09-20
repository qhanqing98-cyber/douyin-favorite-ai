"""本地 Web 服务：FastAPI 后端 + 单页前端。

路由：GET / 返回 static/index.html；API 全部挂在 /api 下。
长任务（转写/概要）用后台线程跑，前端轮询 /api/job 看进度；
同一时刻只允许一个任务，防止 Whisper 把内存打爆。
"""
import json
import queue
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import db
from app.agent.runtime import AgentExecution, AgentResult, AgentRuntime
from app.llm_router import LLMRouter, LLMRouterError, PoolEndpoint, parse_pool

STATIC = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="抖音收藏知识库")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
db.init_db()

# 任务状态（进程内单例；转写/概要都是单线程批处理，够用）。
# 取消是协作式的：任务循环里查 cancel 标志，在"条与条之间"安全退出；
# 每条数据都是独立写库的，中断后重跑自动续上。
_job = {"id": None, "running": False, "name": "", "progress": "", "error": "",
        "done": 0, "total": 0, "cancel": False, "started_at": 0.0,
        "eta_seconds": None}
_lock = threading.Lock()
_agent_lock = threading.Lock()
_agent_runs: dict[str, dict] = {}


def _start_job(name: str, fn) -> None:
    with _lock:
        if _job["running"]:
            raise HTTPException(409, f"已有任务在跑：{_job['name']}")
        job_id = db.create_job(name)
        _job.update(running=True, name=name, progress="启动中...", error="",
                    done=0, total=0, cancel=False, started_at=time.time(),
                    eta_seconds=None, id=job_id)

    def wrapper():
        cancelled = False
        error = ""
        try:
            fn()
            cancelled = _job["cancel"]
            _job["progress"] = "已取消" if cancelled else "完成"
        except Exception as e:
            error = str(e)[:300]
            _job["error"] = error
        finally:
            error = error or _job["error"]
            status = "failed" if error else ("cancelled" if cancelled else "completed")
            db.finish_job(
                _job["id"], status=status,
                progress="失败" if error else ("已取消" if cancelled else "完成"),
                error=error, done=_job["done"], total=_job["total"],
            )
            _job["running"] = False

    threading.Thread(target=wrapper, daemon=True).start()


def _set_progress(**fields) -> None:
    """同时更新内存状态和 SQLite，避免刷新页面时进度丢失。"""
    with _lock:
        _job.update(fields)
        job_id = _job["id"]
    if job_id is not None:
        db.update_job(job_id, **fields)


def _reindex_quietly() -> None:
    """长任务（转写/概要）收尾后静默增量更新向量索引。

    失败不阻塞任务收尾——ask 前还会自动重试一次，且缺模型时会退化为纯关键词检索。
    """
    try:
        from app import indexer

        indexer.ensure_ready()
    except Exception:
        pass


class AgentAskReq(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, max_length=80)
    llm_pool: list[dict] = Field(default_factory=list, max_length=10)


class AgentApprovalReq(BaseModel):
    approved: bool
    llm_pool: list[dict] = Field(default_factory=list, max_length=10)


class AgentContinueReq(BaseModel):
    llm_pool: list[dict] = Field(default_factory=list, max_length=10)


class LLMTestReq(BaseModel):
    llm_pool: list[dict] = Field(default_factory=list, max_length=10)


class AgentSessionRenameReq(BaseModel):
    title: str = Field(min_length=1, max_length=80)


def _request_router(
    entries: list[dict],
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> LLMRouter:
    """Build a task-local router; headers remain as a legacy single-config path."""
    payload = entries
    if not payload and api_key and base_url and model:
        payload = [{"name": "当前配置", "api_key": api_key, "base_url": base_url, "model": model}]
    try:
        return LLMRouter(parse_pool(payload))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def _agent_runtime(router: LLMRouter) -> AgentRuntime:
    return AgentRuntime(
        chat=lambda messages, max_tokens=2000: router.chat(messages, max_tokens=max_tokens),
        stream_chat=lambda messages, on_delta, max_tokens=2000: router.chat_stream(
            messages, on_delta, max_tokens=max_tokens,
        ),
    )


@app.post("/api/llm/test")
def test_llm(req: LLMTestReq | None = None,
             api_key: str | None = Header(default=None, alias="X-LLM-API-Key"),
             base_url: str | None = Header(default=None, alias="X-LLM-Base-URL"),
             model: str | None = Header(default=None, alias="X-LLM-Model")):
    router = _request_router((req or LLMTestReq()).llm_pool, api_key, base_url, model)
    try:
        reply = router.chat([{"role": "user", "content": "只回复 OK"}], max_tokens=8)
    except Exception as exc:
        raise HTTPException(400, f"连接失败：{str(exc)[:900]}") from exc
    return {"ok": True, "reply": reply[:20], "used": router.last_used}


@app.post("/api/llm/pool/test")
def test_llm_pool(req: LLMTestReq | None = None,
                  api_key: str | None = Header(default=None, alias="X-LLM-API-Key"),
                  base_url: str | None = Header(default=None, alias="X-LLM-Base-URL"),
                  model: str | None = Header(default=None, alias="X-LLM-Model")):
    return test_llm(req, api_key, base_url, model)


@app.post("/api/llm/endpoint/test")
def test_llm_endpoint(req: LLMTestReq | None = None,
                      api_key: str | None = Header(default=None, alias="X-LLM-API-Key"),
                      base_url: str | None = Header(default=None, alias="X-LLM-Base-URL"),
                      model: str | None = Header(default=None, alias="X-LLM-Model")):
    return test_llm(req, api_key, base_url, model)


def _persist_agent_record(run_id: str, record: dict, result: AgentResult) -> None:
    """将内存中的执行对象、最终结果和步骤快照同步到 SQLite。"""
    execution = record["execution"]
    db.update_agent_run(
        run_id,
        status=result.status,
        state=execution.to_state(),
        result=result.as_dict(),
        error=result.error or "",
    )
    for step in execution.steps:
        db.add_agent_event(run_id, int(step["step"]), "step", step)


def _checkpoint_agent(run_id: str, execution: AgentExecution) -> None:
    """Agent 每完成一步时写入快照，避免长任务只在结束时落盘。"""
    db.update_agent_run(
        run_id,
        status=execution.status,
        state=execution.to_state(),
        error=execution.error or "",
    )
    for step in execution.steps:
        db.add_agent_event(run_id, int(step["step"]), "step", step)


def _emit_agent_event(record: dict, event: dict) -> None:
    """把事件广播给当前 SSE 订阅者；没有订阅者时 Agent 仍继续运行。"""
    with record["lock"]:
        subscribers = list(record.get("subscribers", ()))
    for subscriber in subscribers:
        subscriber.put(event)


def _agent_tool_progress(run_id: str, record: dict, done: int, total: int, title: str) -> None:
    """把转写工具的进度广播给 Agent 右侧执行详情。"""
    now = time.monotonic()
    with record["lock"]:
        started = record.get("tool_started_at")
        if started is None or done == 0:
            started = now
            record["tool_started_at"] = started
        elapsed = max(0.0, now - started)
        rate = done / elapsed if done > 0 and elapsed > 0 else 0.0
        eta = round((total - done) / rate) if rate > 0 else None
        record["tool_progress"] = {
            "kind": "transcription",
            "done": done,
            "total": total,
            "title": title,
            "elapsed_seconds": round(elapsed),
            "eta_seconds": eta,
        }
    _emit_agent_event(record, {"type": "snapshot", "run": _agent_payload(run_id, record)})


def _agent_changed(run_id: str, record: dict, execution: AgentExecution) -> None:
    _checkpoint_agent(run_id, execution)
    _emit_agent_event(record, {"type": "snapshot", "run": _agent_payload(run_id, record)})


def _agent_text_delta(run_id: str, record: dict, delta: str, answer: str) -> None:
    _emit_agent_event(record, {
        "type": "text_delta", "run_id": run_id, "delta": delta,
        "answer_length": len(answer),
    })


def _load_agent_record(run_id: str) -> dict | None:
    """进程重启后从数据库恢复一次 Agent 执行。"""
    row = db.get_agent_run(run_id)
    if row is None:
        return None
    state = json.loads(row["state_json"])
    execution = AgentExecution.from_state(AgentRuntime(), state)
    if row["status"] == "interrupted":
        execution.status = "running"
        execution.error = None
    result = AgentResult.from_dict(json.loads(row["result_json"])) if row["result_json"] else None
    return {
        "execution": execution,
        "result": result,
        "router": None,
        "needs_credentials": True,
        "status": row["status"],
        "session_id": row["session_id"],
        "error": row["error"],
        "lock": threading.RLock(),
        "subscribers": set(),
        "tool_progress": None,
        "tool_started_at": None,
        "created_at": row["created_at"],
    }


def _agent_payload(run_id: str, record: dict) -> dict:
    with record["lock"]:
        result = record["result"]
        router = record.get("router")
        llm_state = None
        if router is not None:
            llm_state = {
                "last_used": router.last_used,
                "cooldowns": [item.safe_info() for item in router.endpoints if item.cooldown_until > time.monotonic()],
            }
        if result is None:
            missing_credentials = record.get("needs_credentials", False)
            return {
                "run_id": run_id,
                "session_id": record.get("session_id"),
                "question": record["execution"].question,
                "status": "interrupted" if missing_credentials else record["execution"].status,
                "error": ("服务已重启，请重新提交模型池后继续" if missing_credentials
                          else record.get("error") or record["execution"].error),
                "answer": record["execution"].answer,
                "sources": record["execution"].sources,
                "plan": record["execution"].plan,
                "steps": record["execution"].steps,
                "pending_tool": record["execution"].pending_tool,
                "tool_progress": record.get("tool_progress"),
                "llm": llm_state,
            }
        return {
            "run_id": run_id,
            "session_id": record.get("session_id"),
            "question": record["execution"].question,
            "llm": llm_state,
            "tool_progress": record.get("tool_progress"),
            **result.as_dict(),
        }


def _run_agent(run_id: str, record: dict) -> None:
    try:
        result = record["execution"].advance()
    except Exception as exc:
        result = AgentResult(status="failed", error=str(exc)[:300])
    with record["lock"]:
        record["result"] = result
        record["status"] = result.status
    _persist_agent_record(run_id, record, result)
    _emit_agent_event(record, {"type": "done", "run": _agent_payload(run_id, record)})
    if result.status in {"completed", "failed", "cancelled"}:
        with record["lock"]:
            record["router"] = None
            record["pool"] = None
            record["execution"].runtime = AgentRuntime()


@app.post("/api/agent/ask")
def agent_ask(
    req: AgentAskReq,
    api_key: str | None = Header(default=None, alias="X-LLM-API-Key"),
    base_url: str | None = Header(default=None, alias="X-LLM-Base-URL"),
    model: str | None = Header(default=None, alias="X-LLM-Model"),
):
    """异步启动一次 Agent 研究任务，返回 run_id 供前端轮询。"""
    router = _request_router(req.llm_pool, api_key, base_url, model)
    run_id = uuid.uuid4().hex
    session_id = req.session_id or uuid.uuid4().hex
    latest = db.latest_agent_run(session_id) if req.session_id else None
    if latest and latest["status"] not in {"completed", "failed", "cancelled"}:
        raise HTTPException(409, "当前会话还有未完成的 Agent 任务，请先继续或结束它")
    if latest:
        execution = AgentExecution.from_state(_agent_runtime(router), json.loads(latest["state_json"]))
        execution.question = req.question.strip()
        execution.messages.append({"role": "user", "content": execution.question})
        execution.sources = []
        execution.steps = []
        execution.plan = []
        execution.active_task_id = None
        execution.step_no = 0
        execution.status = "running"
        execution.answer = ""
        execution.error = None
        execution.pending_tool = None
    else:
        execution = _agent_runtime(router).start(req.question)
    db.create_agent_session(session_id, req.question[:80])
    db.create_agent_run(run_id, session_id, req.question, execution.to_state())
    record = {
        "execution": execution,
        "result": None,
        "status": "running",
        "router": router,
        "pool": router.endpoints,
        "needs_credentials": False,
        "session_id": session_id,
        "lock": threading.RLock(),
        "subscribers": set(),
        "tool_progress": None,
        "tool_started_at": None,
        "created_at": time.time(),
    }
    execution.runtime.tool_progress = lambda done, total, title: _agent_tool_progress(
        run_id, record, done, total, title,
    )
    execution.on_change = lambda current: _agent_changed(run_id, record, current)
    execution.on_text_delta = lambda delta, answer: _agent_text_delta(
        run_id, record, delta, answer,
    )
    with _agent_lock:
        _agent_runs[run_id] = record
        # 本地单用户只保留最近 20 次；正在运行或等待确认的任务不清理。
        if len(_agent_runs) > 20:
            removable = sorted(
                ((rid, item) for rid, item in _agent_runs.items() if item["status"] not in {"running", "waiting_approval"}),
                key=lambda pair: pair[1]["created_at"],
            )
            for rid, _ in removable[: max(0, len(_agent_runs) - 20)]:
                _agent_runs.pop(rid, None)
    threading.Thread(target=_run_agent, args=(run_id, record), daemon=True).start()
    return {"run_id": run_id, "session_id": session_id, "question": req.question.strip(), "status": "running"}


@app.get("/api/agent/history")
def agent_history(limit: int = 20):
    """返回 Agent 历史摘要，完整上下文仍只通过 run_id 读取。"""
    return {"items": db.recent_agent_runs(max(1, min(limit, 50)))}


@app.get("/api/agent/sessions")
def agent_sessions(limit: int = 20):
    """返回会话列表；每个会话附带最近一次运行的摘要。"""
    return {"items": db.recent_agent_sessions(max(1, min(limit, 50)))}


@app.get("/api/agent/sessions/{session_id}")
def agent_session_detail(session_id: str):
    detail = db.get_agent_session_detail(session_id)
    if detail is None:
        raise HTTPException(404, "会话不存在")
    return detail


@app.patch("/api/agent/sessions/{session_id}")
def agent_session_rename(session_id: str, req: AgentSessionRenameReq):
    if not db.rename_agent_session(session_id, req.title):
        raise HTTPException(404, "会话不存在")
    return {"ok": True, "session_id": session_id, "title": req.title.strip()}


@app.delete("/api/agent/sessions/{session_id}")
def agent_session_delete(session_id: str):
    try:
        run_ids = db.delete_agent_session(session_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    with _agent_lock:
        for run_id in run_ids:
            _agent_runs.pop(run_id, None)
    return {"ok": True, "session_id": session_id}


@app.get("/api/agent/{run_id}/stream")
def agent_stream(run_id: str):
    """SSE：推送执行快照和最终回答增量；客户端断开不会取消任务。"""
    record = _agent_runs.get(run_id)
    if record is None:
        record = _load_agent_record(run_id)
        if record is None:
            raise HTTPException(404, "Agent 任务不存在或已过期")
        with _agent_lock:
            _agent_runs[run_id] = record

    subscriber: queue.Queue = queue.Queue()

    def events():
        with record["lock"]:
            record["subscribers"].add(subscriber)
        try:
            initial = _agent_payload(run_id, record)
            yield f"data: {json.dumps({'type': 'snapshot', 'run': initial}, ensure_ascii=False)}\n\n"
            if initial["status"] not in {"running"}:
                return
            while True:
                try:
                    event = subscriber.get(timeout=15)
                except queue.Empty:
                    yield ": keep-alive\n\n"
                    continue
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event.get("type") == "done":
                    return
        finally:
            with record["lock"]:
                record["subscribers"].discard(subscriber)

    return StreamingResponse(
        events(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/agent/{run_id}")
def agent_status(run_id: str):
    record = _agent_runs.get(run_id)
    if record is None:
        record = _load_agent_record(run_id)
        if record is None:
            raise HTTPException(404, "Agent 任务不存在或已过期")
        with _agent_lock:
            _agent_runs[run_id] = record
    return _agent_payload(run_id, record)


@app.post("/api/agent/{run_id}/approval")
def agent_approval(run_id: str, req: AgentApprovalReq,
                   api_key: str | None = Header(default=None, alias="X-LLM-API-Key"),
                   base_url: str | None = Header(default=None, alias="X-LLM-Base-URL"),
                   model: str | None = Header(default=None, alias="X-LLM-Model")):
    record = _agent_runs.get(run_id)
    if record is None:
        record = _load_agent_record(run_id)
        if record is None:
            raise HTTPException(404, "Agent 任务不存在或已过期")
        with _agent_lock:
            _agent_runs[run_id] = record
    with record["lock"]:
        # execution.status 是实时状态源；record["status"] 只在线程收尾时同步，
        # 直接用它会在“轮询到等待批准后立刻批准”时产生竞态误判。
        if record["execution"].status != "waiting_approval":
            raise HTTPException(409, "当前 Agent 没有等待批准的操作")
        record["status"] = "waiting_approval"
        record["execution"].on_change = lambda current: _agent_changed(run_id, record, current)
        record["execution"].on_text_delta = lambda delta, answer: _agent_text_delta(
            run_id, record, delta, answer,
        )
        router = _request_router(req.llm_pool, api_key, base_url, model) if req.llm_pool else record.get("router")
        if router is None:
            raise HTTPException(401, "请重新提交模型池后再继续 Agent 任务")
        record["router"] = router
        record["pool"] = router.endpoints
        record["needs_credentials"] = False
        record["execution"].runtime = _agent_runtime(router)
        record["execution"].runtime.tool_progress = lambda done, total, title: _agent_tool_progress(
            run_id, record, done, total, title,
        )
        result = record["execution"].approve(req.approved, continue_run=False)
        record["result"] = result
        record["status"] = result.status
    _persist_agent_record(run_id, record, result)
    if result.status == "running":
        with record["lock"]:
            record["result"] = None
        threading.Thread(target=_run_agent, args=(run_id, record), daemon=True).start()
    elif result.status in {"completed", "failed", "cancelled"}:
        with record["lock"]:
            record["router"] = None
            record["pool"] = None
            record["execution"].runtime = AgentRuntime()
    return _agent_payload(run_id, record)


@app.post("/api/agent/{run_id}/continue")
def agent_continue(
    run_id: str,
    req: AgentContinueReq | None = None,
    api_key: str | None = Header(default=None, alias="X-LLM-API-Key"),
    base_url: str | None = Header(default=None, alias="X-LLM-Base-URL"),
    model: str | None = Header(default=None, alias="X-LLM-Model"),
):
    """继续执行服务重启或达到步数上限后暂停的 Agent。"""
    record = _agent_runs.get(run_id)
    if record is None:
        record = _load_agent_record(run_id)
        if record is None:
            raise HTTPException(404, "Agent 任务不存在或已过期")
        with _agent_lock:
            _agent_runs[run_id] = record
    with record["lock"]:
        if record["status"] not in {"interrupted", "paused"}:
            raise HTTPException(409, "当前 Agent 不需要继续执行")
        request_pool = (req or AgentContinueReq()).llm_pool
        router = _request_router(request_pool, api_key, base_url, model) if request_pool else record.get("router")
        if router is None:
            raise HTTPException(401, "请重新提交模型池后再继续 Agent 任务")
        record["router"] = router
        record["pool"] = router.endpoints
        record["needs_credentials"] = False
        record["execution"].on_change = lambda current: _agent_changed(run_id, record, current)
        record["execution"].on_text_delta = lambda delta, answer: _agent_text_delta(
            run_id, record, delta, answer,
        )
        record["execution"].runtime = _agent_runtime(router)
        record["execution"].runtime.tool_progress = lambda done, total, title: _agent_tool_progress(
            run_id, record, done, total, title,
        )
        record["execution"].status = "running"
        record["execution"].error = None
        record["result"] = None
        record["status"] = "running"
        record["error"] = ""
        db.update_agent_run(
            run_id,
            status="running",
            state=record["execution"].to_state(),
            clear_result=True,
            error="",
        )
    threading.Thread(target=_run_agent, args=(run_id, record), daemon=True).start()
    return _agent_payload(run_id, record)


@app.post("/api/agent/{run_id}/cancel")
def agent_cancel(run_id: str):
    record = _agent_runs.get(run_id)
    if record is None:
        record = _load_agent_record(run_id)
        if record is None:
            raise HTTPException(404, "Agent 任务不存在或已过期")
        with _agent_lock:
            _agent_runs[run_id] = record
    with record["lock"]:
        if record["status"] in {"completed", "failed", "cancelled"}:
            raise HTTPException(409, "Agent 任务已经结束")
        result = record["execution"].cancel()
        record["result"] = result
        record["status"] = result.status
    _persist_agent_record(run_id, record, result)
    with record["lock"]:
        record["router"] = None
        record["pool"] = None
        record["execution"].runtime = AgentRuntime()
    return _agent_payload(run_id, record)


@app.post("/api/cancel")
def cancel():
    if not _job["running"]:
        raise HTTPException(409, "当前没有在跑的任务")
    _job["cancel"] = True
    _job["progress"] = "取消中（等当前这条处理完）..."
    db.update_job(_job["id"], cancel=1, progress=_job["progress"])
    return {"ok": True}


@app.delete("/api/history/{hid}")
def delete_history(hid: int):
    db.delete_job(hid)
    return {"ok": True}


@app.delete("/api/history")
def clear_history():
    db.clear_jobs()
    return {"ok": True}


@app.post("/api/login")
def login_ep():
    """打开有头浏览器等待扫码（最长 5 分钟），可在页面上取消。"""
    from app import crawler

    def run():
        ok = crawler.login(
            progress=lambda m: _set_progress(progress=m[:60]),
            should_stop=lambda: _job["cancel"],
        )
        if not ok and not _job["cancel"]:
            _set_progress(error="等待超时，未检测到登录")

    _start_job("扫码登录抖音", run)
    return {"ok": True}


@app.post("/api/crawl")
def crawl_ep():
    """打开浏览器滚动收藏夹采集入库（需已登录）。"""
    from app import crawler

    def run():
        crawler.crawl(
            progress=lambda m: _set_progress(progress=m[:60]),
            should_stop=lambda: _job["cancel"],
        )
        # 采集是"捕获条数"型任务，没有固定总量，不放比例进度条

    _start_job("同步收藏夹", run)
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/stats")
def stats():
    return db.stats()


@app.get("/api/videos")
def videos(q: str = "", limit: int = 30, category: str = ""):
    """q 非空 → 全文搜索；否则按分类（可空）列收藏库。"""
    if q.strip():
        rows = db.search(q, limit=limit)
        return {"mode": "search", "hits": [dict(r) for r in rows]}
    rows = db.list_videos(limit=limit, category=category or None)
    return {"mode": "list", "hits": [dict(r) for r in rows]}


class AskReq(BaseModel):
    question: str
    llm_pool: list[dict] = Field(default_factory=list, max_length=10)


@app.post("/api/ask")
def ask(req: AskReq,
        api_key: str | None = Header(default=None, alias="X-LLM-API-Key"),
        base_url: str | None = Header(default=None, alias="X-LLM-Base-URL"),
        model: str | None = Header(default=None, alias="X-LLM-Model")):
    """检索问答：语义+关键词混合召回 → 命中片段作上下文喂给 LLM。"""
    from app import retriever
    router = _request_router(req.llm_pool, api_key, base_url, model)

    try:  # 索引缺失/内容变更 → 自动增量构建；向量模型缺失则退化为纯关键词
        from app import indexer

        indexer.ensure_ready()
    except Exception:
        pass
    try:
        hits = retriever.hybrid_search(req.question)
    except Exception as e:
        return {"answer": f"检索失败：{str(e)[:200]}", "sources": []}
    if not hits:
        return {
            "answer": "没检索到相关内容。可以先点上方「转写一批」补充可检索内容，"
                      "或者换个说法再问（目前只有部分视频有转写/概要）。",
            "sources": [],
        }
    try:
        answer = router.chat(
            [{"role": "system", "content": "根据收藏内容回答并标注来源。"},
             {"role": "user", "content": req.question + "\n\n" + str(retriever.build_contexts(hits))}],
            max_tokens=2000,
        )
    except Exception as e:
        return {"answer": f"AI 调用失败：{str(e)[:300]}", "sources": []}
    return {
        "answer": answer,
        "sources": [
            dict(title=h["title"], aweme_id=h["aweme_id"], matched_by=h["matched_by"])
            for h in hits
        ],
    }


class JobReq(BaseModel):
    n: int = 10
    all: bool = False  # True = 跑完全部剩余
    ids: list[str] = []  # 非空 = 只转写这些视频（搜索结果勾选）
    llm_pool: list[dict] = Field(default_factory=list, max_length=10)


@app.post("/api/transcribe")
def transcribe(req: JobReq):
    from app import db
    from app import transcribe

    if req.ids:
        total = len(db.get_untranscribed(10**9, ids=req.ids))
        name = f"转写勾选（可转 {total} 条）"
        limit, ids = 10**9, req.ids
    elif req.all:
        total = db.count_untranscribed()
        name = f"一键转写全部（待转写 {total}）"
        limit, ids = 10**9, None
    else:
        total = min(req.n, db.count_untranscribed())
        name = f"转写 {req.n} 条（待转写 {total}）"
        limit, ids = req.n, None

    def run():
        def on_progress(done: int, t: int, title: str):
            elapsed = max(0.0, time.time() - _job["started_at"])
            rate = done / elapsed if done > 0 and elapsed > 0 else 0.0
            eta = round((t - done) / rate) if rate > 0 else None
            _set_progress(
                done=done, total=t, progress=title[:48], eta_seconds=eta,
            )

        report = transcribe.run(
            limit=limit, progress=on_progress, ids=ids,
            should_stop=lambda: _job["cancel"],
        )
        if report.failed:
            _set_progress(
                error=(f"已完成 {report.processed}/{report.requested} 条，"
                       f"{len(report.failed)} 条失败，失败项保留待重试"),
            )
        _reindex_quietly()  # 新转写的内容补进向量索引

    _start_job(name, run)
    return {"ok": True}


@app.post("/api/summarize")
def summarize(req: JobReq,
              api_key: str | None = Header(default=None, alias="X-LLM-API-Key"),
              base_url: str | None = Header(default=None, alias="X-LLM-Base-URL"),
              model: str | None = Header(default=None, alias="X-LLM-Model")):
    from app import db as _db
    router = _request_router(req.llm_pool, api_key, base_url, model)

    total = _db.count_unsummarized() if req.all else min(req.n, _db.count_unsummarized())

    def run():
        remaining = 10**9 if req.all else req.n
        done = 0
        while remaining > 0 and not _job["cancel"]:
            rows = _db.get_unsummarized(min(50, remaining))
            if not rows:
                break
            for row in rows:
                if _job["cancel"]:
                    return
                _set_progress(done=done, total=total, progress=f"{row['title'][:20]}")
                prompt = (
                    f"请为视频生成 3-5 句话概要，不要编造信息。标题：{row['title']}"
                    f" 作者：{row['author']} 转写：{row['transcript'][:8000]}"
                )
                summary = router.chat([{"role": "user", "content": prompt}], max_tokens=1500)
                _db.set_summary(row["aweme_id"], summary)
                done += 1
                remaining -= 1
        _reindex_quietly()  # 新概要补进向量索引

    _start_job(f"{'一键概要全部' if req.all else f'概要 {req.n} 条'}（待概要 {total}）", run)
    return {"ok": True}


@app.post("/api/reindex")
def reindex_ep():
    """全量重建语义向量索引（换了模型、或觉得检索不准时用）。"""

    def run():
        from app import indexer

        def on_progress(done: int, t: int, title: str):
            _set_progress(done=done, total=t, progress=title[:24])

        n = indexer.build(all=True, progress=on_progress)
        _set_progress(progress=f"已重建 {n} 个视频的向量索引")

    _start_job("重建语义索引", run)
    return {"ok": True}


# 分类集合不写死：首次分类时由 LLM 根据收藏内容自动设计，存进 meta 表。
# DEFAULT 只在 LLM 设计失败时兜底。
DEFAULT_CATEGORIES = ["知识学习", "科技数码", "生活", "娱乐", "其他"]


def load_categories() -> list[str]:
    raw = db.get_meta("categories")
    if raw:
        try:
            cats = json.loads(raw)
            if isinstance(cats, list) and cats and all(isinstance(c, str) and c.strip() for c in cats):
                return cats
        except Exception:
            pass
    return []


def propose_categories(router: LLMRouter) -> list[str] | None:
    """抽样标题/标签让 LLM 设计一套贴合当前收藏夹的分类（6~12 个 + 兜底「其他」）。"""
    sample = db.sample_videos(100)
    if not sample:
        return None
    listing = "\n".join(
        f"{(r['title'] or '')[:40]}｜{r['tags'] or ''}" for r in sample
    )
    prompt = (
        "以下是随机抽样的视频标题和标签。请为整理这个收藏夹设计 6~12 个内容分类：\n"
        "类别名 2~4 个字、互斥、合起来能覆盖绝大多数内容；最后必须包含一个兜底类别「其他」。\n"
        '只输出 JSON 数组，格式 ["类别1", "类别2", ...]，不要输出任何其他文字。\n\n' + listing
    )
    try:
        text = router.chat([{"role": "user", "content": prompt}], max_tokens=500).strip()
        if text.startswith("```"):
            text = text.strip("`").lstrip("json").strip()
        cats = json.loads(text)
        if not isinstance(cats, list):
            return None
        seen: list[str] = []
        for c in cats:
            if isinstance(c, str) and 2 <= len(c.strip()) <= 6 and c.strip() not in seen:
                seen.append(c.strip())
        if len(seen) < 4 or len(seen) > 14:
            return None
        if "其他" not in seen:
            seen.append("其他")
        return seen
    except Exception:
        return None


@app.get("/api/categories")
def categories():
    counts = db.category_counts()
    unclassified = db.count_unclassified()
    return {"counts": counts, "unclassified": unclassified}


@app.post("/api/classify")
def classify(req: JobReq,
             api_key: str | None = Header(default=None, alias="X-LLM-API-Key"),
             base_url: str | None = Header(default=None, alias="X-LLM-Base-URL"),
             model: str | None = Header(default=None, alias="X-LLM-Model")):
    """LLM 自动分类：首次先按收藏内容设计分类集合，再批量归类。"""
    import json as _json
    router = _request_router(req.llm_pool, api_key, base_url, model)

    def run():
        cats = load_categories()
        if req.all:
            db.reset_categories()  # all=true：清空重分，且重新设计分类集合
            db.del_meta("categories")
            cats = []
        if not cats:
            cats = propose_categories(router) or list(DEFAULT_CATEGORIES)
            db.set_meta("categories", _json.dumps(cats, ensure_ascii=False))
        cat_line = "、".join(cats)
        total = db.count_unclassified()
        done = 0
        while not _job["cancel"]:
            rows = db.get_unclassified(40)
            if not rows:
                break
            listing = "\n".join(
                f"{r['aweme_id']}|{r['title'][:50]}|{r['author']}|{r['tags']}"
                for r in rows
            )
            prompt = (
                f"把每个视频分到以下类别之一：{cat_line}。\n"
                "如果某个视频不属于其中任何一类（或信息太少无法判断），必须归入「其他」，不要自创类别。\n"
                "只输出一个 JSON 对象，格式 {\"视频id\": \"类别\", ...}，不要输出任何其他文字。\n"
                "视频列表（id|标题|作者|标签）：\n" + listing
            )
            resp = router.chat([{"role": "user", "content": prompt}], max_tokens=2000)
            text = resp.strip()
            if text.startswith("```"):  # 剥掉可能的 markdown 代码围栏
                text = text.strip("`").lstrip("json").strip()
            try:
                mapping = _json.loads(text)
            except Exception as e:
                _set_progress(error=f"分类 JSON 解析失败：{str(e)[:150]}")
                return
            for r in rows:
                cat = mapping.get(r["aweme_id"])
                db.set_category(r["aweme_id"], cat if cat in cats else "其他")
                done += 1
            _set_progress(done=done, total=total)

    if req.all:
        with db.get_conn() as conn:
            pending = conn.execute("SELECT COUNT(*) FROM favorites").fetchone()[0]
    else:
        pending = db.count_unclassified()
    _start_job(f"{'重新分类全部' if req.all else '一键分类全部'}（共 {pending} 条）", run)
    return {"ok": True}


@app.get("/api/job")
def job():
    status_text = {
        "completed": "完成",
        "cancelled": "已取消",
        "failed": "失败",
        "interrupted": "已中断",
        "running": "运行中",
    }
    return {
        **_job,
        "elapsed": round(time.time() - _job["started_at"]) if _job["running"] else 0,
        "history": [
            {**item, "status": status_text.get(item["status"], item["status"])}
            for item in db.recent_jobs()
        ],
    }
