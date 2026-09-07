"""本地 Web 服务：FastAPI 后端 + 单页前端。

路由：GET / 返回 static/index.html；API 全部挂在 /api 下。
长任务（转写/概要）用后台线程跑，前端轮询 /api/job 看进度；
同一时刻只允许一个任务，防止 Whisper 把内存打爆。
"""
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app import db

STATIC = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="抖音收藏知识库")

# 任务状态（进程内单例；转写/概要都是单线程批处理，够用）
_job = {"running": False, "name": "", "progress": "", "error": "", "done": 0, "total": 0}
_lock = threading.Lock()


def _start_job(name: str, fn) -> None:
    with _lock:
        if _job["running"]:
            raise HTTPException(409, f"已有任务在跑：{_job['name']}")
        _job.update(running=True, name=name, progress="启动中...", error="", done=0, total=0)

    def wrapper():
        try:
            fn()
            _job["progress"] = "完成"
        except Exception as e:
            _job["error"] = str(e)[:300]
        finally:
            _job["running"] = False

    threading.Thread(target=wrapper, daemon=True).start()


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/stats")
def stats():
    return db.stats()


@app.get("/api/videos")
def videos(q: str = "", limit: int = 30):
    """q 为空 → 收藏库列表；q 非空 → 全文搜索。"""
    if q.strip():
        rows = db.search(q, limit=limit)
        return {"mode": "search", "hits": [dict(r) for r in rows]}
    rows = db.list_videos(limit=limit)
    return {"mode": "list", "hits": [dict(r) for r in rows]}


class AskReq(BaseModel):
    question: str


@app.post("/api/ask")
def ask(req: AskReq):
    """检索问答：搜到带转写的命中视频 → 作为上下文喂给 LLM。"""
    from app import llm

    rows = db.search(req.question, limit=5)
    with_transcript = [r for r in rows if r["transcript"]]
    if not with_transcript:
        return {
            "answer": "没有带转写内容的命中视频。可以先点上方「转写一批」，"
                      "或换个关键词试试（目前只有部分视频有转写）。",
            "sources": [],
        }
    contexts = [
        {
            "title": r["title"],
            "author": r["author"],
            "aweme_id": r["aweme_id"],
            "transcript": r["transcript"],
        }
        for r in with_transcript
    ]
    try:
        answer = llm.answer(req.question, contexts)
    except Exception as e:
        return {"answer": f"AI 调用失败：{str(e)[:200]}", "sources": []}
    return {"answer": answer, "sources": [dict(title=c["title"], aweme_id=c["aweme_id"]) for c in contexts]}


class JobReq(BaseModel):
    n: int = 10
    all: bool = False  # True = 跑完全部剩余


@app.post("/api/transcribe")
def transcribe(req: JobReq):
    from app import db
    from app import transcribe

    total = db.count_untranscribed() if req.all else req.n

    def run():
        def on_progress(done: int, t: int, title: str):
            _job["done"], _job["total"] = done, t
            _job["progress"] = title[:24]

        transcribe.run(limit=10**9 if req.all else req.n, progress=on_progress)

    _start_job(f"{'一键转写全部' if req.all else f'转写 {req.n} 条'}（待转写 {total}）", run)
    return {"ok": True}


@app.post("/api/summarize")
def summarize(req: JobReq):
    from app import db as _db
    from app import llm

    total = _db.count_unsummarized() if req.all else min(req.n, _db.count_unsummarized())

    def run():
        remaining = 10**9 if req.all else req.n
        done = 0
        while remaining > 0:
            rows = _db.get_unsummarized(min(50, remaining))
            if not rows:
                break
            for row in rows:
                _job["done"], _job["total"] = done, total
                _job["progress"] = f"{row['title'][:20]}"
                summary = llm.summarize(row["title"], row["author"], row["transcript"])
                _db.set_summary(row["aweme_id"], summary)
                done += 1
                remaining -= 1

    _start_job(f"{'一键概要全部' if req.all else f'概要 {req.n} 条'}（待概要 {total}）", run)
    return {"ok": True}


@app.get("/api/job")
def job():
    return _job
