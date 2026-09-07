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
def videos(q: str = "", limit: int = 30, category: str = ""):
    """q 非空 → 全文搜索；否则按分类（可空）列收藏库。"""
    if q.strip():
        rows = db.search(q, limit=limit)
        return {"mode": "search", "hits": [dict(r) for r in rows]}
    rows = db.list_videos(limit=limit, category=category or None)
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
    ids: list[str] = []  # 非空 = 只转写这些视频（搜索结果勾选）


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
            _job["done"], _job["total"] = done, t
            _job["progress"] = title[:24]

        transcribe.run(limit=limit, progress=on_progress, ids=ids)

    _start_job(name, run)
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


# 分类固定集合：控制 LLM 输出可预期，前端筛选也按这套来
CATEGORIES = ["AI技术", "编程开发", "软件工具", "知识学习", "心理成长", "小说写作", "游戏娱乐", "生活其他"]


@app.get("/api/categories")
def categories():
    counts = db.category_counts()
    unclassified = db.count_unclassified()
    return {"counts": counts, "unclassified": unclassified}


@app.post("/api/classify")
def classify(req: JobReq):
    """用 LLM 按标题/标签/作者给视频打分类，循环处理直到没有未分类。"""
    import json as _json

    def run():
        from app import llm

        cat_line = "、".join(CATEGORIES)
        total = db.count_unclassified()
        done = 0
        while True:
            rows = db.get_unclassified(40)
            if not rows:
                break
            listing = "\n".join(
                f"{r['aweme_id']}|{r['title'][:50]}|{r['author']}|{r['tags']}"
                for r in rows
            )
            prompt = (
                f"把每个视频分到以下类别之一：{cat_line}。\n"
                "只输出一个 JSON 对象，格式 {\"视频id\": \"类别\", ...}，不要输出任何其他文字。\n"
                "视频列表（id|标题|作者|标签）：\n" + listing
            )
            resp = llm._chat([{"role": "user", "content": prompt}], max_tokens=2000)
            text = resp.strip()
            if text.startswith("```"):  # 剥掉可能的 markdown 代码围栏
                text = text.strip("`").lstrip("json").strip()
            try:
                mapping = _json.loads(text)
            except Exception as e:
                _job["error"] = f"分类 JSON 解析失败：{str(e)[:150]}"
                return
            for r in rows:
                cat = mapping.get(r["aweme_id"])
                db.set_category(r["aweme_id"], cat if cat in CATEGORIES else "生活其他")
                done += 1
            _job["done"], _job["total"] = done, total

    _start_job(f"一键分类全部（待分类 {db.count_unclassified()}）", run)
    return {"ok": True}


@app.get("/api/job")
def job():
    return _job
