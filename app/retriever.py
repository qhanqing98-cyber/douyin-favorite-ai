"""混合检索：语义向量 + 关键词两路召回 → RRF 融合 → 问答上下文。

- 向量路：问题向量化后与 chunks 全量矩阵内积（语料 ≤千级，暴力检索微秒级完成，
  进程内缓存，按「块数/最大 id/构建轮次」版本号失效）；
- 关键词路：db.search_for_ask（只搜有效转写、全词 AND 优先、bm25 列权重排序）；
- 融合：Reciprocal Rank Fusion(k=60) —— 两路都命中的视频排最前；
- 每个命中视频给出「命中片段」作为 LLM 上下文，替代旧版固定取转写开头 3000 字。

向量模型缺失时自动降级为纯关键词检索，问答始终可用。
"""
import threading

import numpy as np

from app import db, embedder

RRF_K = 60
K_VECTOR = 20  # 向量路候选块数
K_KEYWORD = 20  # 关键词路候选视频数
_EXCERPT_WIDTH = 400  # 命中窗口：命中点前 400 字 + 后 800 字

_lock = threading.Lock()
_cache: dict = {"version": None, "ids": [], "matrix": None}


def _matrix():
    """全部块向量（进程内缓存）。返回 (ids, matrix)：ids=[(aweme_id, source, text)]。"""
    with db.get_conn() as conn:
        n, max_id = conn.execute(
            "SELECT COUNT(*), IFNULL(MAX(id), 0) FROM chunks"
        ).fetchone()
    version = (n, max_id, db.get_meta("index_builds"))
    with _lock:
        if _cache["version"] == version:
            return _cache["ids"], _cache["matrix"]

    rows = db.all_chunks()
    if rows:
        # 只保留维度正确的块：万一有「换模型后重建到一半」的混合索引，
        # 也不会因 np.vstack 维度不一致而整条向量路崩掉
        pairs = [(r, np.frombuffer(r["vec"], dtype=np.float32)) for r in rows]
        pairs = [(r, v) for r, v in pairs if v.shape[0] == embedder.DIM]
        ids = [(r["aweme_id"], r["source"], r["text"]) for r, _ in pairs]
        mat = (np.vstack([v for _, v in pairs]) if pairs
               else np.zeros((0, embedder.DIM), dtype=np.float32))
    else:
        ids, mat = [], np.zeros((0, embedder.DIM), dtype=np.float32)
    with _lock:
        _cache.update(version=version, ids=ids, matrix=mat)
    return ids, mat


def _vector_hits(question: str, k: int) -> list[tuple[str, dict]]:
    """向量召回：按 aweme_id 聚合，只保留每个视频的最佳块。"""
    ids, mat = _matrix()
    if mat.shape[0] == 0:
        return []
    query_vec = embedder.encode_query(question)[0]
    scores = mat @ query_vec  # 都归一化过，内积 = 余弦
    best: dict[str, dict] = {}
    for i in np.argsort(-scores)[:k]:
        idx = int(i)
        aweme_id, source, text = ids[idx]
        if aweme_id not in best or scores[idx] > best[aweme_id]["score"]:
            best[aweme_id] = {"source": source, "text": text, "score": float(scores[idx])}
    ranked = sorted(best.items(), key=lambda kv: -kv[1]["score"])
    return [(aid, {**info, "rank": r + 1}) for r, (aid, info) in enumerate(ranked)]


def _excerpt(transcript: str, question: str) -> str:
    """在转写里定位与问题相关的窗口（命中词附近，而非固定取开头）。"""
    t = transcript or ""
    w = _EXCERPT_WIDTH
    if len(t) <= w * 2:
        return t
    pos = -1
    for term in db.ask_terms(question):
        pos = t.find(term)
        if pos >= 0:
            break
    if pos < 0:
        return t[: w * 2]
    start = max(0, pos - w)
    end = min(len(t), pos + w * 2)
    return ("…" if start > 0 else "") + t[start:end] + ("…" if end < len(t) else "")


def hybrid_search(question: str, k_videos: int = 5) -> list[dict]:
    """返回 top 命中视频：[{aweme_id, title, author, matched_by, score, excerpt}]。"""
    try:
        vec_hits = _vector_hits(question, K_VECTOR)
    except Exception:
        # 向量模型缺失/依赖异常/文件损坏等 → 静默退化为纯关键词检索，问答始终可用
        vec_hits = []
    kw_hits = db.search_for_ask(question, limit=K_KEYWORD)

    rrf: dict[str, float] = {}
    vec_best: dict[str, dict] = {}
    sources: dict[str, set] = {}
    for aweme_id, h in vec_hits:
        rrf[aweme_id] = rrf.get(aweme_id, 0.0) + 1.0 / (RRF_K + h["rank"])
        sources.setdefault(aweme_id, set()).add("vector")
        vec_best[aweme_id] = h
    for rank, row in enumerate(kw_hits, 1):
        aweme_id = row["aweme_id"]
        rrf[aweme_id] = rrf.get(aweme_id, 0.0) + 1.0 / (RRF_K + rank)
        sources.setdefault(aweme_id, set()).add("keyword")

    hits: list[dict] = []
    for aweme_id, score in sorted(rrf.items(), key=lambda kv: -kv[1])[:k_videos]:
        row = db.get_video(aweme_id)
        if not row:
            continue
        vec = vec_best.get(aweme_id)
        if vec and vec["source"] == "transcript":
            excerpt = vec["text"]  # 向量命中的就是转写片段，直接当上下文
        else:
            tr = row["transcript"] or ""
            if tr and not tr.startswith("【"):
                excerpt = _excerpt(tr, question)
            elif (row["summary"] or "").strip():
                excerpt = f"概要：{row['summary']}"
            else:
                excerpt = vec["text"] if vec else f"标题：{row['title']}"
        hits.append({
            "aweme_id": aweme_id,
            "title": row["title"],
            "author": row["author"],
            "matched_by": "+".join(sorted(sources[aweme_id])),
            "score": round(score, 4),
            "excerpt": excerpt,
        })
    return hits


def build_contexts(hits: list[dict]) -> list[dict]:
    """命中 → llm.answer 的 contexts（excerpt 优先，transcript 兜底）。"""
    return [
        {
            "title": h["title"],
            "author": h["author"],
            "aweme_id": h["aweme_id"],
            "transcript": "",
            "excerpt": h["excerpt"],
        }
        for h in hits
    ]
