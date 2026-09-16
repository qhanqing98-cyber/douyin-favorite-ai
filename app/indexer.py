"""向量索引：转写/概要/标题 → 分块 → bge 向量化 → chunks 表。

分块设计（问答上下文的最小单位）：
- transcript：~500 字、按句子边界聚合、块间重叠 ~60 字。只取真实转写，
  【音频不可用】等占位文本不参与（否则占位内容会污染上下文）；
- summary：标题+概要 1 块 —— 覆盖「已概要但转写不可用」的视频；
- meta：标题+标签+作者 1 块 —— 全部视频都有，兜底可搜性。

增量策略：每个视频存内容签名（sig），内容没变就不重嵌入；
ensure_ready() 供 ask 前调用，自动补齐缺失与内容已变的视频。
"""
import hashlib
import re

import numpy as np

from app import db

CHUNK_SIZE = 500
CHUNK_OVERLAP = 60
_SENT = re.compile(r"[。！？!?；;\n]+")


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """按句子边界聚合成 ~size 字的块；超长单句硬切（带重叠）。"""
    text = (text or "").strip()
    if not text:
        return []
    sents: list[str] = []
    start = 0
    for m in _SENT.finditer(text):
        if m.end() > start:
            sents.append(text[start : m.end()])
        start = m.end()
    if start < len(text):
        sents.append(text[start:])
    if not sents:
        sents = [text]

    chunks: list[str] = []
    cur = ""
    for s in sents:
        while len(s) > size + 200:  # 无标点的超长单句
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(s[:size])
            s = s[max(size - overlap, 1) :]
        if cur and len(cur) + len(s) > size:
            chunks.append(cur)
            cur = cur[-overlap:] if overlap > 0 else ""
        cur += s
    if cur.strip():
        chunks.append(cur)
    return [c.strip() for c in chunks if c.strip()]


def video_sig(v) -> str:
    """视频参与索引的内容指纹——任一字段变化都会触发重嵌入。"""
    h = hashlib.md5()
    for key in ("title", "tags", "author", "transcript", "summary"):
        h.update((v[key] or "").encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()


def _chunks_for(v) -> list[dict]:
    title = v["title"] or ""
    chunks: list[dict] = []
    tr = v["transcript"] or ""
    if tr and not tr.startswith("【"):
        for j, piece in enumerate(chunk_text(tr)):
            chunks.append({"source": "transcript", "idx": j, "text": piece})
    if (v["summary"] or "").strip():
        chunks.append({"source": "summary", "idx": 0,
                       "text": f"标题：{title}\n概要：{v['summary']}"})
    chunks.append({"source": "meta", "idx": 0,
                   "text": f"标题：{title} 标签：{(v['tags'] or '').strip()} 作者：{v['author']}"})
    return chunks


def build(all: bool = False, progress=None) -> int:
    """构建/增量更新向量索引。返回本次（重）嵌入的视频数。

    all=True 全量重建；否则只处理「没有索引或内容签名变化」的视频。
    全量重建也是**逐视频替换**（不清表）：中途失败/取消时，索引里始终是
    「旧的可用块 + 已更新的新块」，而不是一片空白。
    progress(done, total, title) 可选，风格与 transcribe.run 一致。
    """
    from app import embedder

    videos = db.all_videos()
    sigs = {} if all else db.chunk_sigs()  # 全量重建：签名表视为空 → 所有视频都待处理
    todo = []
    for v in videos:
        sig = video_sig(v)
        if sigs.get(v["aweme_id"]) != sig:
            todo.append((v, sig))

    for i, (v, sig) in enumerate(todo):
        if progress:
            progress(i, len(todo), v["title"])
        chunks = _chunks_for(v)
        if not chunks:
            continue
        vecs = embedder.encode_docs([c["text"] for c in chunks])
        for c, vec in zip(chunks, vecs):
            c["vec"] = np.asarray(vec, dtype=np.float32).tobytes()
        db.replace_chunks(v["aweme_id"], chunks, sig)

    db.drop_chunks_not_in([v["aweme_id"] for v in videos])  # 清理孤儿块
    db.set_meta("embed_model", embedder.MODEL_TAG)
    db.set_meta("index_builds", str(int(db.get_meta("index_builds") or 0) + 1))
    return len(todo)


def ensure_ready(progress=None) -> int:
    """ask 前调用：索引为空/内容有更新 → 增量补建；换了模型 → 全量重建。

    没活干时只是两条轻量查询，可以放心每次 ask 都调。
    """
    from app import embedder

    if db.count_chunks() > 0 and db.get_meta("embed_model") != embedder.MODEL_TAG:
        return build(all=True, progress=progress)
    return build(all=False, progress=progress)
