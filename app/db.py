"""SQLite 存储层：建表、写入、FTS5 中文搜索。

数据流：crawler 解析出的 dict 列表 → save_favorites() 入库（自动同步 FTS 索引）
        → search() 用关键词召回 → 返回视频列表。
"""
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

# 数据库固定放在项目根目录 data/ 下；__file__ 是本文件路径，
# .parent 两次上溯到项目根，保证从任何工作目录运行都能找到同一个库。
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "favorites.db"


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    """提供一个自动提交/回滚并关闭的连接。

    sqlite3.Connection 自身的 with 只管理事务，不负责 close；
    这里再包一层，避免 Windows 下数据库文件被遗留连接锁住。
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


FTS_COLUMNS = ("title", "tags", "author", "transcript")


def init_db() -> None:
    """建表（已存在则跳过）+ 列迁移 + FTS 索引与主表结构对齐。"""
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS favorites (
            aweme_id   TEXT PRIMARY KEY,
            title      TEXT,
            tags       TEXT,
            author     TEXT,
            share_url  TEXT,
            play_url   TEXT,
            fav_time   INTEGER,
            crawled_at INTEGER
        );
        """)
        # 列迁移：老库补新列（阶段二的转写与概要）
        cols = {r[1] for r in conn.execute("PRAGMA table_info(favorites)")}
        if "transcript" not in cols:
            conn.execute("ALTER TABLE favorites ADD COLUMN transcript TEXT")
        if "summary" not in cols:
            conn.execute("ALTER TABLE favorites ADD COLUMN summary TEXT")
        if "category" not in cols:
            conn.execute("ALTER TABLE favorites ADD COLUMN category TEXT")

        # meta 表：存键值对（目前只放自动生成的分类集合 categories）
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT
        );
        """)

        # 后台任务状态：Web 进程重启后仍能看到历史、进度和中断原因。
        # running 任务不可能在进程重启后继续执行，因此启动时标为 interrupted；
        # 任务本身按条写库，用户重新发起同一操作时会从未完成项继续。
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS jobs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            status      TEXT NOT NULL,
            progress    TEXT NOT NULL DEFAULT '',
            error       TEXT NOT NULL DEFAULT '',
            done        INTEGER NOT NULL DEFAULT 0,
            total       INTEGER NOT NULL DEFAULT 0,
            cancel      INTEGER NOT NULL DEFAULT 0,
            started_at  REAL NOT NULL,
            finished_at REAL,
            updated_at  REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
        """)
        conn.execute(
            "UPDATE jobs SET status = 'interrupted', "
            "error = ?, updated_at = ? WHERE status = 'running'",
            ("服务重启，任务未完成；可重新运行", time.time()),
        )

        # chunks 表：语义向量索引（indexer 写入、retriever 检索）。
        # vec 是 float32 小端字节串（512 维 ≈ 2KB/块）。语料 ≤千级，检索用
        # numpy 暴力内积即可，不引入 sqlite-vec（Windows 加载兼容性差、无收益）。
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS chunks (
            id       INTEGER PRIMARY KEY,
            aweme_id TEXT NOT NULL,
            source   TEXT NOT NULL,
            idx      INTEGER NOT NULL,
            text     TEXT NOT NULL,
            vec      BLOB NOT NULL,
            sig      TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_aweme ON chunks(aweme_id);
        """)

        # FTS 虚表：content='favorites' 表示它不存数据，只存倒排索引，
        # 真正内容在 favorites 表里，靠触发器保持同步。
        # tokenize='trigram' 把文本切成 3 字符滑窗，天然支持中文子串匹配。
        fts_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='favorites_fts'"
        ).fetchone()
        if fts_sql is None or "transcript" not in (fts_sql[0] or ""):
            # 索引缺 transcript 列（或不存在）→ 重建并全量回填
            conn.executescript("""
            DROP TRIGGER IF EXISTS favorites_ai;
            DROP TRIGGER IF EXISTS favorites_ad;
            DROP TRIGGER IF EXISTS favorites_au;
            DROP TABLE IF EXISTS favorites_fts;

            CREATE VIRTUAL TABLE favorites_fts USING fts5(
                title, tags, author, transcript,
                content='favorites', content_rowid='rowid',
                tokenize='trigram'
            );
            CREATE TRIGGER favorites_ai AFTER INSERT ON favorites BEGIN
                INSERT INTO favorites_fts(rowid, title, tags, author, transcript)
                VALUES (new.rowid, new.title, new.tags, new.author, new.transcript);
            END;
            CREATE TRIGGER favorites_ad AFTER DELETE ON favorites BEGIN
                INSERT INTO favorites_fts(favorites_fts, rowid, title, tags, author, transcript)
                VALUES ('delete', old.rowid, old.title, old.tags, old.author, old.transcript);
            END;
            CREATE TRIGGER favorites_au AFTER UPDATE ON favorites BEGIN
                INSERT INTO favorites_fts(favorites_fts, rowid, title, tags, author, transcript)
                VALUES ('delete', old.rowid, old.title, old.tags, old.author, old.transcript);
                INSERT INTO favorites_fts(rowid, title, tags, author, transcript)
                VALUES (new.rowid, new.title, new.tags, new.author, new.transcript);
            END;

            INSERT INTO favorites_fts(rowid, title, tags, author, transcript)
                SELECT rowid, title, tags, author, transcript FROM favorites;
            """)


def save_favorites(items: list[dict]) -> int:
    """批量写入（或忽略已存在的）。返回实际新插入的条数。

    aweme_id 是主键，INSERT OR IGNORE 遇到重复 id 就静默跳过 → 天然增量去重。
    """
    now = int(time.time())
    rows = [
        (
            it["aweme_id"], it["title"], it["tags"], it["author"],
            it["share_url"], it["play_url"], it["fav_time"], now,
        )
        for it in items
        if it.get("aweme_id")
    ]
    if not rows:
        return 0
    with get_conn() as conn:
        cur = conn.executemany(
            """INSERT OR IGNORE INTO favorites
               (aweme_id, title, tags, author, share_url, play_url, fav_time, crawled_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        return cur.rowcount


def search(query: str, limit: int = 30) -> list[sqlite3.Row]:
    """全文搜索。返回命中的视频行。

    trigram 分词下 MATCH = 子串匹配：搜"操作系统"能命中"...讲操作系统内核..."。
    但 trigram 最小匹配单位是 3 个字符，1-2 字的查询 MATCH 不出结果，
    所以短查询自动降级为 LIKE 全表扫描（收藏几千条内性能足够）。
    """
    q = query.strip()
    if not q:
        return []
    if len(q) < 3:
        with get_conn() as conn:
            like = f"%{q}%"
            return conn.execute(
                """SELECT aweme_id, title, tags, author, share_url, transcript
                   FROM favorites
                   WHERE title LIKE ? OR tags LIKE ? OR author LIKE ?
                      OR IFNULL(transcript,'') LIKE ?
                   LIMIT ?""",
                (like, like, like, like, limit),
            ).fetchall()
    words = _query_terms(q)
    if not words:
        return []
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT f.aweme_id, f.title, f.tags, f.author, f.share_url, f.transcript
               FROM favorites_fts t
               JOIN favorites f ON f.rowid = t.rowid
               WHERE favorites_fts MATCH ?
               ORDER BY bm25(favorites_fts)
               LIMIT ?""",
            (" OR ".join(f'"{w}"' for w in words), limit),
        ).fetchall()
        if rows:
            return rows
        # FTS 零命中 → LIKE 兜底（覆盖数字串、混合串等分词切不好的情况）
        like = f"%{q}%"
        return conn.execute(
            """SELECT aweme_id, title, tags, author, share_url, transcript
               FROM favorites
               WHERE title LIKE ? OR tags LIKE ? OR author LIKE ?
                  OR IFNULL(transcript,'') LIKE ?
               LIMIT ?""",
            (like, like, like, like, limit),
        ).fetchall()


def _query_terms(q: str, max_terms: int = 12) -> list[str]:
    """把查询拆成可检索的词。

    - 短查询（3-4 字）本身就是个词，整串做短语（trigram 短语 = 子串匹配）
    - 长查询用 jieba 分词，取长度 ≥2 的词做 OR 召回，bm25 按相关性排序
      （整句当单一短语要求原文一字不差，几乎必然零命中）
    """
    if len(q) <= 4:
        return [q.replace('"', '""')]
    import jieba

    seen: set[str] = set()
    for w in jieba.cut(q):
        w = w.strip()
        if len(w) >= 2 and w not in seen:
            seen.add(w)
            if len(seen) >= max_terms:
                break
    return list(seen)


# 问 AI 场景的停用词：这些词几乎出现在任何视频里，参与召回只会引入噪声。
_ASK_STOPWORDS = frozenset("""
一个 一些 什么 怎么 怎么样 如何 这个 那个 这些 那些 就是 但是 因为 所以 如果 还是 虽然 不过
而且 然后 其实 关于 通过 进行 已经 可以 应该 可能 或者 以及 我们 你们 他们 自己 大家 现在
时候 东西 地方 情况 方面 有点 一点 没有 是不是 为什么 这样 那样 之后 以后 之前
哪些 哪个 哪里 多少 什么样 有什么 几个 介绍 讲讲 说过 看过 分享
视频 抖音 收藏
""".split())


def ask_terms(q: str, max_terms: int = 8) -> list[str]:
    """问答查询词：jieba 分词 + 停用词过滤（比 _query_terms 更适合整句口语提问）。"""
    import jieba

    words: list[str] = []
    for w in jieba.cut(q):
        w = w.strip()
        if len(w) < 2 or w in _ASK_STOPWORDS or w in words:
            continue
        words.append(w)
        if len(words) >= max_terms:
            break
    if not words:  # 全被过滤（例如整句都是虚词）→ 退回整句
        words = [q.strip()]
    return words


def search_for_ask(query: str, limit: int = 30) -> list[sqlite3.Row]:
    """问答专用的关键词召回。相比 search() 的升级：
    1. 只召回「有有效转写」的视频 —— 仅标题命中的、以及【音频不可用】占位行都不占名额；
    2. 词表按长度分流：FTS5 trigram 至少要 3 个字符才建索引，中文双字词
       （副业/赚钱/面试…）交给 FTS 会静默零命中，必须走 LIKE；
    3. 召回顺序：长词全命中 ∩ 短词都出现（强相关）→ 长词任一命中（放宽）
       → 短词 LIKE → 整句 LIKE 兜底；
    4. 长词用 bm25 列权重排序：标题 8 / 标签 3 / 作者 2 / 转写 1。
    """
    q = query.strip()
    if not q:
        return []
    if len(q) < 3:
        terms = [q]  # 1~2 字：FTS 用不了，整串走 LIKE
    elif len(q) <= 4:
        # 3~4 字短查询：先当整串短语（trigram 子串匹配最精确），再叠加分词结果
        terms = [q] + ask_terms(q)
    else:
        terms = ask_terms(q)
    long_terms = [t for t in terms if len(t) >= 3]  # trigram 可检索
    short_terms = [t for t in terms if len(t) < 3]  # 双字词 → LIKE

    cols = ("f.title LIKE ? OR f.tags LIKE ? OR f.author LIKE ? "
            "OR IFNULL(f.transcript,'') LIKE ?")
    valid = "f.transcript IS NOT NULL AND f.transcript NOT LIKE '【%'"
    select = "SELECT f.aweme_id, f.title, f.tags, f.author, f.share_url, f.transcript "
    ranks = "ORDER BY bm25(favorites_fts, 8.0, 3.0, 2.0, 1.0) LIMIT ?"

    def like_args(term: str) -> list[str]:
        p = f"%{term}%"
        return [p, p, p, p]

    with get_conn() as conn:
        if long_terms:
            base = (select + "FROM favorites_fts t JOIN favorites f ON f.rowid = t.rowid "
                    "WHERE ")
            scope = "f.rowid = t.rowid AND " + valid
            # ① 长词全命中 + 短词也都出现：最相关
            where = "favorites_fts MATCH ?"
            args: list = [" AND ".join(f'"{t}"' for t in long_terms)]
            for t in short_terms:
                where += f" AND ({cols})"
                args += like_args(t)
            rows = conn.execute(
                f"{base}{where} AND {scope} {ranks}", (*args, limit)
            ).fetchall()
            if rows:
                return rows
            # ② 放宽到「任一长词命中」
            rows = conn.execute(
                f"{base}favorites_fts MATCH ? AND {scope} {ranks}",
                (" OR ".join(f'"{t}"' for t in long_terms), limit),
            ).fetchall()
            if rows:
                return rows
        # ③ 短词 LIKE（或整句不足 3 字的短查询）：任一命中都算候选；
        #    多取几倍再按「命中词数 + 标题优先」重排，避免只按 rowid 出锅
        if short_terms:
            where = " OR ".join(f"({cols})" for _ in short_terms)
            args = []
            for t in short_terms:
                args += like_args(t)
            rows = conn.execute(
                f"{select}FROM favorites f WHERE ({where}) AND {valid} LIMIT ?",
                (*args, limit * 5),
            ).fetchall()
            if rows:
                return _like_rank(rows, short_terms, limit)
        # ④ 兜底：整句 LIKE
        return conn.execute(
            f"{select}FROM favorites f WHERE ({cols}) AND {valid} LIMIT ?",
            (*like_args(q), limit),
        ).fetchall()


def _like_rank(rows: list[sqlite3.Row], terms: list[str], limit: int) -> list[sqlite3.Row]:
    """LIKE 召回结果的朴素排序：标题命中(3) > 标签/作者(2) > 转写(1)，累计得分降序。"""
    def score(r: sqlite3.Row) -> int:
        title, tags = r["title"] or "", r["tags"] or ""
        author, text = r["author"] or "", r["transcript"] or ""
        s = 0
        for t in terms:
            if t in title:
                s += 3
            elif t in tags or t in author:
                s += 2
            elif t in text:
                s += 1
        return s

    return sorted(rows, key=lambda r: -score(r))[:limit]


def stats() -> dict:
    """库内概况，用于验收。"""
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM favorites").fetchone()[0]
        transcribed = conn.execute(
            "SELECT COUNT(*) FROM favorites WHERE transcript IS NOT NULL AND transcript NOT LIKE '【%'"
        ).fetchone()[0]
        summarized = conn.execute(
            "SELECT COUNT(*) FROM favorites WHERE summary IS NOT NULL"
        ).fetchone()[0]
        return {
            "total": total,
            "transcribed": transcribed,
            "summarized": summarized,
            "db": str(DB_PATH),
        }


# ---------- 后台任务持久化 ----------

def create_job(name: str) -> int:
    """创建一个运行中的任务，返回任务 id。"""
    now = time.time()
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO jobs
               (name, status, started_at, updated_at)
               VALUES (?, 'running', ?, ?)""",
            (name, now, now),
        )
        return int(cur.lastrowid)


def update_job(job_id: int, **fields) -> None:
    """更新任务的可变状态字段。字段名白名单防止拼接任意 SQL 列名。"""
    allowed = {"status", "progress", "error", "done", "total", "cancel"}
    values = {k: v for k, v in fields.items() if k in allowed}
    if not values:
        return
    values["updated_at"] = time.time()
    assignments = ", ".join(f"{key} = ?" for key in values)
    with get_conn() as conn:
        conn.execute(
            f"UPDATE jobs SET {assignments} WHERE id = ?",
            (*values.values(), job_id),
        )


def finish_job(job_id: int, status: str, progress: str = "", error: str = "",
               done: int = 0, total: int = 0) -> None:
    """以终态写入任务，并记录结束时间。"""
    now = time.time()
    with get_conn() as conn:
        conn.execute(
            """UPDATE jobs
               SET status = ?, progress = ?, error = ?, done = ?, total = ?,
                   finished_at = ?, updated_at = ?
               WHERE id = ?""",
            (status, progress, error, done, total, now, now, job_id),
        )


def recent_jobs(limit: int = 20) -> list[dict]:
    """返回最近任务，按时间正序，便于前端从旧到新展示。"""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT id, name, status, progress, error, done, total,
                      started_at, finished_at
               FROM jobs ORDER BY id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    result = []
    for row in reversed(rows):
        end = row["finished_at"] or time.time()
        result.append({
            "id": row["id"],
            "name": row["name"],
            "status": row["status"],
            "progress": row["progress"],
            "error": row["error"],
            "done": row["done"],
            "total": row["total"],
            "seconds": max(0, round(end - row["started_at"])),
        })
    return result


def delete_job(job_id: int) -> None:
    """删除一条任务历史，不影响收藏数据。"""
    with get_conn() as conn:
        conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))


def clear_jobs() -> None:
    """清空任务历史，不影响收藏数据。"""
    with get_conn() as conn:
        conn.execute("DELETE FROM jobs")


def list_videos(limit: int = 30, offset: int = 0, category: str | None = None) -> list[sqlite3.Row]:
    """按采集时间倒序列出视频（收藏库页），可按分类过滤；"__none__" = 未分类。"""
    with get_conn() as conn:
        if category == "__none__":
            return conn.execute(
                """SELECT aweme_id, title, tags, author, share_url, transcript, summary, category
                   FROM favorites WHERE category IS NULL ORDER BY crawled_at DESC LIMIT ? OFFSET ?""",
                (limit, offset),
            ).fetchall()
        if category:
            return conn.execute(
                """SELECT aweme_id, title, tags, author, share_url, transcript, summary, category
                   FROM favorites WHERE category = ? ORDER BY crawled_at DESC LIMIT ? OFFSET ?""",
                (category, limit, offset),
            ).fetchall()
        return conn.execute(
            """SELECT aweme_id, title, tags, author, share_url, transcript, summary, category
               FROM favorites ORDER BY crawled_at DESC LIMIT ? OFFSET ?""",
            (limit, offset),
        ).fetchall()


def dumps_tags(tag_list: list) -> str:
    """标签列表 → JSON 字符串入库（ensure_ascii=False 保留中文可读）。"""
    return json.dumps(tag_list, ensure_ascii=False)


# ---------- 阶段二：转写与概要 ----------

def get_untranscribed(limit: int, ids: list[str] | None = None) -> list[sqlite3.Row]:
    """还没转写的视频（transcript 为 NULL），新的优先；传 ids 时只查指定视频。"""
    with get_conn() as conn:
        if ids:
            marks = ",".join("?" * len(ids))
            return conn.execute(
                f"""SELECT aweme_id, title, author FROM favorites
                    WHERE transcript IS NULL AND aweme_id IN ({marks})""",
                ids,
            ).fetchall()
        return conn.execute(
            """SELECT aweme_id, title, author FROM favorites
               WHERE transcript IS NULL
               ORDER BY crawled_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()


def set_transcript(aweme_id: str, text: str) -> None:
    """写入转写文本。UPDATE 会触发 favorites_au，FTS 索引自动同步。"""
    with get_conn() as conn:
        conn.execute(
            "UPDATE favorites SET transcript = ? WHERE aweme_id = ?",
            (text, aweme_id),
        )


def get_unsummarized(limit: int) -> list[sqlite3.Row]:
    """已转写但没生成概要的视频。"""
    with get_conn() as conn:
        return conn.execute(
            """SELECT aweme_id, title, author, transcript FROM favorites
               WHERE transcript IS NOT NULL AND summary IS NULL
               ORDER BY crawled_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()


def set_summary(aweme_id: str, summary: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE favorites SET summary = ? WHERE aweme_id = ?",
            (summary, aweme_id),
        )


def get_video(aweme_id: str) -> sqlite3.Row | None:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM favorites WHERE aweme_id = ?", (aweme_id,)
        ).fetchone()


def count_untranscribed() -> int:
    with get_conn() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM favorites WHERE transcript IS NULL"
        ).fetchone()[0]


def count_unsummarized() -> int:
    with get_conn() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM favorites WHERE transcript IS NOT NULL AND summary IS NULL"
        ).fetchone()[0]


# ---------- 语义向量分块（问 AI 混合检索用） ----------

def all_videos() -> list[sqlite3.Row]:
    """索引所需的全部视频字段（indexer 构建向量索引用）。"""
    with get_conn() as conn:
        return conn.execute(
            """SELECT aweme_id, title, tags, author, transcript, summary
               FROM favorites ORDER BY crawled_at DESC"""
        ).fetchall()


def replace_chunks(aweme_id: str, chunks: list[dict], sig: str = "") -> None:
    """整体重写某视频的向量分块。chunks: [{source, idx, text, vec(bytes)}]。"""
    with get_conn() as conn:
        conn.execute("DELETE FROM chunks WHERE aweme_id = ?", (aweme_id,))
        conn.executemany(
            "INSERT INTO chunks (aweme_id, source, idx, text, vec, sig) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [(aweme_id, c["source"], c["idx"], c["text"], c["vec"], sig) for c in chunks],
        )


def all_chunks() -> list[sqlite3.Row]:
    """全部向量块（retriever 载入内存做暴力余弦）。"""
    with get_conn() as conn:
        return conn.execute(
            "SELECT aweme_id, source, idx, text, vec FROM chunks"
        ).fetchall()


def count_chunks() -> int:
    with get_conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]


def chunk_sigs() -> dict[str, str]:
    """每个视频当前已建索引的内容签名（增量构建时比对内容是否变化）。"""
    with get_conn() as conn:
        return {
            r[0]: r[1]
            for r in conn.execute("SELECT aweme_id, sig FROM chunks GROUP BY aweme_id")
        }


def drop_chunks_not_in(aweme_ids: list[str]) -> int:
    """删除不在给定集合里的向量块（视频被删/不再有内容时留下的孤儿块）。

    返回清理掉的视频数。之所以不「整表清空再重建」，是为了让全量重建在
    中途失败时索引仍然可用（旧块还在，下一轮会覆盖），而不是变成空的。
    """
    keep = set(aweme_ids)
    with get_conn() as conn:
        stale = [
            r[0]
            for r in conn.execute("SELECT DISTINCT aweme_id FROM chunks").fetchall()
            if r[0] not in keep
        ]
        for aweme_id in stale:
            conn.execute("DELETE FROM chunks WHERE aweme_id = ?", (aweme_id,))
    return len(stale)


# ---------- 分类 ----------

def count_unclassified() -> int:
    with get_conn() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM favorites WHERE category IS NULL"
        ).fetchone()[0]


def get_unclassified(limit: int) -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            """SELECT aweme_id, title, tags, author FROM favorites
               WHERE category IS NULL LIMIT ?""",
            (limit,),
        ).fetchall()


def set_category(aweme_id: str, category: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE favorites SET category = ? WHERE aweme_id = ?", (category, aweme_id)
        )


def reset_categories() -> None:
    """清空所有分类（换类别集合后重新分类用）。"""
    with get_conn() as conn:
        conn.execute("UPDATE favorites SET category = NULL")


def get_meta(key: str) -> str | None:
    with get_conn() as conn:
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None


def set_meta(key: str, value: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )


def del_meta(key: str) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM meta WHERE key = ?", (key,))


def sample_videos(limit: int) -> list[sqlite3.Row]:
    """随机抽样若干条视频（标题+标签），供 LLM 设计分类集合用。"""
    with get_conn() as conn:
        return conn.execute(
            "SELECT title, tags FROM favorites ORDER BY RANDOM() LIMIT ?", (limit,)
        ).fetchall()


def category_counts() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT category, COUNT(*) AS n FROM favorites
               WHERE category IS NOT NULL GROUP BY category ORDER BY n DESC"""
        ).fetchall()
        return [{"category": r["category"], "n": r["n"]} for r in rows]
