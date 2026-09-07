"""SQLite 存储层：建表、写入、FTS5 中文搜索。

数据流：crawler 解析出的 dict 列表 → save_favorites() 入库（自动同步 FTS 索引）
        → search() 用关键词召回 → 返回视频列表。
"""
import json
import sqlite3
import time
from pathlib import Path

# 数据库固定放在项目根目录 data/ 下；__file__ 是本文件路径，
# .parent 两次上溯到项目根，保证从任何工作目录运行都能找到同一个库。
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "favorites.db"


def get_conn() -> sqlite3.Connection:
    """返回一个数据库连接。row_factory 让查询结果可以用列名访问（row["title"]）。"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


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


def category_counts() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT category, COUNT(*) AS n FROM favorites
               WHERE category IS NOT NULL GROUP BY category ORDER BY n DESC"""
        ).fetchall()
        return [{"category": r["category"], "n": r["n"]} for r in rows]
