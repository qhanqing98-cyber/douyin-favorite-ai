"""CLI 入口：login / crawl / search / transcribe / summarize / ask / index / stats。

用法：
    python main.py login                    # 首次：扫码登录，落盘登录态
    python main.py crawl                    # 打开收藏夹滚动采集入库（需已 login）
    python main.py search 关键词            # 全文搜索（标题/标签/作者/转写）
    python main.py transcribe [N]           # 转写 N 条（默认 10，--all 全部）
    python main.py summarize [N]            # 给已转写的视频生成 AI 概要
    python main.py ask 问题                 # 语义+关键词混合检索并让 AI 回答
    python main.py index [--all]            # 构建/更新语义向量索引（ask 时自动增量）
    python main.py stats                    # 查看库内条数
"""
import argparse
import json
import sys

from app import crawler, db


def cmd_search(query: str) -> None:
    rows = db.search(query)
    if not rows:
        print("没有命中。试试更短的关键词。")
        return
    for r in rows:
        tags = " ".join(f"#{t}" for t in json.loads(r["tags"] or "[]"))
        flag = " [已转写]" if ("transcript" in r.keys() and r["transcript"]) else ""
        print(f"【{r['title']}】{flag}")
        print(f"  作者: {r['author']}  标签: {tags}")
        print(f"  {r['share_url']}")
        print()


def cmd_transcribe(args) -> None:
    from app import transcribe

    n = 10**9 if args.all else (args.n or 10)
    ok = transcribe.run(limit=n, model_size=args.model)
    print(f"完成 {ok} 条转写")


def cmd_summarize(args) -> None:
    from app import db, llm

    rows = db.get_unsummarized(args.n or 10)
    if not rows:
        print("没有待生成概要的视频（需先 transcribe）。")
        return
    print(f"待生成概要 {len(rows)} 条")
    for i, row in enumerate(rows):
        print(f"[{i + 1}/{len(rows)}] {row['title'][:30]}", end="", flush=True)
        try:
            summary = llm.summarize(row["title"], row["author"], row["transcript"])
            db.set_summary(row["aweme_id"], summary)
            print(" → 完成", flush=True)
        except Exception as e:
            print(f" → 失败: {str(e)[:150]}", flush=True)


def cmd_ask(question: str) -> None:
    from app import indexer, llm, retriever

    try:  # 索引缺失/内容变更 → 自动增量构建；模型缺失则退化为纯关键词检索
        indexer.ensure_ready()
    except Exception as e:
        print(f"向量索引不可用（退化为纯关键词检索）：{str(e)[:150]}")
    hits = retriever.hybrid_search(question)
    if not hits:
        print("没找到相关内容（需要有转写/概要的视频），先跑 transcribe。")
        return
    tag = {"vector": "语义", "keyword": "关键词"}
    print("引用来源：")
    for h in hits:
        how = "+".join(tag.get(m, m) for m in h["matched_by"].split("+"))
        print(f"  【{h['title']}】[{how}]")
    print("\n回答：")
    try:
        print(llm.answer(question, retriever.build_contexts(hits)))
    except Exception as e:
        print(f"调用失败: {e}")


def cmd_index(args) -> None:
    from app import indexer

    def on_progress(done: int, t: int, title: str):
        print(f"\r[{done + 1}/{t}] {title[:36]}", end="", flush=True)

    n = indexer.build(all=args.all, progress=on_progress)
    print(f"\n索引完成：本次处理 {n} 个视频")


def main() -> None:
    # Windows 控制台默认 GBK：标题/概要里的 emoji 会让 print 抛 UnicodeEncodeError，
    # 这里统一把不可编码字符替换掉（只影响控制台输出，不影响入库数据）。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass
    db.init_db()
    parser = argparse.ArgumentParser(description="抖音收藏夹采集、转写与问答")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("login", help="扫码登录并保存登录态")
    sub.add_parser("crawl", help="滚动收藏夹采集入库")
    p_search = sub.add_parser("search", help="全文搜索")
    p_search.add_argument("query", help="搜索关键词")

    p_tr = sub.add_parser("transcribe", help="下载音频并转写入库")
    p_tr.add_argument("n", nargs="?", type=int, default=10, help="本次转写条数（默认 10）")
    p_tr.add_argument("--all", action="store_true", help="转写全部未转写视频")
    p_tr.add_argument("--model", default="small", help="whisper 模型: tiny/base/small/medium")

    p_sum = sub.add_parser("summarize", help="生成 AI 概要（需 .env 配置密钥）")
    p_sum.add_argument("n", nargs="?", type=int, default=10, help="本次处理条数（默认 10）")

    p_ask = sub.add_parser("ask", help="语义+关键词混合检索并 AI 问答")
    p_ask.add_argument("question", help="你的问题")

    p_idx = sub.add_parser("index", help="构建/更新语义向量索引（ask 时自动增量）")
    p_idx.add_argument("--all", action="store_true", help="清空全量重建（换模型后用）")

    p_web = sub.add_parser("web", help="启动本地 Web 页面")
    p_web.add_argument("--port", type=int, default=8642, help="监听端口（默认 8642）")

    sub.add_parser("stats", help="查看库内统计")

    args = parser.parse_args()
    if args.cmd == "login":
        crawler.login()
    elif args.cmd == "crawl":
        crawler.crawl()
    elif args.cmd == "search":
        cmd_search(args.query)
    elif args.cmd == "transcribe":
        cmd_transcribe(args)
    elif args.cmd == "summarize":
        cmd_summarize(args)
    elif args.cmd == "ask":
        cmd_ask(args.question)
    elif args.cmd == "index":
        cmd_index(args)
    elif args.cmd == "web":
        import uvicorn

        uvicorn.run("app.web:app", host="127.0.0.1", port=args.port, log_level="warning")
    elif args.cmd == "stats":
        print(db.stats())


if __name__ == "__main__":
    sys.exit(main())
