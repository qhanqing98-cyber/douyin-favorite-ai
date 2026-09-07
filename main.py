"""CLI 入口：login / crawl / search / transcribe / summarize / ask / stats。

用法：
    python main.py login                    # 首次：扫码登录，落盘登录态
    python main.py crawl                    # 打开收藏夹滚动采集入库（需已 login）
    python main.py search 关键词            # 全文搜索（标题/标签/作者/转写）
    python main.py transcribe [N]           # 转写 N 条（默认 10，--all 全部）
    python main.py summarize [N]            # 给已转写的视频生成 AI 概要
    python main.py ask 问题                 # 检索转写内容并让 AI 回答
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
    from app import db, llm

    rows = db.search(question, limit=5)
    with_transcript = [r for r in rows if "transcript" in r.keys() and r["transcript"]]
    if not with_transcript:
        print("没有带转写内容的命中视频，先跑 transcribe。")
        return
    contexts = [
        {
            "title": r["title"],
            "author": r["author"],
            "aweme_id": r["aweme_id"],
            "transcript": r["transcript"],
        }
        for r in with_transcript
    ]
    print("引用来源：")
    for c in contexts:
        print(f"  【{c['title']}】")
    print("\n回答：")
    try:
        print(llm.answer(question, contexts))
    except Exception as e:
        print(f"调用失败: {e}")


def main() -> None:
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

    p_ask = sub.add_parser("ask", help="检索转写内容并 AI 问答")
    p_ask.add_argument("question", help="你的问题")

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
    elif args.cmd == "web":
        import uvicorn

        uvicorn.run("app.web:app", host="127.0.0.1", port=args.port, log_level="warning")
    elif args.cmd == "stats":
        print(db.stats())


if __name__ == "__main__":
    sys.exit(main())
