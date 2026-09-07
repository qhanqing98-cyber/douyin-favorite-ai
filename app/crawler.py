"""采集层：登录态管理、滚动加载、接口拦截、字段解析。

核心思路（不逆向签名）：抖音页面自己发带签名的收藏夹接口请求，
我们用 page.on("response") 当"旁听者"，把响应 JSON 捡走解析。
签名、风控参数全部由真实浏览器生成，我们零逆向。
"""
import json
import random
import time
from pathlib import Path

from playwright.sync_api import BrowserContext, Page, Response, sync_playwright

# 登录态目录：浏览器 profile 整个落盘，第二次启动免扫码
USER_DATA_DIR = Path(__file__).resolve().parent.parent / "browser_data"

# 个人主页。收藏数据必须点击"收藏"tab 才会请求（实测带 showTab 参数
# 直接打开不触发接口调用），所以先到主页再模拟点击。
FAVORITE_URL = "https://www.douyin.com/user/self"
FAVORITES_TAB_TEXT = "收藏"

# 接口 URL 特征：首页用 .../aweme/favorite/，翻页后改用 .../aweme/listcollection/
# （实测：两个都含 aweme_list 字段），其他请求一律放过
_API_HINTS = ("favorite", "listcollection")


def _open_browser(headless: bool = False) -> tuple[object, BrowserContext]:
    """启动持久化上下文浏览器。

    launch_persistent_context = 带 user_data_dir 的浏览器，
    cookies / localStorage 全部落盘，相当于一个独立的 Chrome 配置目录。
    有头模式更像真人，风控风险最低。

    sync_playwright().start() 返回 driver，用完必须 pw.stop() 释放进程。
    """
    pw = sync_playwright().start()
    ctx = pw.chromium.launch_persistent_context(
        user_data_dir=str(USER_DATA_DIR),
        headless=headless,
        viewport={"width": 1380, "height": 900},
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        ),
    )
    return pw, ctx


def _has_login_cookie(ctx: BrowserContext) -> bool:
    """检查是否已有登录态 cookie。sessionid / sessionid_ss 是抖音的会话凭证。"""
    names = {c["name"] for c in ctx.cookies("https://www.douyin.com")}
    return bool(names & {"sessionid", "sessionid_ss"})


def login(timeout_s: int = 300, progress=None, should_stop=None) -> bool:
    """打开有头浏览器，轮询等待扫码；登录态自动落盘。

    不用 input() 阻塞——每 3 秒查一次 cookies，检测到 sessionid 即成功。
    progress(msg): 可选，向 Web 端汇报等待状态。
    should_stop: 可选，返回 True 时放弃等待（协作式取消）。
    返回 True 表示登录成功。
    """
    def report(msg: str) -> None:
        print(msg, flush=True)
        if progress:
            progress(msg)

    pw, ctx = _open_browser()
    page = ctx.pages[0] if ctx.pages else ctx.new_page()
    page.goto("https://www.douyin.com/", wait_until="domcontentloaded")
    report("→ 若页面未自动弹出二维码，请点击右上角『登录』后扫码...")
    deadline = time.time() + timeout_s
    ok = False
    while time.time() < deadline:
        if should_stop and should_stop():
            report("收到取消信号，放弃登录。")
            break
        if _has_login_cookie(ctx):
            ok = True
            break
        report(f"等待扫码中...（已等 {int(time.time() - (deadline - timeout_s))} 秒）")
        time.sleep(3)
    ctx.close()
    pw.stop()
    if ok:
        report(f"登录成功，登录态已保存到 {USER_DATA_DIR}")
    elif not (should_stop and should_stop()):
        report("等待超时，未检测到登录。请重跑 login 再扫一次。")
    return ok


def parse_aweme(item: dict) -> dict | None:
    """把接口返回的单条视频 dict 解析成库表结构。改版时只需修这个函数。

    全程用 .get()：抖音改版少字段/改字段名时返回空值，
    而不是抛 KeyError 让整个采集崩掉。
    """
    aweme_id = item.get("aweme_id")
    if not aweme_id:
        return None
    # text_extra 里混着 @用户 和 #话题，只取带 hashtag_name 的（即标签）
    tags = [
        t.get("hashtag_name")
        for t in (item.get("text_extra") or [])
        if t.get("hashtag_name")
    ]
    # play_addr.url_list 是数组，取第一个直链（有时效，仅作参考存档）
    url_list = ((item.get("video") or {}).get("play_addr") or {}).get("url_list") or []
    return {
        "aweme_id": aweme_id,
        "title": (item.get("desc") or "").strip(),
        "tags": json.dumps(tags, ensure_ascii=False),
        "author": (item.get("author") or {}).get("nickname") or "",
        "share_url": item.get("share_url") or "",
        "play_url": url_list[0] if url_list else "",
        "fav_time": item.get("create_time") or 0,
    }


def _scroll_page(page: Page) -> None:
    """推进一屏滚动。抖音个人页的 window 不可滚动（实测 scrollY 恒 0），
    滚动条挂在 .route-scroll-container 上，必须直接推它的 scrollTop；
    找不到该容器时退回 window.scrollBy。
    """
    page.evaluate(
        """() => {
            const el = document.querySelector('.route-scroll-container');
            if (el) { el.scrollTop += 2000; return; }
            window.scrollBy(0, 2000);
        }"""
    )


def export_cookies(out_path: Path | None = None) -> Path:
    """把持久化 profile 里的 cookies 导出为 Netscape cookies.txt。

    yt-dlp 只认这个格式。用无头浏览器启动同一 profile 读取最新 cookie，
    不影响已有登录态。目标站限定 douyin.com，避免把无关 cookie 写进去。
    """
    out = out_path or (Path(__file__).resolve().parent.parent / "data" / "cookies.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    pw, ctx = _open_browser(headless=True)
    try:
        cookies = ctx.cookies("https://www.douyin.com")
    finally:
        ctx.close()
        pw.stop()
    lines = ["# Netscape HTTP Cookie File"]
    for c in cookies:
        domain = c["domain"]
        # Netscape 规范：includeSubdomains=TRUE 时域必须以点开头，
        # 否则 yt-dlp 的 cookiejar 判为非法行、整个文件被拒
        subdomains = "TRUE" if domain.startswith(".") else "FALSE"
        secure = "TRUE" if c.get("secure") else "FALSE"
        expiry = int(c.get("expires", 0)) or 0
        lines.append(
            f"{domain}\t{subdomains}\t{c['path']}\t{secure}\t{expiry}\t{c['name']}\t{c['value']}"
        )
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def crawl(max_rounds: int = 400, progress=None, should_stop=None) -> int:
    """滚动收藏夹页面并拦截接口响应，把新视频入库。返回新增条数。

    终止条件：连续 3 轮滚动捕获量不增长 → 认为加载到底。
    progress(msg): 可选，向 Web 端汇报采集状态。
    should_stop: 可选，返回 True 时停止滚动，但已捕获的数据照常入库。
    """
    from app import db

    def report(msg: str) -> None:
        print(msg, flush=True)
        if progress:
            progress(msg)

    buffer: list[dict] = []      # 原始 aweme dict 暂存
    seen_ids: set[str] = set()   # 缓冲区内去重（接口分页可能有重叠）
    state = {"has_more": None}   # 接口权威信号：0 = 没有下一页

    def on_response(resp: Response) -> None:
        """监听器：URL 命中任一接口特征才看；JSON 里要有 aweme_list 才收。"""
        if not any(h in resp.url for h in _API_HINTS):
            return
        try:
            data = resp.json()
        except Exception:
            return  # 图片/HTML 等非 JSON 响应直接放过
        if "aweme_list" not in data:
            return
        state["has_more"] = data.get("has_more")
        for item in data.get("aweme_list") or []:
            aid = item.get("aweme_id")
            if aid and aid not in seen_ids:
                seen_ids.add(aid)
                buffer.append(item)

    pw, ctx = _open_browser()
    page: Page = ctx.pages[0] if ctx.pages else ctx.new_page()
    page.on("response", on_response)  # 必须在导航前注册，否则漏掉首屏请求

    page.goto(FAVORITE_URL, wait_until="domcontentloaded")
    time.sleep(6)  # 等个人主页渲染；过早点击会落在未挂载的元素上
    report("→ 尝试自动点击『收藏』标签（最多 3 次）...")
    for attempt in range(3):
        if should_stop and should_stop():
            break
        try:
            page.get_by_text(FAVORITES_TAB_TEXT, exact=True).first.click(timeout=5000)
            report(f"第 {attempt + 1} 次点击完成")
        except Exception:
            report("自动点击失败，请在浏览器窗口里手动点开你的收藏夹...")
        # 点击后等数据：20 秒内接口回来就继续，否则重试点击
        for _ in range(10):
            if buffer or (should_stop and should_stop()):
                break
            time.sleep(2)
        if buffer:
            break
    if not buffer and not (should_stop and should_stop()):
        report("→ 自动点击未拿到数据，等你手动点开收藏夹（最长 90 秒）...")
        deadline = time.time() + 90
        while time.time() < deadline and not buffer:
            if should_stop and should_stop():
                break
            time.sleep(2)
    if not buffer:
        report("未捕获到收藏数据：请确认已登录、页面停在收藏夹列表，再重试。")
        ctx.close()
        pw.stop()
        return 0

    stale_rounds = 0   # 连续无增长的轮数
    last_count = 0     # 上一轮结束时的捕获量
    for round_no in range(max_rounds):
        if should_stop and should_stop():
            report("收到取消信号，停止滚动，已捕获的数据照常入库。")
            break
        _scroll_page(page)                     # JS 推滚动容器
        time.sleep(random.uniform(2.0, 3.5))   # 随机停顿，降低风控风险
        report(f"滚动第 {round_no + 1} 轮，已捕获 {len(buffer)} 条")
        if state.get("has_more") == 0 and len(buffer) > last_count:
            report("接口返回 has_more=0，已到收藏夹最后一页。")
            break
        if len(buffer) > last_count:
            last_count = len(buffer)
            stale_rounds = 0
        else:
            stale_rounds += 1
            if stale_rounds >= 3:
                report("连续 3 轮无新数据，判定已到收藏夹底部。")
                break

    ctx.close()
    pw.stop()

    parsed = [p for p in (parse_aweme(it) for it in buffer) if p]
    inserted = db.save_favorites(parsed)
    report(f"捕获 {len(parsed)} 条，新入库 {inserted} 条（其余为重复，已跳过）")
    return inserted
