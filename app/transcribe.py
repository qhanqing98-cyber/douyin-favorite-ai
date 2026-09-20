"""转写流水线：浏览器/yt-dlp 下载音频 → faster-whisper 本地转写 → 入库。

数据流：favorites 表里 transcript IS NULL 的行 → 取 aweme_id 拼 https://www.douyin.com/video/{id}
      → yt-dlp（带 cookies.txt）或已登录浏览器下音频到 audio_cache/ → Whisper 转文本
      → set_transcript() 写回（FTS 自动同步）→ 音频文件保留供重复调试，可手动清。
下载失败不会写入占位 transcript；登录失效会提示用户重新登录，普通下载失败的条目保留为待重试。
"""
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


def _configure_console_encoding() -> None:
    """后台线程可能处理包含 emoji 的标题，Windows GBK 输出不能让任务失败。"""
    stream = getattr(sys, "stdout", None)
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_configure_console_encoding()

# HuggingFace 在国内直连不稳，默认走 hf-mirror 镜像下载 Whisper 模型。
# 必须在 import faster_whisper 之前设置（huggingface_hub 读 import 时的环境变量）。
# HF_HUB_DISABLE_XET=1：新版 hub 默认用 Xet 协议下载，hf-mirror 不支持（401），
# 禁用后退回普通 HTTP resolve 通道。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

PROJECT = Path(__file__).resolve().parent.parent
AUDIO_DIR = PROJECT / "audio_cache"
COOKIES = PROJECT / "data" / "cookies.txt"

_model = None  # 模型很重，进程内只加载一次
_opencc = None

LOGIN_COOKIE_NAMES = {
    "sessionid", "sessionid_ss", "sid_guard", "uid_tt", "uid_tt_ss",
}


class TranscriptionError(RuntimeError):
    """转写失败，但错误可以安全展示给用户。"""

    code = "transcription_error"

    def __init__(self, message: str, *, processed: int = 0, requested: int = 0):
        super().__init__(message)
        self.processed = processed
        self.requested = requested


class LoginRequiredError(TranscriptionError):
    code = "login_required"


@dataclass
class TranscriptionReport:
    requested: int
    processed: int
    processed_ids: list[str] = field(default_factory=list)
    failed: list[dict] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def average_seconds(self) -> float | None:
        return self.elapsed_seconds / self.processed if self.processed else None

    def as_dict(self) -> dict:
        return {
            "requested": self.requested,
            "processed": self.processed,
            "processed_ids": self.processed_ids,
            "failed": self.failed,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "average_seconds": round(self.average_seconds, 1) if self.average_seconds else None,
        }


def _cookie_names(path: Path) -> set[str]:
    if not path.exists():
        return set()
    names: set[str] = set()
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line or line.startswith("#"):
                continue
            columns = line.split("\t")
            if len(columns) >= 7:
                names.add(columns[5].strip().lower())
    except OSError:
        return set()
    return names


def _has_login_cookies() -> bool:
    return bool(_cookie_names(COOKIES) & LOGIN_COOKIE_NAMES)


def _ensure_login_cookies(crawler) -> None:
    """每次转写前刷新浏览器 Cookie，避免沿用已经失效的 cookies.txt。"""
    try:
        crawler.export_cookies()
    except Exception as exc:
        raise LoginRequiredError(
            "没有检测到可用的抖音登录状态，请先到“数据维护”点击“扫码登录”，登录成功后再重新开始转写。"
        ) from exc
    if not _has_login_cookies():
        raise LoginRequiredError(
            "没有检测到可用的抖音登录状态，请先到“数据维护”点击“扫码登录”，登录成功后再重新开始转写。"
        )


def _to_simplified(text: str) -> str:
    """繁体 → 简体。Whisper 中文输出常混繁体字，而搜索输入是简体，
    trigram 不做字形归一化，不转换会漏召回（实测"人性規律"搜不到"规律"）。"""
    global _opencc
    if _opencc is None:
        from opencc import OpenCC

        _opencc = OpenCC("t2s")
    return _opencc.convert(text)


def _get_model(model_size: str):
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        # 优先用项目内 models/ 下手动下载的模型（HF 的 Xet 协议在国内
        # 经镜像会拿到 0 字节文件，不可靠），否则回落 HF 模型名
        local = PROJECT / "models" / f"faster-whisper-{model_size}"
        target = str(local) if local.is_dir() else model_size
        print(f"加载 Whisper 模型 {target}（首次会下载权重）...", flush=True)
        _model = WhisperModel(target, device="auto", compute_type="int8")
    return _model


def _looks_like_login_error(message: str) -> bool:
    text = message.lower()
    markers = (
        "cookies are no longer valid", "invalid cookie", "login required",
        "please log in", "sign in", "authentication", "session expired",
        "需要登录", "登录后", "未登录", "登录状态已失效",
    )
    return any(marker in text for marker in markers)


def _is_cookie_challenge(message: str) -> bool:
    return "fresh cookies" in message.lower()


def _has_audio_stream(path: Path) -> bool:
    """确认缓存文件确实包含音频轨，避免复用浏览器捕获到的视频-only 文件。"""
    import av

    try:
        container = av.open(str(path))
        has_audio = any(stream.type == "audio" for stream in container.streams)
        container.close()
        return has_audio
    except Exception:
        return False


def _download_audio_via_browser(aweme_id: str) -> tuple[Path | None, str]:
    """用已登录的 Playwright 会话获取当前媒体地址，绕过 yt-dlp 的网页验证挑战。"""
    import av
    import io

    from app.crawler import _open_browser

    pw = None
    ctx = None
    page = None
    media_urls: list[str] = []
    audio_urls: list[str] = []
    try:
        pw, ctx = _open_browser(headless=True)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()

        def on_response(response) -> None:
            url = response.url
            if response.status in {200, 206} and "douyinvod.com" in url and url not in media_urls:
                media_urls.append(url)
                if "media-audio" in url:
                    audio_urls.append(url)

        page.on("response", on_response)
        page.goto(
            f"https://www.douyin.com/video/{aweme_id}",
            wait_until="domcontentloaded",
            timeout=30_000,
        )
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline and not media_urls:
            page.wait_for_timeout(1000)

        # 页面通常同时请求视频和音频，音频 URL 优先；视频 URL 只作为兼容回退。
        candidates = list(dict.fromkeys(reversed(audio_urls + media_urls)))
        for media_url in candidates:
            try:
                response = ctx.request.get(
                    media_url,
                    headers={
                        "Referer": "https://www.douyin.com/",
                        "User-Agent": "Mozilla/5.0",
                    },
                    timeout=30_000,
                )
                body = response.body() if response.status == 200 else b""
                if len(body) < 1024:
                    continue
                container = av.open(io.BytesIO(body))
                has_audio = any(stream.type == "audio" for stream in container.streams)
                container.close()
                if not has_audio:
                    continue
                path = AUDIO_DIR / f"{aweme_id}.mp4"
                path.write_bytes(body)
                return path, ""
            except Exception:
                continue
        return None, "抖音浏览器验证未返回可用的媒体文件"
    except Exception as exc:
        return None, f"浏览器下载失败：{str(exc)[:180]}"
    finally:
        if page is not None:
            try:
                page.close()
            except Exception:
                pass
        if ctx is not None:
            try:
                ctx.close()
            except Exception:
                pass
        if pw is not None:
            try:
                pw.stop()
            except Exception:
                pass


def _download_audio(aweme_id: str, retries: int = 3) -> tuple[Path | None, str]:
    """yt-dlp 下载该视频的音频轨。成功返回文件路径，失败返回 None。

    用 sys.executable -m yt_dlp 保证调用的是当前 venv 里的 yt-dlp。
    -f bestaudio/best：优先纯音频轨；没有就下整个视频，
    faster-whisper 底层的 PyAV 能直接从视频容器里解出音轨，无需 ffmpeg。
    如果 yt-dlp 被抖音的网页验证挑战拦截，则回退到已登录的浏览器会话获取媒体地址。
    """
    import random
    import subprocess

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(AUDIO_DIR.glob(f"{aweme_id}.*"))
    for path in existing:
        if _has_audio_stream(path):
            return path, ""
    last_err = "未找到可下载的音频"
    for attempt in range(retries):
        cmd = [
            sys.executable, "-m", "yt_dlp",
            "--cookies", str(COOKIES),
            "--no-warnings", "--no-playlist", "--quiet",
            "-f", "bestaudio/best",
            "-o", str(AUDIO_DIR / f"{aweme_id}.%(ext)s"),
            f"https://www.douyin.com/video/{aweme_id}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode == 0:
            files = list(AUDIO_DIR.glob(f"{aweme_id}.*"))
            for path in files:
                if _has_audio_stream(path):
                    return path, ""
            last_err = "下载完成但没有找到可用的音频轨"
            continue
        last_err = (result.stderr or result.stdout or "下载失败").strip()[-240:]
        if _looks_like_login_error(last_err):
            break
        if attempt < retries - 1:
            time.sleep(random.uniform(3, 8))
    if _is_cookie_challenge(last_err):
        print("\n  yt-dlp 触发抖音网页验证，改用已登录浏览器下载...", flush=True)
        browser_path, browser_error = _download_audio_via_browser(aweme_id)
        if browser_path is not None:
            return browser_path, ""
        last_err = browser_error or last_err
    print(f"\n  音频下载失败(重试{retries}次): {last_err}", flush=True)
    return None, last_err


def _transcribe_file(path: Path, model_size: str) -> str:
    """跑 Whisper，返回拼接后的全文。vad_filter 过滤静音段，提速明显。"""
    model = _get_model(model_size)
    segments, info = model.transcribe(str(path), language="zh", vad_filter=True)
    parts = [seg.text.strip() for seg in segments]
    return "".join(parts).strip()


def run(limit: int = 10, model_size: str = "small", progress=None, ids=None, should_stop=None) -> TranscriptionReport:
    """批量转写；下载失败不会写入占位 transcript，保留为待重试状态。

    progress: 可选回调 progress(done, total, title)，Web 端用来更新进度和预计剩余时间。
    ids: 可选，只转写这些 aweme_id（搜索结果勾选的按需转写）。
    should_stop: 可选回调，返回 True 时在当前视频完成后停止（协作式取消）。
    """
    from app import crawler, db

    rows = db.get_untranscribed(limit, ids=ids)
    total = len(rows)
    print(f"待转写 {total} 条", flush=True)
    if not rows:
        return TranscriptionReport(requested=0, processed=0)
    _ensure_login_cookies(crawler)
    started_at = time.monotonic()
    processed = 0
    processed_ids: list[str] = []
    failed: list[dict] = []
    for i, row in enumerate(rows):
        if should_stop and should_stop():
            print("收到取消信号，停止转写。", flush=True)
            break
        if i > 0:
            time.sleep(random.uniform(2, 5))  # 请求间隔，降低限流风险
        if progress:
            progress(i, total, f"正在处理：{row['title']}")
        aid = row["aweme_id"]
        print(f"[{i + 1}/{len(rows)}] {row['title'][:30]}", end="", flush=True)
        audio, download_error = _download_audio(aid)
        if audio is None:
            if _looks_like_login_error(download_error):
                raise LoginRequiredError(
                    "抖音登录状态已失效，无法下载视频音频。请先到“数据维护”点击“扫码登录”，"
                    "登录成功后再重新开始转写。",
                    processed=processed,
                    requested=total,
                )
            failed.append({"aweme_id": aid, "title": row["title"], "reason": download_error})
            print(" → 下载失败（保留待重试）", flush=True)
            if progress:
                progress(i + 1, total, f"下载失败：{row['title']}")
            continue
        try:
            t0 = time.time()
            text = _transcribe_file(audio, model_size)
            text = _to_simplified(text)
            db.set_transcript(aid, text or f"【无语音内容】{row['title']}")
            print(f" → {len(text)} 字，{time.time() - t0:.0f}s", flush=True)
            processed += 1
            processed_ids.append(aid)
        except Exception as e:
            message = str(e)[:240]
            failed.append({"aweme_id": aid, "title": row["title"], "reason": message})
            print(f" → 转写异常: {message}", flush=True)
        if progress:
            progress(i + 1, total, f"已处理：{row['title']}")
    return TranscriptionReport(
        requested=total,
        processed=processed,
        processed_ids=processed_ids,
        failed=failed,
        elapsed_seconds=time.monotonic() - started_at,
    )
