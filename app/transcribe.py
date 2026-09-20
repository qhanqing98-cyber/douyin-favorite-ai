"""转写流水线：yt-dlp 下载音频 → faster-whisper 本地转写 → 入库。

数据流：favorites 表里 transcript IS NULL 的行 → 取 aweme_id 拼 https://www.douyin.com/video/{id}
      → yt-dlp（带 cookies.txt）下音频到 audio_cache/ → Whisper 转文本
      → set_transcript() 写回（FTS 自动同步）→ 音频文件保留供重复调试，可手动清。
下载失败不会写入占位 transcript；登录失效会提示用户重新登录，普通下载失败的条目保留为待重试。
"""
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

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
    """优先从当前登录浏览器导出 cookie；没有登录态时给出可操作提示。"""
    if not _has_login_cookies():
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
        "fresh cookies", "cookies are no longer valid", "login", "sign in",
        "authentication", "需要登录", "登录", "cookie", "403 forbidden",
    )
    return any(marker in text for marker in markers)


def _download_audio(aweme_id: str, retries: int = 3) -> tuple[Path | None, str]:
    """yt-dlp 下载该视频的音频轨。成功返回文件路径，失败返回 None。

    用 sys.executable -m yt_dlp 保证调用的是当前 venv 里的 yt-dlp。
    -f bestaudio/best：优先纯音频轨；没有就下整个视频，
    faster-whisper 底层的 PyAV 能直接从视频容器里解出音轨，无需 ffmpeg。
    连续请求会触发抖音的间歇性 403 限流（"Fresh cookies" 报错），
    实测重试即可成功，所以失败后随机退避重试。
    """
    import random
    import subprocess

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(AUDIO_DIR.glob(f"{aweme_id}.*"))
    if existing:
        return existing[0], ""
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
            return (files[0], "") if files else (None, "下载完成但没有找到音频文件")
        last_err = (result.stderr or result.stdout or "下载失败").strip()[-240:]
        if _looks_like_login_error(last_err):
            break
        if attempt < retries - 1:
            time.sleep(random.uniform(3, 8))
    print(f"\n  yt-dlp 失败(重试{retries}次): {last_err}", flush=True)
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
