"""转写流水线：yt-dlp 下载音频 → faster-whisper 本地转写 → 入库。

数据流：favorites 表里 transcript IS NULL 的行 → 取 aweme_id 拼 https://www.douyin.com/video/{id}
      → yt-dlp（带 cookies.txt）下音频到 audio_cache/ → Whisper 转文本
      → set_transcript() 写回（FTS 自动同步）→ 音频文件保留供重复调试，可手动清。
下载失败的行按参考方案兜底：transcript = "【音频不可用】标题"，保证可搜索且不无限重试。
"""
import os
import random
import sys
import time
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


def _download_audio(aweme_id: str, retries: int = 3) -> Path | None:
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
        return existing[0]
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
            return files[0] if files else None
        last_err = (result.stderr or "").strip()[-120:]
        if attempt < retries - 1:
            time.sleep(random.uniform(3, 8))
    print(f"\n  yt-dlp 失败(重试{retries}次): {last_err}", flush=True)
    return None


def _transcribe_file(path: Path, model_size: str) -> str:
    """跑 Whisper，返回拼接后的全文。vad_filter 过滤静音段，提速明显。"""
    model = _get_model(model_size)
    segments, info = model.transcribe(str(path), language="zh", vad_filter=True)
    parts = [seg.text.strip() for seg in segments]
    return "".join(parts).strip()


def run(limit: int = 10, model_size: str = "small", progress=None, ids=None) -> int:
    """批量转写。返回成功写入的条数（含兜底标记的也算写入）。

    progress: 可选回调 progress(done, total, title)，Web 端用来更新进度条。
    ids: 可选，只转写这些 aweme_id（搜索结果勾选的按需转写）。
    """
    from app import crawler, db

    if not COOKIES.exists():
        print("导出登录 cookies...", flush=True)
        crawler.export_cookies()

    rows = db.get_untranscribed(limit, ids=ids)
    total = len(rows)
    print(f"待转写 {total} 条", flush=True)
    ok = 0
    for i, row in enumerate(rows):
        if i > 0:
            time.sleep(random.uniform(2, 5))  # 请求间隔，降低限流风险
        if progress:
            progress(i, total, row["title"])
        aid = row["aweme_id"]
        print(f"[{i + 1}/{len(rows)}] {row['title'][:30]}", end="", flush=True)
        audio = _download_audio(aid)
        if audio is None:
            # 兜底：版权保护/已删除的视频下不了音频，用标题+作者占位，
            # 仍可被搜索召回，且 transcript 非空不会反复重试
            db.set_transcript(aid, f"【音频不可用】{row['title']} 作者:{row['author']}")
            print(" → 兜底标记", flush=True)
            ok += 1
            continue
        try:
            t0 = time.time()
            text = _transcribe_file(audio, model_size)
            text = _to_simplified(text)
            db.set_transcript(aid, text or f"【无语音内容】{row['title']}")
            print(f" → {len(text)} 字，{time.time() - t0:.0f}s", flush=True)
            ok += 1
        except Exception as e:
            print(f" → 转写异常: {str(e)[:120]}", flush=True)
    return ok
