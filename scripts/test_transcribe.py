"""转写并发与下载超时的离线回归测试。"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import transcribe


def test_download_timeout_is_bounded() -> None:
    with tempfile.TemporaryDirectory() as directory:
        with patch.object(transcribe, "AUDIO_DIR", Path(directory)):
            with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("yt-dlp", 120)):
                path, error = transcribe._download_audio("demo", retries=1)
    assert path is None
    assert "120" in error


def test_cancel_skips_subprocess() -> None:
    with tempfile.TemporaryDirectory() as directory:
        with patch.object(transcribe, "AUDIO_DIR", Path(directory)):
            with patch("subprocess.run") as run:
                path, error = transcribe._download_audio(
                    "demo", retries=1, should_stop=lambda: True,
                )
    assert path is None
    assert "取消" in error
    run.assert_not_called()


def test_transcription_lock_rejects_parallel_run() -> None:
    assert transcribe._transcription_lock.acquire(blocking=False)
    try:
        try:
            transcribe.run(limit=0)
        except transcribe.TranscriptionError as exc:
            assert "已有转写任务" in str(exc)
        else:
            raise AssertionError("parallel transcription must be rejected")
    finally:
        transcribe._transcription_lock.release()


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"Transcription tests: {len(tests)}/{len(tests)} passed")
