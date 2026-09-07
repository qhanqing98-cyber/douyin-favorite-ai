"""Pre-download the faster-whisper model into models/ (one-time, ~461 MB for small).

Called by start.bat. Downloads from hf-mirror.com (CN-friendly mirror),
shows per-file progress bars, and skips if the model already exists.

Usage: python download_model.py [tiny|base|small|medium]   (default: small)
"""
import os
import sys
from pathlib import Path

# Must be set before importing huggingface_hub:
# - hf-mirror.com is a CN-friendly mirror of huggingface.co
# - Xet protocol is unsupported by the mirror (returns 401), disable it
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

PROJECT = Path(__file__).resolve().parent.parent
SIZES = ("tiny", "base", "small", "medium")
# Approximate download sizes, shown to the user before starting
APPROX_MB = {"tiny": 75, "base": 145, "small": 461, "medium": 1530}


def main() -> int:
    size = sys.argv[1] if len(sys.argv) > 1 else "small"
    if size not in SIZES:
        print(f"[ERROR] Unknown model size '{size}'. Choose one of: {', '.join(SIZES)}")
        return 1

    dest = PROJECT / "models" / f"faster-whisper-{size}"
    marker = dest / "model.bin"
    if marker.is_file() and marker.stat().st_size > 10_000_000:
        print(f"[OK] Model already downloaded: {dest}")
        print("     (Skip. Delete this folder to re-download.)")
        return 0

    print("=" * 56)
    print("Downloading Whisper speech-recognition model")
    print(f"  Model     : faster-whisper-{size}  (~{APPROX_MB[size]} MB)")
    print(f"  Source    : {os.environ['HF_ENDPOINT']}  (one-time download)")
    print(f"  Save to   : {dest}")
    print("  Used for  : video audio -> text transcription (local, offline)")
    print("=" * 56)

    from huggingface_hub import snapshot_download

    try:
        snapshot_download(
            repo_id=f"Systran/faster-whisper-{size}",
            local_dir=str(dest),
        )
    except Exception as e:
        print(f"[WARN] Model download failed: {str(e)[:200]}")
        print("       You can retry later by running start.bat again;")
        print("       transcription will also auto-download on first use.")
        return 1

    print(f"[OK] Model saved to {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
