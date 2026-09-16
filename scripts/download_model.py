"""Pre-download local models into models/ (one-time; called by start.bat).

Two kinds of models, both fetched from hf-mirror.com (CN-friendly mirror) and
skipped automatically when already present:
  - Whisper speech recognition (Systran/faster-whisper-<size>): audio -> text
  - BGE sentence embedding (Xenova/bge-small-zh-v1.5): semantic search vectors

Usage: python download_model.py [tiny|base|small|medium|bge]   (default: small)
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


def download_whisper(size: str) -> int:
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


# BGE 句向量模型（问 AI 的语义检索用，app/embedder.py 消费）。
# 选 Xenova 的 ONNX 导出：自带 onnx/model_quantized.onnx（int8），
# 运行时只需 onnxruntime + tokenizers（faster-whisper 已传递引入），无需 torch。
BGE_REPO = "Xenova/bge-small-zh-v1.5"
BGE_DIR = PROJECT / "models" / "bge-small-zh-v1.5"
BGE_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "onnx/model_quantized.onnx",
    "onnx/model.onnx",
)


def download_bge() -> int:
    marker = BGE_DIR / "onnx" / "model_quantized.onnx"
    if marker.is_file() and marker.stat().st_size > 5_000_000:
        print(f"[OK] Model already downloaded: {BGE_DIR}")
        print("     (Skip. Delete this folder to re-download.)")
        return 0

    print("=" * 56)
    print("Downloading BGE sentence-embedding model (semantic search)")
    print("  Model     : bge-small-zh-v1.5 ONNX  (~120 MB, fp32 + int8)")
    print(f"  Source    : {os.environ['HF_ENDPOINT']}  (one-time download)")
    print(f"  Save to   : {BGE_DIR}")
    print("  Used for  : semantic vector retrieval for AI Q&A (local, offline)")
    print("=" * 56)

    from huggingface_hub import snapshot_download

    try:
        snapshot_download(
            repo_id=BGE_REPO,
            local_dir=str(BGE_DIR),
            allow_patterns=list(BGE_FILES),
        )
    except Exception as e:
        print(f"[WARN] Model download failed: {str(e)[:200]}")
        print("       AI Q&A will fall back to keyword-only retrieval;")
        print("       retry later with: python scripts/download_model.py bge")
        return 1

    print(f"[OK] Model saved to {BGE_DIR}")
    return 0


def main() -> int:
    target = sys.argv[1] if len(sys.argv) > 1 else "small"
    if target == "bge":
        return download_bge()
    if target in SIZES:
        return download_whisper(target)
    print(f"[ERROR] Unknown model '{target}'. Choose one of: {', '.join(SIZES)}, bge")
    return 1


if __name__ == "__main__":
    sys.exit(main())
