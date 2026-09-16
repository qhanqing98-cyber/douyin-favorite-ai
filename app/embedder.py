"""本地句向量：bge-small-zh-v1.5（ONNX int8 量化）+ tokenizers。

模型文件在 models/bge-small-zh-v1.5/（用 `python scripts/download_model.py bge`
下载，来自 Xenova/bge-small-zh-v1.5 的 ONNX 导出）。运行时只用到
onnxruntime / tokenizers / numpy —— 都由 faster-whisper 传递引入，零新增依赖。

用法遵循 BGE 官方约定：
- 文档侧直接编码，取 [CLS] 向量后 L2 归一化；
- 查询侧加指令前缀「为这个句子生成表示以用于检索相关文章：」；
- 归一化后内积即余弦相似度。
"""
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT / "models" / "bge-small-zh-v1.5"
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
MODEL_TAG = "bge-small-zh-v1.5/int8"  # 存 meta.embed_model；换模型即触发全量重建
DIM = 512
MAX_LEN = 512
BATCH = 32  # 一次喂太多会把 last_hidden_state 撑到几百 MB

_session = None
_tokenizer = None
_need_token_type = False


def _load() -> None:
    """懒加载模型（首次编码时才读文件，进程内只加载一次）。"""
    global _session, _tokenizer, _need_token_type
    if _session is not None:
        return
    onnx_path = MODEL_DIR / "onnx" / "model_quantized.onnx"
    tok_path = MODEL_DIR / "tokenizer.json"
    if not onnx_path.is_file() or not tok_path.is_file():
        raise RuntimeError(
            "缺少 bge 向量模型：请先运行  python scripts/download_model.py bge  "
            f"（预期位置：{MODEL_DIR}）"
        )
    import onnxruntime as ort
    from tokenizers import Tokenizer

    _tokenizer = Tokenizer.from_file(str(tok_path))
    _tokenizer.enable_truncation(max_length=MAX_LEN)
    _tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")  # 批内对齐，PAD 不影响 [CLS]
    _session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    _need_token_type = "token_type_ids" in {i.name for i in _session.get_inputs()}


def _encode(texts: list[str], prefix: str = "") -> np.ndarray:
    _load()
    texts = [(prefix + (t or "").strip()) or "空" for t in texts]
    out: list[np.ndarray] = []
    for i in range(0, len(texts), BATCH):
        enc = _tokenizer.encode_batch(texts[i : i + BATCH])
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        att = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": att}
        if _need_token_type:
            feed["token_type_ids"] = np.zeros_like(ids)
        hidden = _session.run(None, feed)[0]  # (n, seq, hidden)
        cls = hidden[:, 0, :].astype(np.float32)  # BGE 用 [CLS] 向量
        norm = np.linalg.norm(cls, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        out.append(cls / norm)
    return np.vstack(out)


def encode_docs(texts: list[str]) -> np.ndarray:
    """文档（转写块/概要/标题）向量，形状 (n, 512)，已 L2 归一化。"""
    if not texts:
        return np.zeros((0, DIM), dtype=np.float32)
    return _encode(texts)


def encode_query(text: str) -> np.ndarray:
    """查询向量（加官方检索指令前缀），形状 (1, 512)。"""
    return _encode([text], prefix=QUERY_PREFIX)
