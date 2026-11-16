"""为五邑大学知识库建稠密向量索引（bge-m3 + Faiss）。

用法：
    python src/build_wyu_dense.py
    WYU_DEVICE=cpu python src/build_wyu_dense.py
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]  # 05-项目源码
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from models import Chunk

INDEX_DIR = ROOT / "data" / "index" / "wyu"
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass
MODEL = os.environ.get("WYU_MODEL_PATH", "BAAI/bge-m3")
MAX_LEN = 512
BATCH = 16


def encode(tok, model, texts: list[str], device: str) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        inputs = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model(**inputs)
        emb = torch.nn.functional.normalize(
            out.last_hidden_state[:, 0].float(), p=2, dim=1
        )
        vecs.append(emb.cpu().numpy())
    return np.vstack(vecs)


def load_model():
    """先读本地缓存，避免联网检查/下载导致超时。"""
    try:
        tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        try:
            model = AutoModel.from_pretrained(
                MODEL, local_files_only=True, use_safetensors=True
            )
        except Exception:
            model = AutoModel.from_pretrained(
                MODEL, local_files_only=True, use_safetensors=False
            )
        return tok, model
    except Exception:
        tok = AutoTokenizer.from_pretrained(MODEL)
        try:
            model = AutoModel.from_pretrained(
                MODEL, use_safetensors=True
            )
        except Exception:
            model = AutoModel.from_pretrained(
                MODEL, use_safetensors=False
            )
        return tok, model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    chunks_path = INDEX_DIR / "chunks.pkl"
    if not chunks_path.exists():
        raise SystemExit(f"缺少切片文件：{chunks_path}，先运行 build_wyu_index.py")
    with open(chunks_path, "rb") as fh:
        chunks: list[Chunk] = pickle.load(fh)
    print(f"chunks: {len(chunks)}", flush=True)

    if args.device == "cpu":
        device = "cpu"
    elif args.device == "cuda":
        device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    tok, model = load_model()
    model = model.to(device).eval()
    if device == "cuda":
        model = model.half()
    print("model loaded", flush=True)

    vectors = encode(tok, model, [chunk.text for chunk in chunks], device)
    print("vectors:", vectors.shape, flush=True)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    # 同样绕开 faiss C++ 文件 IO 的中文路径问题。
    serialized = faiss.serialize_index(index)
    (INDEX_DIR / "faiss.index").write_bytes(serialized.tobytes())
    print("saved:", INDEX_DIR / "faiss.index", flush=True)


if __name__ == "__main__":
    main()
