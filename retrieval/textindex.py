"""Cac kenh van ban: meta, meta_fold, asr, ocr -- tat ca deu la BM25.

Dung chung mot bo may BM25 (fusion.MetaIndex), chi khac nguon van ban va bo
tach tu:

  meta       tokenize_vi    metadata BTC, query tieng Viet co dau
  meta_fold  tokenize_fold  cung metadata, cuu truong hop go query thieu dau
  asr        tokenize_vi    Whisper xuat tieng Viet co dau dung
  ocr        tokenize_fold  OCR chay sai model ngon ngu, moi ky tu co dau deu
                            hong (NO LU'C <- no luc) ma conf van 0.9+; tokenize
                            co dau se khong khop duoc tu nao
"""

import glob
import json
import os

from fusion import MetaIndex, tokenize_fold, tokenize_vi
from retrieval.config import ARTIFACT_ROOT, META_DIR


def asr_texts(root=ARTIFACT_ROOT):
    """{video_id: toan bo transcript}"""
    out = {}
    for f in sorted(glob.glob(os.path.join(root, "*", "asr", "*.json"))):
        vid = os.path.basename(f)[:-5]
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        segs = d.get("segments") or []
        txt = " ".join(s.get("text", "") for s in segs).strip()
        if txt:
            out[vid] = txt
    return out


def ocr_texts(root=ARTIFACT_ROOT):
    """{video_id: toan bo chu da trich}"""
    out = {}
    for f in sorted(glob.glob(os.path.join(root, "*", "ocr", "*.json"))):
        vid = os.path.basename(f)[:-5]
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        buf = []
        for fr in d.get("frames") or []:
            for it in fr.get("items") or []:
                t = it.get("text")
                if t:
                    buf.append(t)
        txt = " ".join(buf).strip()
        if txt:
            out[vid] = txt
    return out


class TextChannels:
    """Nap va giu bon index BM25. Kenh nao thieu du lieu thi vang mat khoi
    `self.index` va se khong gop vao RRF -- khong lam hong cac kenh con lai."""

    def __init__(self, root=ARTIFACT_ROOT, meta_dir=META_DIR, quiet=False):
        self.index = {}
        self.errors = {}

        if os.path.isdir(meta_dir):
            try:
                self.index["meta"] = MetaIndex(
                    meta_dir, tokenizer=tokenize_vi, name="meta", quiet=quiet)
                self.index["meta_fold"] = MetaIndex(
                    meta_dir, tokenizer=tokenize_fold, name="meta_fold", quiet=quiet)
            except Exception as e:
                self.errors["meta"] = f"{type(e).__name__}: {e}"
        else:
            self.errors["meta"] = f"khong thay thu muc metadata: {meta_dir}"

        try:
            a = asr_texts(root)
            if a:
                self.index["asr"] = MetaIndex.from_texts(
                    a, tokenizer=tokenize_vi, name="asr", quiet=quiet)
            else:
                self.errors["asr"] = "khong co transcript"
        except Exception as e:
            self.errors["asr"] = f"{type(e).__name__}: {e}"

        try:
            o = ocr_texts(root)
            if o:
                self.index["ocr"] = MetaIndex.from_texts(
                    o, tokenizer=tokenize_fold, name="ocr", quiet=quiet)
            else:
                self.errors["ocr"] = "khong co chu da trich"
        except Exception as e:
            self.errors["ocr"] = f"{type(e).__name__}: {e}"

    # ------------------------------------------------------------------
    def search(self, name, query, topn=100):
        idx = self.index.get(name)
        if idx is None or not query or not query.strip():
            return []
        return idx.search(query, topn=topn)

    def why(self, name, video_id, query):
        idx = self.index.get(name)
        if idx is None or not query:
            return []
        return idx.why(video_id, query)

    def status(self):
        return {
            "channels": {
                n: {"n_video": ix.N, "n_token": len(ix.df),
                    "avgdl": round(float(ix.avgdl), 1),
                    "tokenizer": ix.tokenizer.__name__}
                for n, ix in self.index.items()
            },
            "errors": self.errors,
        }
