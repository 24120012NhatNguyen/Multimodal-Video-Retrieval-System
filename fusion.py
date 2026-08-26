#!/usr/bin/env python3
"""
Tang hop nhat cap video: nhieu kenh bang chung -> mot xep hang video duy nhat.

Kien truc: moi kenh tra ve mot DANH SACH VIDEO DA XEP HANG. RRF gop chung lai
bang THU HANG, khong bang diem, nen khong can chuan hoa giua cac kenh khac
ban chat (cosine 0.31 va BM25 12.4 khong cung thang do).

Them mot kenh moi (ASR, OCR, clip-features cu, mot menh de phu) = them mot
danh sach vao rrf(). Khong phai sua gi khac.

    from fusion import MetaIndex, siglip_video_rank, rrf, explain
    mi = MetaIndex(f"{BTC}/metadata")
    lists = {
        "meta":   mi.search("múa lân Chợ Lớn", topn=100),
        "siglip": siglip_video_rank(X, meta, emb, "a lion dance in a street"),
    }
    for vid, sc in rrf(lists)[:10]: print(vid, round(sc, 4))
"""

import glob, json, os, re, unicodedata
from collections import defaultdict

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
def tokenize_vi(text):
    """Tach tu tieng Viet muc don gian: bo dau cau, ha chu thuong, giu dau thanh.

    Khong dung pyvi/underthesea de khong them phu thuoc. Voi BM25 tren tieu de
    + mo ta thi tach theo khoang trang da du; ten rieng va so van khop chinh xac.
    """
    text = unicodedata.normalize("NFC", str(text).lower())
    return [t for t in re.split(r"[^0-9a-zA-ZÀ-ỹ]+", text) if len(t) > 1]


# Dau nhay o moi bien the ma OCR co the xuat ra. U+00B4 (´) khong thuoc category
# Mn nen NFD KHONG xoa duoc -> phai liet ke tay.
_APOSTROPHES = "'’‘ʼ`´"


def fold(text):
    """Bo toan bo dau thanh + dau nhay, ha chu thuong.

    Dung cho OCR: bo nhan dang dang chay sai model ngon ngu, moi ky tu co dau
    deu hong ma conf van 0.9+ (NO LU'C <- NO LUC, Nep cai <- Nep cai,
    Dau xanh khong vo <- Dau xanh khong vo). Dau moc (u', o') duoc bieu dien
    bang dau nhay, nen phai xoa dau nhay TRUOC khi tach tu -- neu tach truoc,
    "LU'C" thanh ["lu", "c"] va khong bao gio khop duoc "luc".
    """
    t = str(text).lower()
    for ch in _APOSTROPHES:
        t = t.replace(ch, "")
    t = unicodedata.normalize("NFD", t)
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return t.replace("đ", "d").replace("ð", "d")


def tokenize_fold(text):
    """Tach tu tren van ban da bo dau. Ca hai phia (doc va query) phai dung
    cung ham nay, neu khong se khong khop gi ca."""
    return [w for w in re.split(r"[^0-9a-z]+", fold(text)) if len(w) > 1]


class MetaIndex:
    """BM25 tren metadata cua BTC (tieu de + mo ta + tu khoa).

    `tokenizer` cho phep dung chung bo may BM25 nay cho cac kenh van ban khac
    (ASR giu tokenize_vi vi Whisper xuat dau dung; OCR dung tokenize_fold).
    """

    FIELDS = ("title", "description", "keywords", "author", "channel_name")

    def __init__(self, meta_dir, k1=1.5, b=0.75, weight_title=3,
                 tokenizer=None, name="meta", quiet=False):
        self.k1, self.b = k1, b
        self.tokenizer = tokenizer or tokenize_vi
        self.name = name
        self.docs, self.raw = {}, {}

        for f in sorted(glob.glob(f"{meta_dir}/*.json")):
            vid = os.path.basename(f)[:-5]
            try:
                m = json.load(open(f))
            except Exception:
                continue
            parts = []
            for k in self.FIELDS:
                v = m.get(k)
                if not v:
                    continue
                s = " ".join(v) if isinstance(v, list) else str(v)
                # tieu de dang tin hon mo ta -> lap lai de tang trong so
                parts += [s] * (weight_title if k == "title" else 1)
            self.raw[vid] = m
            self.docs[vid] = self.tokenizer(" ".join(parts))

        self._build(quiet=quiet)

    @classmethod
    def from_texts(cls, texts, k1=1.5, b=0.75, tokenizer=None, name="text",
                   quiet=False):
        """Dung index BM25 tu {video_id: "toan bo van ban cua video do"}.

        Dung cho ASR / OCR: cung thuat toan BM25, cung ham search/idf/why,
        chi khac nguon van ban va bo tach tu.
        """
        self = cls.__new__(cls)
        self.k1, self.b = k1, b
        self.tokenizer = tokenizer or tokenize_vi
        self.name = name
        self.raw = {}
        self.docs = {vid: self.tokenizer(t) for vid, t in texts.items()}
        self._build(quiet=quiet)
        return self

    def _build(self, quiet=False):
        self.ids = list(self.docs)
        self.N = len(self.ids)
        self.avgdl = np.mean([len(d) for d in self.docs.values()]) if self.N else 1.0

        self.tf = {v: defaultdict(int) for v in self.ids}
        self.df = defaultdict(int)
        for v, toks in self.docs.items():
            for t in toks:
                self.tf[v][t] += 1
            for t in set(toks):
                self.df[t] += 1
        if not quiet:
            print(f"MetaIndex[{self.name}]: {self.N} video, {len(self.df)} tu, "
                  f"do dai trung binh {self.avgdl:.0f}")

    def idf(self, t):
        n = self.df.get(t, 0)
        return np.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def search(self, query, topn=100):
        """-> [(video_id, diem)] xep giam dan. Query bang TIENG VIET."""
        q = self.tokenizer(query)
        if not q:
            return []
        out = []
        for v in self.ids:
            tf, dl, s = self.tf[v], len(self.docs[v]), 0.0
            for t in q:
                f = tf.get(t, 0)
                if not f:
                    continue
                s += self.idf(t) * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            if s > 0:
                out.append((v, float(s)))
        return sorted(out, key=lambda x: -x[1])[:topn]

    def why(self, vid, query):
        """Tu nao trong query khop video nay -- de hien tren UI."""
        q = set(self.tokenizer(query))
        return sorted(t for t in q if self.tf.get(vid, {}).get(t, 0) > 0)


# ----------------------------------------------------------------------
def siglip_video_rank(X, meta, emb, queries, topn=100, per_video="max"):
    """Quy diem frame ve diem VIDEO.

    queries: str hoac list[str] (nhieu menh de -> cong bang chung, kieu TT-4).
    Tra ve [(video_id, diem)] giam dan.
    """
    if isinstance(queries, str):
        queries = [queries]
    Q = emb.encode_texts(queries)                 # (nq, D)
    S = X @ Q.T                                   # (N, nq)

    df = pd.DataFrame({"video_id": meta.video_id.values})
    for j in range(S.shape[1]):
        df[f"s{j}"] = S[:, j]
    agg = df.groupby("video_id").agg(
        {f"s{j}": ("max" if per_video == "max" else "mean")
         for j in range(S.shape[1])})
    # Cong diem tot nhat cua TUNG menh de: video co bang chung cho nhieu menh de
    # se noi len, ke ca khi tung menh de rieng le deu tam thuong.
    agg["total"] = agg.sum(axis=1)
    r = agg.total.sort_values(ascending=False)[:topn]
    return list(zip(r.index.tolist(), r.values.tolist()))


# ----------------------------------------------------------------------
def rrf(ranked_lists, k=60, weights=None):
    """Reciprocal Rank Fusion.

    ranked_lists: {ten_kenh: [(video_id, diem), ...]} da xep hang giam dan.
    Chi dung THU HANG -> khong can chuan hoa diem giua cac kenh.
    """
    weights = weights or {}
    score = defaultdict(float)
    for name, lst in ranked_lists.items():
        w = weights.get(name, 1.0)
        for rank, (vid, _) in enumerate(lst, 1):
            score[vid] += w / (k + rank)
    return sorted(score.items(), key=lambda x: -x[1])


def explain(ranked_lists, vid):
    """Video nay dung hang may o tung kenh -- de debug va hien tren UI."""
    out = {}
    for name, lst in ranked_lists.items():
        ids = [v for v, _ in lst]
        out[name] = ids.index(vid) + 1 if vid in ids else None
    return out


def frames_in_videos(X, meta, emb, query, video_ids, topk=100):
    """Sau khi da chon duoc video, moi xuong tang frame."""
    m = meta.video_id.isin(set(video_ids)).to_numpy()
    if not m.any():
        return pd.DataFrame()
    q = emb.encode_texts([query])[0]
    s = X[m] @ q
    sub = meta[m].copy()
    sub["score"] = s
    return sub.sort_values("score", ascending=False).head(topk)
