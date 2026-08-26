"""Nap tang du lieu artifacts: features SigLIP + bang keyframe.

Rang buoc tuyet doi (xem brief): .npy va .csv khop nhau THEO THU TU DONG,
khong co khoa nao khac. Store nay nap ca hai cung luc va gan cho moi dong mot
chi so toan cuc `gidx`; moi thao tac loc/sap xep ve sau deu di qua `gidx` nen
khong the lam lech hai nguon.
"""

import csv
import glob
import json
import os

import numpy as np
import pandas as pd

from retrieval.config import ARTIFACT_ROOT, VIDEO_DIR

EMBED_DIM = 1152


class ArtifactStore:
    """X: (N, 1152) float32 da L2-normalize. meta: DataFrame cung so dong voi X."""

    def __init__(self, root=ARTIFACT_ROOT, video_dir=VIDEO_DIR):
        self.root = root
        self.video_dir = video_dir
        self.build = self._load_build()
        self.packs = {}          # pack -> so video
        self.skipped = []        # (video_id, ly do)

        frames = []
        chunks = []
        gidx = 0

        for pack_dir in sorted(glob.glob(os.path.join(root, "*"))):
            if not os.path.isdir(pack_dir):
                continue
            pack = os.path.basename(pack_dir)
            feat_dir = os.path.join(pack_dir, "features")
            kf_dir = os.path.join(pack_dir, "keyframes")
            if not (os.path.isdir(feat_dir) and os.path.isdir(kf_dir)):
                continue

            n_video = 0
            for npy in sorted(glob.glob(os.path.join(feat_dir, "*.npy"))):
                vid = os.path.basename(npy)[:-4]
                csv_path = os.path.join(kf_dir, f"{vid}.csv")
                if not os.path.exists(csv_path):
                    self.skipped.append((vid, "thieu keyframes/*.csv"))
                    continue

                X = np.load(npy)
                rows = self._read_csv(csv_path)

                # Lech dong = du lieu hong, khong the doan bu. Bo qua ca video
                # con hon la ghep sai vector voi frame_idx.
                if X.shape[0] != len(rows):
                    self.skipped.append(
                        (vid, f"lech dong: npy={X.shape[0]} csv={len(rows)}")
                    )
                    continue
                if X.ndim != 2 or X.shape[1] != EMBED_DIM:
                    self.skipped.append((vid, f"shape la {X.shape}, can (N,{EMBED_DIM})"))
                    continue

                # float16 -> float32 TRUOC khi nhan ma tran (yeu cau cua brief).
                chunks.append(X.astype(np.float32))
                for r in rows:
                    r["video_id"] = vid
                    r["pack"] = pack
                    r["gidx"] = gidx
                    gidx += 1
                frames.extend(rows)
                n_video += 1

            if n_video:
                self.packs[pack] = n_video

        if chunks:
            self.X = np.vstack(chunks)
        else:
            self.X = np.zeros((0, EMBED_DIM), dtype=np.float32)

        self.meta = pd.DataFrame(frames, columns=[
            "gidx", "video_id", "pack", "n", "frame_idx", "pts_time",
            "lap_var", "cos_to_prev",
        ]) if frames else pd.DataFrame(columns=[
            "gidx", "video_id", "pack", "n", "frame_idx", "pts_time",
            "lap_var", "cos_to_prev",
        ])

        self._index_videos()

    # ------------------------------------------------------------------
    @staticmethod
    def _read_csv(path):
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                def num(key, cast=float):
                    v = r.get(key, "")
                    if v is None or v == "":
                        return None
                    try:
                        return cast(v)
                    except ValueError:
                        return None
                out.append({
                    "n": num("n", int),
                    "frame_idx": num("frame_idx", int),
                    "pts_time": num("pts_time", float),
                    "lap_var": num("lap_var", float),
                    "cos_to_prev": num("cos_to_prev", float),
                })
        return out

    def _load_build(self):
        """BUILD.json ghi build_id va nguong lay mau cua tung pack.

        Khong tim thay thi tra ve marker co ly do, khong tra dict rong -- de
        /diagnostics phan biet "chua doc" voi "khong co file".
        """
        for pat in (os.path.join(self.root, "BUILD.json"),
                    os.path.join(self.root, "*", "BUILD.json")):
            for p in sorted(glob.glob(pat)):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        b = json.load(f)
                    b["_path"] = p
                    return b
                except Exception:
                    continue
        return {"_missing": (
            f"khong tim thay BUILD.json trong {self.root}/ hoac {self.root}/*/ "
            f"-- mat build_id va nguong lay mau cua tung pack")}

    def _index_videos(self):
        """Lat cat [start, end) cua tung video trong X, va fps suy tu CSV."""
        self.video_slice = {}
        self.fps = {}
        if self.meta.empty:
            self.video_ids = []
            return

        vids = self.meta["video_id"].to_numpy()
        # meta duoc dung theo thu tu video nen moi video la mot doan lien tuc
        start = 0
        for i in range(1, len(vids) + 1):
            if i == len(vids) or vids[i] != vids[start]:
                self.video_slice[vids[start]] = (start, i)
                start = i

        self.video_ids = list(self.video_slice)

        # fps suy tu frame_idx / pts_time. Kiem chung tren 400 video: std = 0,
        # phan bo {25.0, 30.0}. Dung median cho chac, bo cac dong pts_time nho
        # (chia cho so gan 0 khuech dai sai so lam tron).
        fi = self.meta["frame_idx"].to_numpy(dtype=float)
        pt = self.meta["pts_time"].to_numpy(dtype=float)
        for vid, (a, b) in self.video_slice.items():
            f, p = fi[a:b], pt[a:b]
            m = p > 0.5
            self.fps[vid] = float(np.median(f[m] / p[m])) if m.sum() else 25.0

    # ------------------------------------------------------------------
    def assert_encoder_matches(self, encoder):
        """Lo i im lang #1: doi nguon vector ma quen doi text encoder.

        Cosine giua vector anh SigLIP va vector text cua model khac VAN ra so
        dep (ca hai deu da chuan hoa) nhung khong co y nghia gi. Kiem tra bat
        buoc: X 1152 chieu thi encoder phai la SigLIP cung so chieu.
        """
        if self.X.shape[1] != EMBED_DIM:
            raise RuntimeError(
                f"features co {self.X.shape[1]} chieu, khong phai {EMBED_DIM} cua SigLIP"
            )
        if hasattr(encoder, "_ensure"):
            encoder._ensure()
        dim = getattr(encoder, "dim", None)
        if dim != EMBED_DIM:
            raise RuntimeError(
                f"text encoder co {dim} chieu, khong phai {EMBED_DIM} cua SigLIP"
            )

    def frames_of(self, video_id):
        """DataFrame cac keyframe cua mot video, giu nguyen thu tu dong."""
        sl = self.video_slice.get(video_id)
        if sl is None:
            return self.meta.iloc[0:0]
        return self.meta.iloc[sl[0]:sl[1]]

    def rows_for_videos(self, video_ids):
        """Mask boolean tren X/meta cho mot tap video."""
        mask = np.zeros(len(self.meta), dtype=bool)
        for v in video_ids:
            sl = self.video_slice.get(v)
            if sl:
                mask[sl[0]:sl[1]] = True
        return mask

    def video_path(self, video_id):
        p = os.path.join(self.video_dir, f"{video_id}.mp4")
        return p if os.path.exists(p) else None

    @property
    def build_id(self):
        return self.build.get("build_id")

    def __repr__(self):
        return (f"<ArtifactStore {len(self.video_ids)} video, "
                f"{len(self.meta)} keyframe, build={self.build_id}>")
