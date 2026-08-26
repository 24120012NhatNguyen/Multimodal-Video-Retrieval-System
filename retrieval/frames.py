"""Anh keyframe MOI, trich on-demand tu data/videos/*.mp4 va cache lai tren dia.

Bo keyframe moi duoc lay mau theo do troi ngu nghia, khong trung voi bo keyframe
cu cua BTC: chi 3,343 / 101,665 frame moi (3.3%) co san JPG trong
frontend/public/static/images/Keyframes/. Trich thang tu mp4 la cach duy nhat
hien dung anh dung voi frame_idx se nop bai.

ffmpeg seek do duoc ~88ms/anh, nen lan dau cham, cac lan sau doc tu cache.
"""

import os
import subprocess
import threading

from retrieval.config import KEYFRAME_CACHE

_locks = {}
_locks_guard = threading.Lock()


def cache_path(video_id, frame_idx, root=KEYFRAME_CACHE):
    return os.path.join(root, video_id, f"{int(frame_idx):06d}.jpg")


def _lock_for(path):
    with _locks_guard:
        lk = _locks.get(path)
        if lk is None:
            lk = _locks[path] = threading.Lock()
        return lk


def extract(video_path, pts_time, out_path, quality=3):
    """Trich mot frame tai pts_time. -ss dat TRUOC -i (seek dau vao) nen nhanh;
    ffmpeg >= 2.1 seek dau vao van chinh xac den frame."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Duoi file phai la .jpg: ffmpeg suy dinh dang dau ra tu duoi file, nen
    # ".jpg.part" lam no bao "Unable to find a suitable output format".
    tmp = f"{out_path}.{os.getpid()}.tmp.jpg"
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-ss", f"{float(pts_time):.3f}",
        "-i", video_path,
        "-frames:v", "1", "-q:v", str(quality),
        "-y", tmp,
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or not os.path.exists(tmp):
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(
            f"ffmpeg loi khi trich {video_path} @ {pts_time}s: "
            f"{r.stderr.decode('utf-8', 'replace')[:300]}"
        )
    os.replace(tmp, out_path)
    return out_path


class KeyframeImages:
    def __init__(self, store, root=KEYFRAME_CACHE):
        self.store = store
        self.root = root

    def get(self, video_id, frame_idx):
        """-> duong dan JPG tren dia, trich neu chua co. None neu khong tra duoc."""
        out = cache_path(video_id, frame_idx, self.root)
        if os.path.exists(out):
            return out

        df = self.store.frames_of(video_id)
        if df.empty:
            return None
        hit = df[df["frame_idx"] == int(frame_idx)]
        if hit.empty:
            # frame_idx khong phai keyframe cua video nay -> quy ra thoi gian
            # bang fps de van trich duoc (dung cho FrameRangeViewer).
            fps = self.store.fps.get(video_id) or 25.0
            pts = float(frame_idx) / fps
        else:
            pts = float(hit.iloc[0]["pts_time"])

        video = self.store.video_path(video_id)
        if not video:
            return None

        with _lock_for(out):
            if os.path.exists(out):
                return out
            try:
                extract(video, pts, out)
            except Exception:
                return None
        return out

    # ------------------------------------------------------------------
    def warm_video(self, video_id, quality=3):
        """Trich toan bo keyframe cua mot video trong MOT lan doc file.

        Nhanh hon nhieu so voi seek tung frame khi muon lam am cache hang loat.
        """
        df = self.store.frames_of(video_id)
        video = self.store.video_path(video_id)
        if df.empty or not video:
            return 0

        todo = [(int(r.frame_idx), float(r.pts_time)) for r in df.itertuples()
                if not os.path.exists(cache_path(video_id, r.frame_idx, self.root))]
        if not todo:
            return 0

        out_dir = os.path.join(self.root, video_id)
        os.makedirs(out_dir, exist_ok=True)
        sel = "+".join(f"eq(n\\,{fi})" for fi, _ in todo)
        tmp = os.path.join(out_dir, "_warm_%06d.jpg")
        cmd = [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-i", video,
            "-vf", f"select='{sel}'", "-vsync", "0", "-q:v", str(quality),
            "-y", tmp,
        ]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0:
            return 0

        # ffmpeg danh so anh xuat theo thu tu 1..n -> anh xa nguoc ve frame_idx
        n = 0
        for i, (fi, _) in enumerate(sorted(todo), start=1):
            src = os.path.join(out_dir, f"_warm_{i:06d}.jpg")
            if os.path.exists(src):
                os.replace(src, cache_path(video_id, fi, self.root))
                n += 1
        for leftover in os.listdir(out_dir):
            if leftover.startswith("_warm_"):
                os.remove(os.path.join(out_dir, leftover))
        return n

    def stats(self):
        n = 0
        if os.path.isdir(self.root):
            for _, _, files in os.walk(self.root):
                n += sum(1 for f in files if f.endswith(".jpg"))
        return {"cache_dir": self.root, "n_cached": n,
                "n_total": len(self.store.meta)}
