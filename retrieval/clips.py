"""Clip ngan quanh mot keyframe: [pts - window, pts + window].

Vi sao can: mot anh tinh khong phan biet duoc "dang buoc vao" voi "dang buoc
ra", "xe dang chay toi" voi "xe dang lui". Nguoi ngoi truoc man hinh can thay
4 giay quanh frame de chot dap an -- do la thao tac nhanh nhat de loai ket qua
sai, va truoc day khong lam duoc.

Chay o LOCAL, cung ly do voi tang anh: data/videos khong co tren Kaggle.

Cat co RE-ENCODE chu khong `-c copy`: cat sao chep chi cat duoc tai keyframe cua
ma hoa (cach nhau 2-10s), nen clip se lech khoi moc mong muon. 4 giay re-encode
o preset veryfast mat ~0,3s -- doi lay moc dung.
"""

import os
import subprocess
import threading

CLIP_CACHE = os.environ.get("CLIP_CACHE", "data/clip_cache")
DEFAULT_WINDOW = 2.0
MAX_WINDOW = 30.0

_locks = {}
_locks_guard = threading.Lock()


def _lock_for(path):
    with _locks_guard:
        lk = _locks.get(path)
        if lk is None:
            lk = _locks[path] = threading.Lock()
        return lk


def cache_path(video_id, frame_idx, window, root=CLIP_CACHE):
    return os.path.join(root, video_id,
                        f"{int(frame_idx):06d}_{float(window):g}.mp4")


def cut(video_path, start, duration, out_path, crf=26):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = f"{out_path}.{os.getpid()}.tmp.mp4"
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error",
        # -ss truoc -i de seek nhanh; -t sau -i de do dai tinh tu diem seek.
        "-ss", f"{max(0.0, float(start)):.3f}",
        "-i", video_path,
        "-t", f"{float(duration):.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "96k",
        "-movflags", "+faststart",
        "-y", tmp,
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or not os.path.exists(tmp):
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(
            f"ffmpeg loi khi cat {video_path} @ {start}s +{duration}s: "
            f"{r.stderr.decode('utf-8', 'replace')[:300]}")
    os.replace(tmp, out_path)
    return out_path


class ClipCache:
    def __init__(self, store, root=CLIP_CACHE):
        self.store = store
        self.root = root

    def pts_of(self, video_id, frame_idx):
        """pts_time cua mot frame. Khong phai keyframe -> quy doi bang fps."""
        df = self.store.frames_of(video_id)
        if not df.empty:
            hit = df[df["frame_idx"] == int(frame_idx)]
            if not hit.empty:
                return float(hit.iloc[0]["pts_time"])
        fps = self.store.fps.get(video_id) or 25.0
        return float(frame_idx) / fps

    def get(self, video_id, frame_idx, window=DEFAULT_WINDOW):
        """-> (duong dan mp4, thong tin) hoac (None, ly do)."""
        try:
            window = float(window)
        except (TypeError, ValueError):
            window = DEFAULT_WINDOW
        window = max(0.5, min(MAX_WINDOW, window))

        video = self.store.video_path(video_id)
        if not video:
            return None, f"khong co file video cho {video_id}"

        pts = self.pts_of(video_id, frame_idx)
        start = max(0.0, pts - window)
        # Cua so bi cat o dau video thi giu tron do dai bang cach keo dai ve sau.
        duration = (pts + window) - start

        out = cache_path(video_id, frame_idx, window, self.root)
        info = {"video_id": video_id, "frame_idx": int(frame_idx),
                "pts_time": round(pts, 3), "start": round(start, 3),
                "end": round(start + duration, 3),
                "duration": round(duration, 3), "window_sec": window,
                # Vi tri cua frame trong clip -> UI dat duoc vach danh dau.
                "frame_offset_sec": round(pts - start, 3)}

        if os.path.exists(out):
            info["cached"] = True
            return out, info

        with _lock_for(out):
            if os.path.exists(out):
                info["cached"] = True
                return out, info
            try:
                cut(video, start, duration, out)
            except Exception as e:
                return None, f"{type(e).__name__}: {e}"
        info["cached"] = False
        return out, info

    def stats(self):
        n = 0
        if os.path.isdir(self.root):
            for _, _, files in os.walk(self.root):
                n += sum(1 for f in files if f.endswith(".mp4"))
        return {"cache_dir": self.root, "n_cached": n}
