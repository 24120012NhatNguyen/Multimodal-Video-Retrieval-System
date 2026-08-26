"""Viec 3 -- lap day 100 dong dap an.

Bai thi cham theo thu hang va cho 100 dong. Nguoi dung thuong chi dien vai
dong, 95 dong con lai la diem bo khong.

Chay tren tap ket qua da co san (vai chuc ms, khong goi model):

  1. Giu nguyen cac frame source="manual" o DAU danh sach.
  2. Mo rong lan can thoi gian: cac keyframe cung video co pts_time trong
     t +/- 3s, xep ngay sau frame thu cong tuong ung.
  3. Lap phan con lai bang MMR thay vi top-k tho:
         next = argmax [ lambda*rel(i) - (1-lambda)*max_{j da chon} sim(i,j) ]
     de phu nhieu gia thuyet khac nhau thay vi 95 frame gan giong het nhau.
  4. Khu trung lap: moi video toi da m frame, hai frame cung video cach nhau
     it nhat 2 giay, loai cac frame trong danh sach ignore.

KHONG co khai niem "shot". Bo keyframe sinh ra bang lay mau theo do troi ngu
nghia, khong theo ranh gioi canh, nen moi quan he lan can deu tinh theo
pts_time. Cot cos_to_prev khong duoc dung o day: phan bo cua no da bi chinh
nguong lay mau cat cut nen khong phai tin hieu ranh gioi canh.
"""

import numpy as np

from retrieval.config import FusionConfig


class _Picker:
    """Giu rang buoc khu trung lap: m frame/video va khoang cach toi thieu."""

    def __init__(self, max_per_video, min_gap_sec):
        self.max_per_video = max_per_video
        self.min_gap_sec = min_gap_sec
        self.per_video = {}

    def accepts(self, video_id, pts_time):
        times = self.per_video.get(video_id)
        if times is None:
            return True
        if len(times) >= self.max_per_video:
            return False
        return all(abs(pts_time - t) >= self.min_gap_sec for t in times)

    def take(self, video_id, pts_time, force=False):
        if not force and not self.accepts(video_id, pts_time):
            return False
        self.per_video.setdefault(video_id, []).append(pts_time)
        return True


def _lookup(store):
    """(video_id, frame_idx) -> (gidx, pts_time)"""
    m = store.meta
    return {
        (v, int(f)): (int(g), float(p))
        for v, f, g, p in zip(m["video_id"], m["frame_idx"], m["gidx"], m["pts_time"])
    }


def autofill(store, manual, candidates=None, config=None, ignore=None, target=None):
    """-> danh sach dap an da xep thu tu, dung schema Viec 2.

    manual      [{"video_id", "frame_idx"}] nguoi chon -- luon nam dau
    candidates  [{"video_id", "frame_idx", "score"}] tap ket qua dang hien
    ignore      [{"video_id", "frame_idx"}] hoac [(video_id, frame_idx)]
    """
    cfg = config or FusionConfig.load()
    target = target or cfg.autofill_target
    look = _lookup(store)

    ign = set()
    for it in (ignore or []):
        if isinstance(it, dict):
            ign.add((it.get("video_id"), int(it.get("frame_idx"))))
        else:
            ign.add((it[0], int(it[1])))

    picker = _Picker(cfg.max_per_video, cfg.min_gap_sec)
    out = []
    seen = set()

    def emit(video_id, frame_idx, source, reason, force=False):
        key = (video_id, int(frame_idx))
        if key in seen or key in ign:
            return False
        hit = look.get(key)
        if hit is None:
            return False
        gidx, pts = hit
        if not picker.take(video_id, pts, force=force):
            return False
        seen.add(key)
        out.append({
            "video_id": video_id,
            "frame_idx": int(frame_idx),
            "source": source,
            "pts_time": round(pts, 3),
            "gidx": gidx,
            "reason": reason,
        })
        return True

    # --- 1 + 2: thu cong truoc, moi frame keo theo lan can cua no ----------
    for it in (manual or []):
        # Chap nhan ca dict lan pydantic model, de noi goi khong phai nho
        # model_dump() moi lan.
        if not isinstance(it, dict):
            it = it.model_dump() if hasattr(it, "model_dump") else dict(it)
        vid = it.get("video_id")
        fi = it.get("frame_idx")
        if vid is None or fi is None:
            continue
        # force=True: rang buoc khu trung lap KHONG duoc day frame thu cong xuong
        if not emit(vid, fi, "manual", "nguoi dung chon", force=True):
            continue
        base = look[(vid, int(fi))][1]
        df = store.frames_of(vid)
        if df.empty:
            continue
        w = cfg.neighbour_window_sec
        near = df[(df["pts_time"] >= base - w) & (df["pts_time"] <= base + w)]
        near = near.reindex(
            near["pts_time"].sub(base).abs().sort_values().index)
        for r in near.itertuples():
            if len(out) >= target:
                break
            emit(vid, int(r.frame_idx), "autofill",
                 f"lan can +/-{w}s cua {vid}#{int(fi)}")
        if len(out) >= target:
            break

    if len(out) >= target:
        return out[:target]

    # --- 3: MMR tren tap ung vien -----------------------------------------
    pool = []
    for c in (candidates or []):
        vid, fi = c.get("video_id"), c.get("frame_idx")
        if vid is None or fi is None:
            continue
        key = (vid, int(fi))
        if key in seen or key in ign or key not in look:
            continue
        pool.append((key, float(c.get("score") or 0.0)))
    if not pool:
        return out[:target]

    # Bo trung, giu diem cao nhat cho moi keyframe
    best = {}
    for key, s in pool:
        if key not in best or s > best[key]:
            best[key] = s
    keys = list(best)
    rel = np.array([best[k] for k in keys], dtype=np.float32)
    # Dua rel ve [0,1] de lambda can bang duoc voi sim (cosine cung ~[0,1]).
    if rel.size and float(rel.max() - rel.min()) > 1e-9:
        rel = (rel - rel.min()) / (rel.max() - rel.min())
    else:
        rel = np.ones_like(rel)

    gidxs = np.array([look[k][0] for k in keys], dtype=np.int64)
    V = store.X[gidxs]                     # da L2-normalize -> dot = cosine

    chosen_g = [look[(o["video_id"], o["frame_idx"])][0] for o in out]
    if chosen_g:
        # max sim toi cac frame DA chon (ke ca cac frame thu cong)
        max_sim = (V @ store.X[np.array(chosen_g, dtype=np.int64)].T).max(axis=1)
    else:
        max_sim = np.zeros(len(keys), dtype=np.float32)

    lam = cfg.mmr_lambda
    alive = np.ones(len(keys), dtype=bool)

    while len(out) < target and alive.any():
        mmr = lam * rel - (1.0 - lam) * max_sim
        mmr[~alive] = -np.inf
        i = int(np.argmax(mmr))
        if not np.isfinite(mmr[i]):
            break
        alive[i] = False
        vid, fi = keys[i]
        if not emit(vid, fi, "autofill",
                    f"MMR lambda={lam} (rel={rel[i]:.2f}, sim={max_sim[i]:.2f})"):
            continue
        # cap nhat do tuong dong toi tap da chon
        max_sim = np.maximum(max_sim, V @ V[i])

    return out[:target]
