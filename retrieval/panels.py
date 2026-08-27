"""Ba duong tim kiem phu cua UI: /panel, /framerange, /imgsearch.

Ca ba deu tra ve CUNG mot dang ma luoi anh cua UI doc duoc:

    {"videos": [{"video_id", "video_info": {"lst_keyframe_paths", "lst_idxs",
                 "lst_keyframe_idxs", "lst_pts_times", "lst_scores"}}]}

`lst_idxs` la khoa "video_id#frame_idx" -- dung chung voi danh sach ignore va
danh sach dap an ben socket_app. He cu dung chi so toan cuc trong dict/id2img.json,
thu muc do da bi xoa nen khoa do khong con y nghia.

Diem chung quan trong: KET QUA LUON LA KEYFRAME CUA TA, khong phai keyframe cua
BTC. Object cua BTC neo vao keyframe cua HO; muon hien anh va nop bai duoc thi
phai bac cau nguoc ve keyframe cua ta qua pts_time, trong dung sai
`object_tolerance_sec`. Ngoai dung sai thi BO, khong lay "gan nhat bat ke xa
bao nhieu".
"""

import json
import os

import numpy as np

from retrieval.config import META_DIR
from retrieval.engine import entry_key, keyframe_url, parse_entry_key
from retrieval.objects import parse_counts


# ---------------------------------------------------------------------------
def pack_videos(rows, order=None, extra_by_video=None):
    """[{video_id, frame_idx, pts_time, score}] -> dang luoi anh cua UI.

    order  {video_id: thu tu} de giu dung xep hang cua tang tren; thieu thi giu
           thu tu xuat hien.
    """
    grouped = {}
    seen = set()
    seq = {}
    for r in rows:
        vid = r["video_id"]
        key = entry_key(vid, r["frame_idx"])
        if key in seen:
            continue
        seen.add(key)
        seq.setdefault(vid, len(seq))
        grouped.setdefault(vid, []).append(r)

    if order is None:
        order = seq

    out = []
    for vid in sorted(grouped, key=lambda v: order.get(v, 1 << 30)):
        rs = grouped[vid]
        item = {
            "video_id": vid,
            "video_info": {
                "lst_keyframe_paths": [keyframe_url(vid, r["frame_idx"]) for r in rs],
                "lst_idxs": [entry_key(vid, r["frame_idx"]) for r in rs],
                "lst_keyframe_idxs": [int(r["frame_idx"]) for r in rs],
                "lst_pts_times": [round(float(r["pts_time"]), 3) for r in rs],
                "lst_scores": [(None if r.get("score") is None
                                else round(float(r["score"]), 4)) for r in rs],
            },
        }
        if extra_by_video and vid in extra_by_video:
            item.update(extra_by_video[vid])
        out.append(item)
    return out


def ignore_set(ignore_idxs):
    """Danh sach ignore -> tap khoa 'video#frame'. Nhan ca khoa chuoi lan dict."""
    out = set()
    for it in ignore_idxs or ():
        if isinstance(it, str):
            p = parse_entry_key(it)
            if p:
                out.add(entry_key(*p))
        elif isinstance(it, dict):
            v, f = it.get("video_id"), it.get("frame_idx")
            if v is not None and f is not None:
                out.add(entry_key(v, f))
    return out


def videos_from_keys(keys):
    """['L21_V001#120', ...] -> ['L21_V001', ...] giu thu tu, khong lap."""
    out, seen = [], set()
    for k in keys or ():
        p = parse_entry_key(k) if isinstance(k, str) else None
        if p and p[0] not in seen:
            seen.add(p[0])
            out.append(p[0])
    return out


# ---------------------------------------------------------------------------
class _Snap:
    """Moc thoi gian -> keyframe cua ta, cho tung video. Nap mot lan, giu lai."""

    def __init__(self, store):
        self.store = store
        self._cache = {}

    def of(self, video_id):
        c = self._cache.get(video_id)
        if c is None:
            df = self.store.frames_of(video_id)
            if df.empty:
                c = (np.zeros(0), np.zeros(0, dtype=int))
            else:
                c = (df["pts_time"].to_numpy(dtype=float),
                     df["frame_idx"].to_numpy(dtype=int))
            self._cache[video_id] = c
        return c

    def nearest(self, video_id, pts, tol):
        """(frame_idx, pts_that, lech) hoac None neu ngoai dung sai."""
        times, fidx = self.of(video_id)
        if times.size == 0:
            return None
        i = int(np.searchsorted(times, pts))
        best, bd = None, None
        for j in (i - 1, i):
            if 0 <= j < times.size:
                d = abs(times[j] - pts)
                if bd is None or d < bd:
                    best, bd = j, d
        if best is None or bd > tol:
            return None
        return int(fidx[best]), float(times[best]), float(bd)


# ---------------------------------------------------------------------------
class PanelSearch:
    def __init__(self, store, objects, channels, config, encoder=None):
        self.store = store
        self.objects = objects
        self.channels = channels
        self.cfg = config
        self.encoder = encoder
        self.snap = _Snap(store)

    # -- /panel ----------------------------------------------------------
    def panel(self, tags=(), drag_objects=(), amount="", ocr="", asr="",
              ids=None, useid=False, ignore_idxs=None, k=500, topk_kf=None):
        """Tim theo LOP object + VI TRI + chu tren hinh + loi noi.

        Tra ve (videos, warnings).
        """
        warnings = []
        boxes = []
        for o in drag_objects or ():
            pos = (o or {}).get("position") or {}
            try:
                # UI goi truc DOC la "x" (xTop = offsetTop/PANEL) va truc NGANG
                # la "y". OpenImages ghi [ymin, xmin, ymax, xmax]. Doi cho hai
                # cai nay thi hop bi lat cheo, ket qua sai ma khong bao loi.
                box = [float(pos["xTop"]), float(pos["yTop"]),
                       float(pos["xBottom"]), float(pos["yBottom"])]
            except (KeyError, TypeError, ValueError):
                continue
            boxes.append({"type": (o or {}).get("type"), "box": box})

        counts = parse_counts(amount)
        if amount and not counts:
            warnings.append(
                f"khong doc duoc rang buoc so luong {amount!r} -- can dang "
                f"'ten_lop so_luong', vi du 'Person 2, Car 1'")

        # --- thu hep theo van ban truoc (re hon nhieu so voi loc object) ----
        restrict = None
        if useid and ids:
            restrict = videos_from_keys(ids)
            if not restrict:
                warnings.append("bat 'ID' nhung danh sach id khong co khoa hop le")
        for name, q in (("ocr", ocr), ("asr", asr)):
            if not (q or "").strip():
                continue
            hits = self.channels.search(name, q, topn=300)
            if not hits:
                warnings.append(f"kenh {name} khong khop video nao voi {q!r}")
                return [], warnings
            vids = [v for v, _ in hits]
            restrict = vids if restrict is None else [v for v in restrict if v in set(vids)]
            if not restrict:
                warnings.append(f"khong video nao thoa dong thoi ca ID lan {name}")
                return [], warnings

        if not (tags or boxes):
            # Khong co rang buoc object -> chi con van ban. Van tra ve ket qua
            # thay vi bao loi: nguoi dung go moi OCR cung phai thay gi do.
            if not restrict:
                warnings.append("chua chon lop object nao va cung khong co OCR/ASR")
                return [], warnings
            return self._frames_of_videos(restrict, k, ignore_idxs), warnings

        if not self.objects.ok:
            warnings.append(f"index object chua san sang: {self.objects.error}")
            return [], warnings

        topk_kf = topk_kf or max(k * 3, 1000)
        hits, note = self.objects.search(
            tags=tags, boxes=boxes, counts=counts,
            restrict_videos=restrict, topk=topk_kf)
        if note:
            warnings.append(note)
        if not hits:
            return [], warnings

        # --- bac cau keyframe BTC -> keyframe cua ta ------------------------
        tol = self.cfg.object_tolerance_sec
        skip = ignore_set(ignore_idxs)
        rows, seen, n_out_of_tol = [], set(), 0
        order, ordered_videos = {}, []
        for h in hits:
            vid = h["video_id"]
            got = self.snap.nearest(vid, h["pts_time"], tol)
            if got is None:
                n_out_of_tol += 1
                continue
            fi, pts, delta = got
            key = entry_key(vid, fi)
            if key in seen or key in skip:
                continue
            seen.add(key)
            if vid not in order:
                order[vid] = len(order)
                ordered_videos.append(vid)
            rows.append({"video_id": vid, "frame_idx": fi, "pts_time": pts,
                         "score": h["score"], "delta_sec": round(delta, 3)})
            if len(rows) >= k:
                break

        if n_out_of_tol:
            warnings.append(
                f"{n_out_of_tol} keyframe BTC bi bo vi khong co keyframe nao cua "
                f"ta trong +/-{tol}s")
        return pack_videos(rows, order=order), warnings

    def _frames_of_videos(self, video_ids, k, ignore_idxs=None):
        """Lay mau trai deu theo thoi gian cho moi video -- dung khi khong co
        rang buoc object nao de xep hang frame."""
        skip = ignore_set(ignore_idxs)
        budget = max(1, k // max(1, len(video_ids)))
        rows, order = [], {}
        for vid in video_ids:
            df = self.store.frames_of(vid)
            if df.empty:
                continue
            step = max(1, len(df) // budget)
            order.setdefault(vid, len(order))
            for r in df.iloc[::step].head(budget).to_dict("records"):
                if entry_key(vid, r["frame_idx"]) in skip:
                    continue
                rows.append({"video_id": vid, "frame_idx": int(r["frame_idx"]),
                             "pts_time": float(r["pts_time"]), "score": None})
        return pack_videos(rows, order=order)

    # -- /framerange -----------------------------------------------------
    def frame_range(self, video_id, start, end, text_query="", limit=200):
        """Keyframe cua mot video trong dai [start, end] tinh theo frame_idx."""
        if video_id not in self.store.video_slice:
            return {"error": f"Video {video_id} khong co trong index"}
        start, end = int(start), int(end)
        if end < start:
            start, end = end, start

        df = self.store.frames_of(video_id)
        sub = df[(df["frame_idx"] >= start) & (df["frame_idx"] <= end)]
        message = None

        if sub.empty:
            # Dai nam LOT giua hai keyframe. Van xem duoc: anh trich thang tu
            # mp4 theo frame_idx nen frame_idx nao cung dung anh, khong nhat
            # thiet phai la keyframe.
            fps = self.store.fps.get(video_id) or 25.0
            n = min(limit, 30)
            step = max(1, (end - start) // max(1, n - 1)) if end > start else 1
            idxs = list(range(start, end + 1, step))[:n] or [start]
            rows = [{"video_id": video_id, "frame_idx": i,
                     "pts_time": i / fps, "score": None} for i in idxs]
            message = (f"khong co keyframe nao trong [{start}, {end}] -- dang "
                       f"hien {len(rows)} frame trich thang tu video")
            videos = pack_videos(rows)
            return {"video_id": video_id, "video_info": videos[0]["video_info"],
                    "message": message, "n": len(rows), "exact": False}

        rows = [{"video_id": video_id, "frame_idx": int(r["frame_idx"]),
                 "pts_time": float(r["pts_time"]), "score": None}
                for r in sub.to_dict("records")]

        q = (text_query or "").strip()
        if q:
            if self.encoder is None:
                message = "khong co encoder -- bo qua o tim trong dai"
            else:
                try:
                    self.store.assert_encoder_matches(self.encoder)
                    gid = sub["gidx"].to_numpy(dtype=int)

                    # O nay nhan van ban tu do cua nguoi dung, tuc la TIENG VIET
                    # va co the RAT DAI. Hai rao can, thieu cai nao cung thanh
                    # loi im lang:
                    #   · dich sang tieng Anh -- SigLIP huan luyen tieng Anh,
                    #     dua tieng Viet vao thi cosine van dep ma vo nghia
                    #   · encode_query thay vi encode_texts -- tu chia cho vua
                    #     64 token thay vi de tokenizer cat cut khong bao gi
                    qen, dich = q, False
                    try:
                        from retrieval.query import _translate

                        t = (_translate(q) or "").strip()
                        if t and t.lower() != q.strip().lower():
                            qen, dich = t, True
                    except Exception:
                        pass

                    v = self.encoder.encode_query(qen)
                    if v is None:
                        raise ValueError("truy van rong")
                    sc = self.store.X[gid] @ v
                    for r, s in zip(rows, sc):
                        r["score"] = float(s)
                    rows.sort(key=lambda r: -r["score"])
                    n_manh = len(self.encoder.pack_text(qen))
                    message = (f"xep theo do khop voi {qen!r}"
                               + (" (da dich)" if dich else "")
                               + (f", chia lam {n_manh} manh cho vua gioi han token"
                                  if n_manh > 1 else "")
                               + "; frame dau la khop nhat trong dai")
                except Exception as e:
                    message = f"khong tim duoc trong dai: {type(e).__name__}: {e}"

        rows = rows[:limit]
        videos = pack_videos(rows)
        return {"video_id": video_id, "video_info": videos[0]["video_info"],
                "message": message, "n": len(rows), "exact": True}

    # -- /imgsearch ------------------------------------------------------
    def img_search(self, imgid, k=200, per_video=0):
        """KNN tu vector cua MOT keyframe -> cac keyframe giong nhat.

        imgid la khoa 'video_id#frame_idx'. Chap nhan ca chi so toan cuc dang
        so cho client cu.
        """
        gidx = self._gidx_of(imgid)
        if gidx is None:
            return [], f"khong tim thay keyframe {imgid!r}"
        if not self.store.X.size:
            return [], "chua nap duoc features"

        v = self.store.X[gidx]
        sims = self.store.X @ v
        n = min(int(k) + 1, sims.shape[0])
        part = np.argpartition(-sims, n - 1)[:n]
        part = part[np.argsort(-sims[part])]

        meta = self.store.meta
        rows, per = [], {}
        for g in part:
            r = meta.iloc[int(g)]
            vid = str(r["video_id"])
            if per_video and per.get(vid, 0) >= per_video:
                continue
            per[vid] = per.get(vid, 0) + 1
            rows.append({"video_id": vid, "frame_idx": int(r["frame_idx"]),
                         "pts_time": float(r["pts_time"]),
                         "score": float(sims[int(g)])})
            if len(rows) >= int(k):
                break
        # Giu thu tu do giong dan -- video cua chinh anh truy van len dau.
        order = {}
        for r in rows:
            order.setdefault(r["video_id"], len(order))
        return pack_videos(rows, order=order), None

    def _gidx_of(self, imgid):
        p = parse_entry_key(imgid) if isinstance(imgid, str) else None
        if p:
            vid, fi = p
            df = self.store.frames_of(vid)
            if df.empty:
                return None
            hit = df[df["frame_idx"] == fi]
            if not hit.empty:
                return int(hit.iloc[0]["gidx"])
            # frame_idx khong phai keyframe -- xay ra khi nguoi dung bam KNN tu
            # FrameRangeViewer (frame trich thang tu mp4, khong nam trong bang).
            # Lay keyframe gan nhat thay vi tra 404: anh nguoi dung dang nhin
            # cach vector gan nhat chua toi mot buoc lay mau.
            j = (df["frame_idx"] - fi).abs().idxmin()
            return int(df.loc[j, "gidx"])
        try:
            g = int(imgid)
        except (TypeError, ValueError):
            return None
        return g if 0 <= g < len(self.store.meta) else None

    # -- /relatedimg ------------------------------------------------------
    def related(self, imgid, k=12, span=6):
        """Bang canh cua mot keyframe: lang gieng THOI GIAN + lang gieng NGU NGHIA.

        Khac /imgsearch: o day khong di tim khap corpus ma tra ve boi canh cua
        chinh anh dang xem -- vai frame truoc/sau tren truc thoi gian, link
        YouTube goc kem moc giay, va moc de cat clip ngan. Do la thu can de
        chot "dung khoanh khac nay" hay khong.
        """
        p = parse_entry_key(imgid) if isinstance(imgid, str) else None
        if not p:
            g = self._gidx_of(imgid)
            if g is None:
                return {}
            r = self.store.meta.iloc[int(g)]
            p = (str(r["video_id"]), int(r["frame_idx"]))
        vid, fi = p

        df = self.store.frames_of(vid)
        if df.empty:
            return {}
        hit = df[df["frame_idx"] == fi]
        if hit.empty:
            j = (df["frame_idx"] - fi).abs().idxmin()
            row = df.loc[j]
        else:
            row = hit.iloc[0]
        pts = float(row["pts_time"])

        pos = int(df.index.get_loc(row.name))
        lo, hi = max(0, pos - span), min(len(df), pos + span + 1)
        near = []
        for r in df.iloc[lo:hi].to_dict("records"):
            near.append({
                "imgpath": keyframe_url(vid, r["frame_idx"]),
                "id": entry_key(vid, r["frame_idx"]),
                "video_id": vid,
                "frame_idx": int(r["frame_idx"]),
                "pts_time": round(float(r["pts_time"]), 3),
                "la_anh_dang_xem": int(r["frame_idx"]) == int(row["frame_idx"]),
            })

        info = {}
        mp = os.path.join(META_DIR, f"{vid}.json")
        if os.path.exists(mp):
            try:
                with open(mp, encoding="utf-8") as f:
                    info = json.load(f)
            except Exception:
                info = {}

        w = float(getattr(self.cfg, "clip_window_sec", 2.0))
        return {
            "video_id": vid,
            "frame_idx": int(row["frame_idx"]),
            "pts_time": round(pts, 3),
            "near_keyframes": near,
            # UI cu doc hai khoa nay -> giu nguyen ten.
            "video_url": info.get("watch_url") or "",
            "video_range": [max(0, int(pts - w)), int(pts + w)],
            "title": info.get("title") or "",
            "author": info.get("author") or "",
            "publish_date": info.get("publish_date") or "",
            # Clip ngan quanh frame -- phuc vu boi server LOCAL (media_url).
            "clip_url": (f"/clip/{vid}/{int(row['frame_idx']):06d}.mp4"
                         f"?window={w:g}"),
            "clip_window_sec": w,
        }
