"""Viec 4 -- cau noi object / OCR / ASR ve keyframe MOI.

Object va OCR cua BTC neo vao id keyframe CU, khong khop keyframe moi. Bac cau
qua pts_time, voi HAI DUNG SAI KHAC NHAU:

    object  3.0s   nguoi, xe, do vat hiem khi bien mat trong 3 giay
    ocr     1.5s   lower-third chi hien ~4 giay, do hoa doi lien tuc

Ngoai dung sai thi TRA VE RONG. Tuyet doi khong lay "gan nhat bat ke xa bao
nhieu" -- thong tin sai con te hon khong co thong tin. Moi ket qua xap xi deu
mang co approx=True de UI gan nhan "xap xi tu du lieu BTC".

ASR KHONG can cau noi: no neo theo thoi gian san, khop bang
pts_time thuoc [start - pad, end + pad].
"""

import bisect
import csv
import glob
import json
import os

from retrieval.config import ARTIFACT_ROOT, FusionConfig

OBJECT_ROOT = os.path.join(ARTIFACT_ROOT, "objects-aic25-b1", "objects")
# Layout chuan cua BTC neu bo map-keyframes duoc bo sung. Thieu file nay thi
# object khong co moc thoi gian nao ca -> kenh object tat, xem `object_status`.
MAP_KEYFRAME_DIRS = (
    os.path.join(ARTIFACT_ROOT, "map-keyframes-aic25-b1", "map-keyframes"),
    os.path.join(ARTIFACT_ROOT, "map-keyframes"),
)


def _nearest(sorted_times, t):
    """Chi so cua phan tu gan t nhat trong danh sach da sap xep."""
    if not sorted_times:
        return None
    i = bisect.bisect_left(sorted_times, t)
    best, bd = None, None
    for j in (i - 1, i):
        if 0 <= j < len(sorted_times):
            d = abs(sorted_times[j] - t)
            if bd is None or d < bd:
                best, bd = j, d
    return best


class ContextBridge:
    def __init__(self, store, config=None, root=ARTIFACT_ROOT):
        self.store = store
        self.cfg = config or FusionConfig.load()
        self.root = root
        self._ocr = {}
        self._asr = {}
        self._objmap = {}          # video_id -> ([pts_time...], [duong dan json...])
        self._map_dir = next((d for d in MAP_KEYFRAME_DIRS if os.path.isdir(d)), None)
        self.object_videos = set()
        if os.path.isdir(OBJECT_ROOT):
            self.object_videos = set(os.listdir(OBJECT_ROOT))

    # ---------------- OCR -------------------------------------------------
    def _ocr_of(self, video_id):
        if video_id in self._ocr:
            return self._ocr[video_id]
        fps = self.store.fps.get(video_id) or 25.0
        entry = ([], [])
        for f in glob.glob(os.path.join(self.root, "*", "ocr", f"{video_id}.json")):
            try:
                d = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            rows = []
            for fr in d.get("frames") or []:
                fi = fr.get("frame_idx")
                if fi is None:
                    continue
                rows.append((float(fi) / fps, int(fi), fr.get("items") or []))
            rows.sort()
            entry = ([r[0] for r in rows], rows)
            break
        self._ocr[video_id] = entry
        return entry

    def ocr_at(self, video_id, pts_time, frame_idx=None, tol=None):
        """Chu tren man hinh tai thoi diem nay.

        Uu tien khop CHINH XAC theo frame_idx (OCR moi neo thang vao keyframe
        moi -- 181/200 video trung khop hoan toan). Chi khi khong co moi roi ve
        cau noi trong dung sai 1.5s.
        """
        tol = self.cfg.ocr_tolerance_sec if tol is None else tol
        times, rows = self._ocr_of(video_id)
        if not rows:
            return {"items": [], "approx": False, "source": "artifacts/ocr",
                    "reason": "video khong co du lieu OCR"}

        if frame_idx is not None:
            for _, fi, items in rows:
                if fi == int(frame_idx):
                    return {"items": items, "approx": False,
                            "source": "artifacts/ocr", "delta_sec": 0.0}

        i = _nearest(times, float(pts_time))
        if i is None:
            return {"items": [], "approx": False, "source": "artifacts/ocr"}
        delta = abs(times[i] - float(pts_time))
        if delta > tol:
            # Ngoai dung sai -> rong. Do hoa doi lien tuc, lay xa hon la bia.
            return {"items": [], "approx": False, "source": "artifacts/ocr",
                    "reason": f"khong co OCR trong +/-{tol}s (gan nhat {delta:.2f}s)"}
        return {"items": rows[i][2], "approx": True, "source": "artifacts/ocr",
                "delta_sec": round(delta, 3), "frame_idx": rows[i][1],
                "tolerance_sec": tol}

    # ---------------- ASR -------------------------------------------------
    def _asr_of(self, video_id):
        if video_id in self._asr:
            return self._asr[video_id]
        segs = []
        for f in glob.glob(os.path.join(self.root, "*", "asr", f"{video_id}.json")):
            try:
                d = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            for s in d.get("segments") or []:
                try:
                    segs.append((float(s["start"]), float(s["end"]),
                                 s.get("text", ""), s.get("avg_logprob")))
                except (KeyError, TypeError, ValueError):
                    continue
            break
        segs.sort()
        self._asr[video_id] = segs
        return segs

    def asr_at(self, video_id, pts_time, pad=None):
        """ASR neo theo thoi gian -> chinh xac tuyet doi, khong phai cau noi."""
        pad = self.cfg.asr_pad_sec if pad is None else pad
        t = float(pts_time)
        out = [{"start": a, "end": b, "text": txt, "avg_logprob": lp}
               for a, b, txt, lp in self._asr_of(video_id)
               if a - pad <= t <= b + pad]
        return {"segments": out, "approx": False, "source": "artifacts/asr",
                "pad_sec": pad}

    # ---------------- Object ----------------------------------------------
    def _obj_of(self, video_id):
        """([pts_time], [duong dan json]) cho keyframe BTC cua video.

        Cac file la objects/{video_id}/{NNN}.json voi NNN la THU TU keyframe BTC
        dem tu 1. Ban than chung khong mang moc thoi gian nao, nen phai lay
        pts_time tu map-keyframes cua BTC. Thieu map-keyframes -> khong co cach
        nao dat object vao truc thoi gian, va cau noi phai tat han.
        """
        if video_id in self._objmap:
            return self._objmap[video_id]

        d = os.path.join(OBJECT_ROOT, video_id)
        if not os.path.isdir(d) or self._map_dir is None:
            self._objmap[video_id] = ([], [])
            return self._objmap[video_id]

        m = os.path.join(self._map_dir, f"{video_id}.csv")
        if not os.path.exists(m):
            self._objmap[video_id] = ([], [])
            return self._objmap[video_id]

        times_by_n = {}
        try:
            with open(m, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    times_by_n[int(r["n"])] = float(r["pts_time"])
        except Exception:
            self._objmap[video_id] = ([], [])
            return self._objmap[video_id]

        rows = []
        for name in os.listdir(d):
            if not name.endswith(".json"):
                continue
            try:
                n = int(name[:-5])
            except ValueError:
                continue
            t = times_by_n.get(n)
            if t is not None:
                rows.append((t, os.path.join(d, name)))
        rows.sort()
        self._objmap[video_id] = ([r[0] for r in rows], [r[1] for r in rows])
        return self._objmap[video_id]

    def objects_at(self, video_id, pts_time, tol=None, min_score=0.15, top=12):
        tol = self.cfg.object_tolerance_sec if tol is None else tol
        times, paths = self._obj_of(video_id)
        if not times:
            reason = ("khong co map-keyframes cua BTC nen object chua co moc "
                      "thoi gian de bac cau")
            if video_id not in self.object_videos:
                reason = "video khong co du lieu object"
            return {"items": [], "approx": False, "source": "artifacts/objects",
                    "reason": reason}

        i = _nearest(times, float(pts_time))
        delta = abs(times[i] - float(pts_time))
        if delta > tol:
            return {"items": [], "approx": False, "source": "artifacts/objects",
                    "reason": f"khong co object trong +/-{tol}s (gan nhat {delta:.2f}s)"}

        try:
            d = json.load(open(paths[i], encoding="utf-8"))
        except Exception:
            return {"items": [], "approx": False, "source": "artifacts/objects",
                    "reason": "khong doc duoc file object"}

        ents = d.get("detection_class_entities") or []
        scores = d.get("detection_scores") or []
        boxes = d.get("detection_boxes") or []
        items = []
        for j, e in enumerate(ents):
            try:
                s = float(scores[j])
            except (IndexError, TypeError, ValueError):
                continue
            if s < min_score:
                continue
            items.append({"entity": e, "score": round(s, 4),
                          "box": [float(x) for x in boxes[j]] if j < len(boxes) else None})
        items.sort(key=lambda x: -x["score"])
        return {"items": items[:top], "approx": True, "source": "artifacts/objects",
                "delta_sec": round(delta, 3), "tolerance_sec": tol}

    # ---------------- gop ---------------------------------------------------
    def context_at(self, video_id, frame_idx):
        df = self.store.frames_of(video_id)
        hit = df[df["frame_idx"] == int(frame_idx)] if not df.empty else df
        if hit.empty:
            fps = self.store.fps.get(video_id) or 25.0
            pts = float(frame_idx) / fps
        else:
            pts = float(hit.iloc[0]["pts_time"])
        return {
            "video_id": video_id,
            "frame_idx": int(frame_idx),
            "pts_time": pts,
            "objects": self.objects_at(video_id, pts),
            "ocr": self.ocr_at(video_id, pts, frame_idx=frame_idx),
            "asr": self.asr_at(video_id, pts),
        }

    def object_status(self):
        return {
            "object_dir": OBJECT_ROOT if os.path.isdir(OBJECT_ROOT) else None,
            "n_video_co_object": len(self.object_videos),
            "map_keyframes_dir": self._map_dir,
            "kha_dung": self._map_dir is not None,
            "ghi_chu": (
                None if self._map_dir else
                "objects/{video}/{NNN}.json danh so theo thu tu keyframe BTC (tu 1) "
                "va khong mang pts_time. Can map-keyframes cua BTC "
                "(n,pts_time,fps,frame_idx) dat tai " + MAP_KEYFRAME_DIRS[0]
            ),
        }
