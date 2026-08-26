"""Tang hop nhat cap video.

    query -> [siglip_video_rank, meta.search, ...] -> rrf() -> top N video
          -> frames_in_videos(top N) -> luoi ket qua frame

Hai chuoi truy van la CO Y: metadata/ASR/OCR la tieng Viet va gia tri lon nhat
nam o khop chinh xac danh tu rieng ("Cho Lon", "Nhan Nghia Duong"); SigLIP huan
luyen chu yeu tieng Anh va khong hieu danh tu rieng tieng Viet.
"""

from fusion import explain, frames_in_videos, rrf, siglip_video_rank
from retrieval.config import FusionConfig
from retrieval.encoder import SigLipTextEncoder
from retrieval.store import ArtifactStore
from retrieval.textindex import TextChannels

# Cac kenh chay bang truy van TIENG VIET (BM25).
VI_CHANNELS = ("meta", "meta_fold", "asr", "ocr")


class FusionEngine:
    def __init__(self, store=None, channels=None, encoder=None, config=None):
        self.cfg = config or FusionConfig.load()
        self.store = store if store is not None else ArtifactStore()
        self.channels = channels if channels is not None else TextChannels()
        self.encoder = encoder if encoder is not None else SigLipTextEncoder()

    # ------------------------------------------------------------------
    def rank_videos(self, query_en=None, query_vi=None, topn=None,
                    weights=None, channels=None):
        """-> (fused, lists) voi fused = [(video_id, diem_rrf)] giam dan."""
        topn = topn or self.cfg.channel_topn
        wanted = set(channels) if channels else None
        lists = {}

        # --- kenh siglip: TIENG ANH ---------------------------------------
        if query_en and query_en.strip() and (wanted is None or "siglip" in wanted):
            queries = [q.strip() for q in query_en.split("\n") if q.strip()]
            try:
                self.store.assert_encoder_matches(self.encoder)
                lists["siglip"] = siglip_video_rank(
                    self.store.X, self.store.meta, self.encoder, queries, topn=topn)
            except Exception as e:
                # Kenh siglip hong khong duoc lam sap ca truy van -- cac kenh
                # van ban van chay va van cho ra xep hang dung nghia.
                lists.setdefault("_errors", {})
                lists["_errors"]["siglip"] = f"{type(e).__name__}: {e}"

        # --- cac kenh BM25: TIENG VIET ------------------------------------
        if query_vi and query_vi.strip():
            for name in VI_CHANNELS:
                if wanted is not None and name not in wanted:
                    continue
                r = self.channels.search(name, query_vi, topn=topn)
                if r:
                    lists[name] = r

        errors = lists.pop("_errors", {})
        w = dict(self.cfg.weights)
        if weights:
            w.update(weights)
        fused = rrf(lists, k=self.cfg.rrf_k, weights=w)
        return fused, lists, errors

    # ------------------------------------------------------------------
    def search(self, query_en=None, query_vi=None, video_topn=None,
               frame_topk=None, weights=None, channels=None,
               ignore_gidx=None, restrict_videos=None):
        """Luong day du -> danh sach video kem frame, dung dang UI dang dung."""
        video_topn = video_topn or self.cfg.video_topn
        frame_topk = frame_topk or self.cfg.frame_topk

        fused, lists, errors = self.rank_videos(
            query_en, query_vi, weights=weights, channels=channels)

        if restrict_videos:
            keep = set(restrict_videos)
            fused = [(v, s) for v, s in fused if v in keep]

        # Chi giu video that su co trong index (metadata co 873 video, artifacts
        # chi co 765 -- kenh meta co the tra ve video chua nam trong index).
        fused = [(v, s) for v, s in fused if v in self.store.video_slice]
        top = fused[:video_topn]
        top_ids = [v for v, _ in top]
        if not top_ids:
            return {"videos": [], "channels": self._channel_summary(lists, query_vi),
                    "errors": errors, "n_videos_ranked": 0}

        frames = self._frames(query_en, top_ids, frame_topk, ignore_gidx)

        rrf_score = dict(top)
        order = {v: i for i, v in enumerate(top_ids)}
        grouped = {}
        for row in frames:
            grouped.setdefault(row["video_id"], []).append(row)

        videos = []
        for vid in sorted(grouped, key=lambda v: order.get(v, 1 << 30)):
            rows = grouped[vid]
            videos.append({
                "video_id": vid,
                "rrf_score": round(float(rrf_score.get(vid, 0.0)), 6),
                "explain": explain(lists, vid),
                "why": {n: self.channels.why(n, vid, query_vi)
                        for n in VI_CHANNELS if n in lists},
                "video_info": {
                    "lst_keyframe_paths": [r["path"] for r in rows],
                    "lst_idxs": [r["gidx"] for r in rows],
                    "lst_keyframe_idxs": [r["frame_idx"] for r in rows],
                    "lst_pts_times": [r["pts_time"] for r in rows],
                    "lst_scores": [r["score"] for r in rows],
                },
            })

        return {
            "videos": videos,
            "channels": self._channel_summary(lists, query_vi),
            "errors": errors,
            "n_videos_ranked": len(fused),
        }

    # ------------------------------------------------------------------
    def _frames(self, query_en, video_ids, topk, ignore_gidx=None):
        ignore = set(ignore_gidx or ())
        use_siglip = bool(query_en and query_en.strip())
        if use_siglip:
            try:
                self.store.assert_encoder_matches(self.encoder)
            except Exception:
                use_siglip = False

        if use_siglip:
            q = query_en.strip().split("\n")[0].strip()
            df = frames_in_videos(self.store.X, self.store.meta, self.encoder,
                                  q, video_ids, topk=topk + len(ignore))
            rows = df.to_dict("records")
        else:
            # Khong co encoder: van tra ve frame cua dung nhung video da duoc
            # xep hang boi cac kenh van ban. Chia deu han muc cho tung video --
            # neu chi sort roi cat thi ca han muc roi vao mot video dau bang.
            budget = max(1, (topk + len(ignore)) // max(1, len(video_ids)))
            rows = []
            for vid in video_ids:
                df = self.store.frames_of(vid)
                if df.empty:
                    continue
                # lay mau trai deu theo thoi gian de bao quat ca video
                step = max(1, len(df) // budget)
                picked = df.iloc[::step].head(budget)
                for r in picked.to_dict("records"):
                    r["score"] = None
                    rows.append(r)

        out = []
        for r in rows:
            g = int(r["gidx"])
            if g in ignore:
                continue
            vid = r["video_id"]
            fi = int(r["frame_idx"])
            out.append({
                "gidx": g,
                "video_id": vid,
                "frame_idx": fi,
                "pts_time": float(r["pts_time"]),
                "score": (None if r.get("score") is None else float(r["score"])),
                "path": keyframe_url(vid, fi),
            })
            if len(out) >= topk:
                break
        return out

    def _channel_summary(self, lists, query_vi):
        return {
            n: {"n": len(lst),
                "top": [v for v, _ in lst[:5]],
                "query_terms": (self.channels.why(n, lst[0][0], query_vi)
                                if n in VI_CHANNELS and lst else [])}
            for n, lst in lists.items()
        }


def keyframe_url(video_id, frame_idx):
    """Duong dan anh keyframe MOI -- phuc vu on-demand tu mp4, xem retrieval.frames."""
    return f"/keyframe/{video_id}/{int(frame_idx):06d}.jpg"
