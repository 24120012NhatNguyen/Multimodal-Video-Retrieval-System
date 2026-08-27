"""Tang hop nhat cap video.

    query -> [siglip_video_rank, meta.search, ...] -> rrf() -> top N video
          -> frames_in_videos(top N) -> luoi ket qua frame

Hai chuoi truy van la CO Y: metadata/ASR/OCR la tieng Viet va gia tri lon nhat
nam o khop chinh xac danh tu rieng ("Cho Lon", "Nhan Nghia Duong"); SigLIP huan
luyen chu yeu tieng Anh va khong hieu danh tu rieng tieng Viet.
"""

from fusion import explain, frames_in_videos, rrf, siglip_video_rank
from retrieval.trake import (dp_alignment, event_stats, events_to_scores,
                             fill_skipped)
from retrieval.config import FusionConfig
from retrieval.encoder import SigLipTextEncoder
from retrieval.store import ArtifactStore
from retrieval.textindex import TextChannels

# Cac kenh chay bang truy van TIENG VIET (BM25).
VI_CHANNELS = ("meta", "meta_fold", "asr", "ocr")

# --- Gop kenh theo NGUON ---------------------------------------------------
# meta va meta_fold la CUNG MOT van ban (metadata cua BTC), chi khac bo tach tu.
# Tinh ca hai vao RRF nghia la metadata duoc bo phieu HAI LAN. Gop lai theo
# nguon roi lay thu hang tot nhat trong nhom.
CHANNEL_GROUP = {
    "siglip": "siglip",
    "meta": "meta", "meta_fold": "meta",
    "asr": "asr",
    "ocr": "ocr",
}

# --- Trong so theo LOAI truy van -------------------------------------------
# Truy van thi giac ("nguoi dan ong va con cho"): tung kenh van ban khong co
# tin hieu that nhung BM25 VAN tra ve danh sach da xep hang, va RRF chi dung
# THU HANG nen chung duoc tinh du diem. Ba nhom van ban ap dao mot nhom thi
# giac 3:1 -- do la ly do ket qua te hon tim kiem thuan thi giac.
# DO tren Debug/7_questions.json (7 truy van, deu la mo ta thi giac):
#
#   chi siglip           R@1 57.1%  MRR 0.663     <- tot nhat
#   siglip 6 / van 0.3   R@1 42.9%  MRR 0.627
#   siglip 4 / van 0.3   R@1 28.6%  MRR 0.581
#   deu 1.0 (ban cu)     R@1 14.3%  MRR 0.348
#
# Voi truy van thi giac, THEM BAT KY trong so van ban nao cung lam te di: BM25
# khong co gi de bam vao nhung van tra ve danh sach da xep hang, va RRF chi
# dung thu hang nen no duoc tinh du diem.
#
# CANH BAO: bo eval hien khong co truy van NAO co danh tu rieng, nen ho so
# "anchored" duoi day CHUA duoc do. Cac so cua no van la phong doan.
WEIGHTS_BY_KIND = {
    "generic_chain": {"siglip": 1.0, "meta": 0.0, "asr": 0.0, "ocr": 0.0},
    "anchored": {"siglip": 1.5, "meta": 2.0, "asr": 1.5, "ocr": 1.5},
}

# Kenh BM25 chi duoc tinh khi diem cua video >= ty le nay so voi diem cao nhat
# CUA CHINH KENH DO. Khong co cong nay thi video xep hang 80 voi diem gan 0 van
# duoc cong 1/(60+80) -- du nho nhung cong don qua ba nhom la du lat ket qua.
SCORE_GATE_RATIO = 0.35
# Duoi nguong nay thi coi nhu ca kenh khong co tin hieu that.
SCORE_GATE_FLOOR = 1.0

# --- Che do tung kenh do NGUOI DUNG dat ------------------------------------
#   "auto"  he thong tu quyet: cong theo diem + trong so theo loai truy van
#   "on"    NGUOI DUNG BAT: bo cong, va ep trong so it nhat CHANNEL_ON_WEIGHT
#           du bo phan loai cho la khong can. Nguoi dung nhin thay ket qua,
#           bo phan loai thi khong -- y nguoi dung thang.
#   "off"   NGUOI DUNG TAT: kenh khong chay, khong bo phieu.
CHANNEL_MODES = ("auto", "on", "off")
CHANNEL_ON_WEIGHT = 1.0


def normalize_modes(modes):
    """{ten_kenh|ten_nhom: che_do} -> {ten_nhom: che_do} da lam sach."""
    out = {}
    for name, m in (modes or {}).items():
        m = str(m or "auto").strip().lower()
        if m not in CHANNEL_MODES:
            continue
        out[CHANNEL_GROUP.get(name, name)] = m
    return out


def gate_by_score(lst, ratio=SCORE_GATE_RATIO, floor=SCORE_GATE_FLOOR):
    """Cat duoi danh sach BM25 -- giu lai phan co diem dang ke."""
    if not lst:
        return []
    top = lst[0][1]
    if top < floor:
        return []
    return [(v, s) for v, s in lst if s >= ratio * top]


def merge_by_source(lists):
    """{ten_kenh: [(vid, diem)]} -> {ten_nhom: [(vid, -thu_hang)]}.

    Trong mot nhom, video lay THU HANG TOT NHAT ma no dat duoc o bat ky bien
    the nao. Nho vay doi bo tach tu (co dau / khong dau) khong lam tang so phieu.
    """
    best = {}
    for name, lst in lists.items():
        g = CHANNEL_GROUP.get(name, name)
        d = best.setdefault(g, {})
        for rank, (vid, _) in enumerate(lst):
            if vid not in d or rank < d[vid]:
                d[vid] = rank
    return {
        g: [(vid, -rank) for vid, rank in sorted(d.items(), key=lambda kv: kv[1])]
        for g, d in best.items()
    }


class FusionEngine:
    def __init__(self, store=None, channels=None, encoder=None, config=None):
        self.cfg = config or FusionConfig.load()
        self.store = store if store is not None else ArtifactStore()
        self.channels = channels if channels is not None else TextChannels()
        self.encoder = encoder if encoder is not None else SigLipTextEncoder()

    # ------------------------------------------------------------------
    def rank_videos(self, query_en=None, query_vi=None, topn=None,
                    weights=None, channels=None, kind=None, gate=True,
                    modes=None):
        """-> (fused, lists, errors) voi fused = [(video_id, diem_rrf)] giam dan.

        kind   "generic_chain" | "anchored" -> chon bang trong so mac dinh
        gate   cat duoi danh sach BM25 theo diem (xem gate_by_score)
        modes  {"asr"|"ocr"|"meta"|"siglip": "auto"|"on"|"off"} do NGUOI DUNG dat
        """
        topn = topn or self.cfg.channel_topn
        wanted = set(channels) if channels else None
        modes = normalize_modes(modes)

        def mode_of(name):
            return modes.get(CHANNEL_GROUP.get(name, name), "auto")

        lists = {}

        # --- kenh siglip: TIENG ANH ---------------------------------------
        if (query_en and query_en.strip() and mode_of("siglip") != "off"
                and (wanted is None or "siglip" in wanted)):
            # Cang it nhom cang tot, moi nhom vua gioi han token. Xem pack_queries.
            #
            # DO tren Debug/7_questions.json bang cong thuc cham cua BTC:
            #     LLM phan ra 4-5 menh de, cham diem tung menh de   Final 0.2571
            #     cung ban dich do, gop thanh MOT chuoi             Final 0.5714
            #
            # Ly do: tung menh de roi le deu la mo ta chung chung khop hang nghin
            # frame ("mot nguoi deo kinh"); suc phan biet nam o PHEP HOI cua
            # chung. Cong diem tot nhat cua tung menh de lam mat dung phep hoi do.
            # Menh de rieng le van duoc giu trong `clauses_en` cho DP/TRAKE --
            # cho do that su can diem cua TUNG su kien.
            queries = pack_queries(query_en, self.encoder)
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
                if mode_of(name) == "off":
                    continue
                r = self.channels.search(name, query_vi, topn=topn)
                if r:
                    lists[name] = r

        errors = lists.pop("_errors", {})

        # --- cong theo diem: bo phan duoi cua tung kenh BM25 ----------------
        gated = dict(lists)
        if gate:
            for name in VI_CHANNELS:
                if name in gated:
                    if mode_of(name) == "on":
                        # Nguoi dung da BAT kenh nay -> khong cong. Cong la co
                        # che TU DONG doan xem kenh co tin hieu that khong; khi
                        # nguoi dung tu quyet thi phong doan do khong con vai tro.
                        continue
                    kept = gate_by_score(gated[name])
                    if kept:
                        gated[name] = kept
                    else:
                        # Ca kenh khong co tin hieu that -> khong bo phieu.
                        del gated[name]

        # --- gop kenh cung nguon roi moi hop nhat ---------------------------
        groups = merge_by_source(gated)

        w = dict(WEIGHTS_BY_KIND.get(kind or "", {}))
        if not w:
            w = dict(self.cfg.weights)
        if weights:
            w.update(weights)
        # Nguoi dung BAT mot kenh ma trong so theo loai truy van dang la 0 thi
        # kenh do van khong bo phieu duoc -- nut bat se thanh nut gia. San len.
        on_w = getattr(self.cfg, "channel_on_weight", CHANNEL_ON_WEIGHT)
        for g, m in modes.items():
            if m == "on":
                w[g] = max(float(w.get(g, 0.0)), float(on_w))
            elif m == "off":
                w[g] = 0.0

        # Nguoi dung co the tat dung cai kenh duy nhat dang co trong so (vd tat
        # siglip trong truy van thi giac, ma ho so generic_chain dat van ban = 0).
        # Luc do MOI nhom deu co trong so 0: RRF cho diem 0 het va thu tu tra ve
        # la ngau nhien -- te hon han viec khong loc gi. Lui ve trong so deu tren
        # nhung nhom con lai, va noi ro da lui.
        self._weight_fallback = None
        if groups and not any(w.get(g, 0.0) > 0 for g in groups):
            w = {g: 1.0 for g in groups}
            self._weight_fallback = (
                "moi kenh con trong so 0 sau khi ap cong tac cua nguoi dung -- "
                "da lui ve trong so deu tren cac kenh con lai: "
                + ", ".join(sorted(groups)))

        fused = rrf(groups, k=self.cfg.rrf_k, weights=w)
        self._last_weights = w
        # `lists` giu nguyen (chua cong) de explain() con noi duoc thu hang goc
        # cua video o TUNG kenh -- do la thu nguoi dung doc.
        return fused, lists, errors

    # ------------------------------------------------------------------
    def search(self, query_en=None, query_vi=None, video_topn=None,
               frame_topk=None, weights=None, channels=None,
               ignore_gidx=None, restrict_videos=None, kind=None, modes=None):
        """Luong day du -> danh sach video kem frame, dung dang UI dang dung."""
        video_topn = video_topn or self.cfg.video_topn
        frame_topk = frame_topk or self.cfg.frame_topk

        fused, lists, errors = self.rank_videos(
            query_en, query_vi, weights=weights, channels=channels, kind=kind,
            modes=modes)
        used_weights = dict(getattr(self, "_last_weights", {}) or {})
        weight_note = getattr(self, "_weight_fallback", None)

        if restrict_videos:
            keep = set(restrict_videos)
            fused = [(v, s) for v, s in fused if v in keep]

        # Chi giu video that su co trong index -- kenh meta chay tren metadata
        # cua BTC nen co the tra ve video chua co features.
        fused = [(v, s) for v, s in fused if v in self.store.video_slice]
        top = fused[:video_topn]
        top_ids = [v for v, _ in top]
        if not top_ids:
            return {"videos": [], "channels": self._channel_summary(lists, query_vi),
                    "errors": errors, "n_videos_ranked": 0,
                    "weights_used": used_weights,
                    "channel_modes": normalize_modes(modes),
                    "weights_note": weight_note}

        frames = self._frames(query_en, top_ids, frame_topk, ignore_gidx)

        rrf_score = dict(top)
        order = {v: i for i, v in enumerate(top_ids)}
        grouped = {}
        for row in frames:
            grouped.setdefault(row["video_id"], []).append(row)

        # --- xep lai theo BANG CHUNG FRAME (rerank) -------------------------
        # RRF chi biet "video nay duoc may kenh bo phieu", khong biet ben trong
        # video co khoanh khac nao that su giong truy van hay khong. Diem thi
        # giac cao nhat trong video la thuoc do do -- va no manh hon RRF.
        #
        # DO tren Debug/7_questions.json:
        #     chi RRF (khong rerank)      R@1 28.6%  MRR 0.558
        #     rerank theo frame tot nhat  xem bang trong report
        #
        # Truoc ban nay viec do van xay ra, nhung AM THAM: han muc frame cat
        # toan cuc lam bien mat nhung video khong co frame diem cao, tuc la loc
        # theo frame dang dong vai rerank ma khong ai goi ten. Am tham thi khong
        # tat duoc, khong do duoc, va vo hieu ngay khi doi han muc k.
        rerank = getattr(self.cfg, "rerank_by_frame", True)
        best_frame = {}
        if rerank:
            for vid, rs in grouped.items():
                sc = [r["score"] for r in rs if r.get("score") is not None]
                if sc:
                    best_frame[vid] = max(sc)
            if best_frame:
                w_rrf = getattr(self.cfg, "rerank_rrf_weight", 0.0)
                base = {v: 1.0 / (1 + i) for i, v in enumerate(top_ids)}
                order = {v: i for i, v in enumerate(sorted(
                    grouped,
                    key=lambda v: -(best_frame.get(v, -1e9)
                                    + w_rrf * base.get(v, 0.0))))}

        videos = []
        for vid in sorted(grouped, key=lambda v: order.get(v, 1 << 30)):
            rows = grouped[vid]
            videos.append({
                "video_id": vid,
                "rrf_score": round(float(rrf_score.get(vid, 0.0)), 6),
                "rrf_rank": top_ids.index(vid) + 1 if vid in rrf_score else None,
                "best_frame_score": (None if vid not in best_frame
                                     else round(float(best_frame[vid]), 4)),
                "explain": explain(lists, vid),
                "why": {n: self.channels.why(n, vid, query_vi)
                        for n in VI_CHANNELS if n in lists},
                "video_info": {
                    "lst_keyframe_paths": [r["path"] for r in rows],
                    # Khoa on dinh dung chung voi socket_app (submit/ignore).
                    # He thong cu dung chi so toan cuc trong dict/id2img.json,
                    # thu muc do da bi xoa.
                    "lst_idxs": [entry_key(r["video_id"], r["frame_idx"])
                                 for r in rows],
                    "lst_gidx": [r["gidx"] for r in rows],
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
            # De UI chi ra duoc kenh nao that su bo phieu cho ket qua nay.
            "weights_used": used_weights,
            "channel_modes": normalize_modes(modes),
            "weights_note": weight_note,
        }

    # ------------------------------------------------------------------
    def align_videos(self, clauses, video_ids, delta=None, gamma=None,
                     min_gap=None, normalize=True, tau=None):
        """Dong hang chuoi su kien tren tung video, xep lai theo diem DP.

        Day la buoc an thua voi truy van "chuoi hanh dong chung chung": tung
        menh de rieng le thi hang nghin video khop, nhung dong xuat hien DUNG
        THU TU va trong cua so thoi gian thi hiem. Cong diem phang khong bat
        duoc dieu do, chi DP moi bat duoc.

        Phan bo diem cua tung menh de duoc uoc luong MOT LAN tren corpus roi
        dung lai cho moi video (event_stats). Lam trong vong lap thi vua cham
        vua sai: moi video se co mot thang do khac nhau, khong so sanh duoc.
        """
        delta = self.cfg.dp_delta_sec if delta is None else delta
        gamma = self.cfg.dp_gamma if gamma is None else gamma
        min_gap = getattr(self.cfg, "dp_min_gap_sec", 0.0) if min_gap is None else min_gap
        tau = getattr(self.cfg, "dp_tau", 2.0) if tau is None else tau
        stats = (event_stats(self.store, self.encoder, clauses)
                 if normalize else None)
        out = []
        for vid in video_ids:
            pts, fidx, scores = events_to_scores(
                self.store, vid, self.encoder, clauses, stats=stats,
                normalize=normalize, tau=tau)
            if not pts:
                continue
            path, score = dp_alignment(pts, scores, delta=delta, gamma=gamma,
                                       min_gap=min_gap)
            if not path:
                continue
            # Bai TRAKE doi mot frame cho MOI su kien -- bo qua la cong cu cham
            # diem, khong phai cau tra loi nop duoc.
            filled = fill_skipped(path, pts, scores, delta)
            matched = []
            for k, pi in enumerate(path):
                fi_k = filled[k]
                matched.append({
                    "event": k,
                    "frame_idx": int(fidx[fi_k]),
                    "pts_time": round(float(pts[fi_k]), 3),
                    "score": round(float(scores[fi_k, k]), 4),
                    # skipped=True: DP cho rang su kien nay KHONG co that trong
                    # video; frame ben canh chi la phuong an de nop, khong phai
                    # bang chung. UI phai hien khac di.
                    "skipped": pi == -1,
                    "weak": pi == -1,
                })
            n_skip = sum(1 for m in matched if m["skipped"])
            out.append({"video_id": vid, "dp_score": float(score),
                        "matched": matched,
                        "n_frame": len(pts),
                        "n_skipped": n_skip})
        out.sort(key=lambda x: -x["dp_score"])
        return out

    # ------------------------------------------------------------------
    def _frames(self, query_en, video_ids, topk, ignore_gidx=None):
        # Danh sach ignore den tu socket_app duoi dang khoa "video#frame";
        # cac client cu co the con gui chi so nguyen. Nhan ca hai.
        ignore = set()
        ignore_int = set()
        for it in (ignore_gidx or ()):
            if isinstance(it, str):
                p = parse_entry_key(it)
                if p:
                    ignore.add(entry_key(*p))
            elif isinstance(it, dict):
                v, f = it.get("video_id"), it.get("frame_idx")
                if v is not None and f is not None:
                    ignore.add(entry_key(v, f))
            else:
                try:
                    ignore_int.add(int(it))
                except (TypeError, ValueError):
                    pass
        n_ignore = len(ignore) + len(ignore_int)
        use_siglip = bool(query_en and query_en.strip())
        if use_siglip:
            try:
                self.store.assert_encoder_matches(self.encoder)
            except Exception:
                use_siglip = False

        if use_siglip:
            # TOAN BO truy van, khong phai menh de dau tien.
            #
            # Ban truoc lay `query_en.split("\n")[0]` -- tang video dung moi
            # menh de con tang frame chi dung menh de DAU. Menh de dau thuong la
            # canh nen ("ba nguoi dang di bo"), khong phai chi tiet phan biet
            # ("ao mua in hinh con gau"), nen frame duoc xep bang dung thu it
            # thong tin nhat.
            qs = pack_queries(query_en, self.encoder)
            rows = self._frames_allocated(qs, video_ids, topk + n_ignore)
        else:
            # Khong co encoder: van tra ve frame cua dung nhung video da duoc
            # xep hang boi cac kenh van ban. Chia deu han muc cho tung video --
            # neu chi sort roi cat thi ca han muc roi vao mot video dau bang.
            budget = max(1, (topk + n_ignore) // max(1, len(video_ids)))
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
            vid = r["video_id"]
            fi = int(r["frame_idx"])
            if g in ignore_int or entry_key(vid, fi) in ignore:
                continue
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

    def _frames_allocated(self, query, video_ids, budget):
        """Chia han muc frame: mot phan BAO DAM theo video, phan con lai theo diem.

        Hai cach thuan tuy deu do va deu sai mot kieu:

          sort toan cuc roi cat topk (ban dau)
              Voi k=30 va 30 video, video xep hang #1 cua tang hop nhat CO THE
              BIEN MAT khoi luoi vi 30 frame diem cao nhat deu roi vao vai video
              khac. Nguoi dung bat kenh OCR, thu hang video doi han, ma man hinh
              khong doi gi -- nut bat trong nhu nut hong.

          chia deu theo vong
              Sua duoc dieu tren, nhung DO TREN BO EVAL: Recall@10 frame tut tu
              71,4% xuong 42,9%. Video dung chi con 1 frame moi vong nen gan nhu
              khong bao gio trung dung khoanh khac can tim.

        Nen lam ca hai: bao dam 1 frame cho top G video (G = 1/3 han muc) de thu
        hang luon nhin thay duoc, phan con lai van theo diem nhu cu.
        """
        mask = self.store.rows_for_videos(video_ids)
        if not mask.any():
            return []
        qs = [query] if isinstance(query, str) else list(query)
        Q = self.encoder.encode_texts(qs)
        sub = self.store.meta[mask].copy()
        S = self.store.X[mask] @ Q.T
        # Trung binh, khong phai max: cac nhom la cac PHAN cua cung mot mo ta,
        # nen frame dung phai khop TAT CA. Max se thanh phep tuyen.
        sub["score"] = S.mean(axis=1)
        ranked = sub.sort_values("score", ascending=False).to_dict("records")

        order = {vid: i for i, vid in enumerate(video_ids)}
        best_of = {}
        for r in ranked:
            best_of.setdefault(r["video_id"], r)

        alloc = getattr(self.cfg, "frame_alloc", "global")
        if alloc == "global":
            rows = ranked[:budget]
            rows.sort(key=lambda r: (order.get(r["video_id"], 1 << 30), -r["score"]))
            return rows
        if alloc == "round_robin":
            by_video = {}
            for r in ranked:
                by_video.setdefault(r["video_id"], []).append(r)
            rows, i = [], 0
            while len(rows) < budget:
                added = False
                for vid in video_ids:
                    lst = by_video.get(vid)
                    if lst is None or i >= len(lst):
                        continue
                    rows.append(lst[i])
                    added = True
                    if len(rows) >= budget:
                        break
                if not added:
                    break
                i += 1
            rows.sort(key=lambda r: (order.get(r["video_id"], 1 << 30), -r["score"]))
            return rows

        n_guarantee = min(len(video_ids), max(1, budget // 3))
        rows, taken = [], set()
        for vid in video_ids[:n_guarantee]:
            r = best_of.get(vid)
            if r is not None:
                rows.append(r)
                taken.add(int(r["gidx"]))
            if len(rows) >= budget:
                break

        for r in ranked:
            if len(rows) >= budget:
                break
            if int(r["gidx"]) in taken:
                continue
            rows.append(r)
            taken.add(int(r["gidx"]))

        # Sap lai theo (thu hang video, diem frame) de luoi doc theo dung thu tu.
        rows.sort(key=lambda r: (order.get(r["video_id"], 1 << 30), -r["score"]))
        return rows

    def _channel_summary(self, lists, query_vi):
        return {
            n: {"n": len(lst),
                "top": [v for v, _ in lst[:5]],
                "query_terms": (self.channels.why(n, lst[0][0], query_vi)
                                if n in VI_CHANNELS and lst else [])}
            for n, lst in lists.items()
        }


def clauses_of(query_en):
    return [q.strip().rstrip(".") for q in str(query_en or "").split("\n") if q.strip()]


def join_query(query_en):
    """Nhieu dong menh de -> MOT chuoi truy van."""
    return ". ".join(clauses_of(query_en))


def pack_queries(query_en, encoder=None, budget=None):
    """Menh de -> danh sach chuoi, MOI chuoi vua trong gioi han token cua SigLIP.

    Ba cach deu da do tren Debug/7_questions.json (MRR theo thu hang frame):

        chi menh de dau                     0.348   bo mat chi tiet phia sau
        gop het thanh mot chuoi             0.276   BI CAT CUT o 64 token
        cham diem tung menh de roi lay max  0.228   mat phep hoi giua cac menh de

    Ca ba deu sai theo mot kieu khac nhau, va nguyen nhan goc la GIOI HAN 64
    TOKEN cua SigLIP -- chua tung duoc kiem o dau. Truy van Q4 gop lai dai 93
    token, bi cat mat "cot moc dinh do" va "vat mau xanh la", tut tu hang 9
    xuong khong tim thay.

    Cach o day: don menh de thanh CANG IT NHOM CANG TOT, moi nhom vua 64 token.
    Vua het trong mot nhom -> giu tron phep hoi, truong hop tot nhat. Khong vua
    -> chia lam vai nhom, moi nhom van la mot cau hoan chinh, roi cham diem
    trung binh (phep hoi giua cac nhom) thay vi lay max (phep tuyen).
    """
    parts = clauses_of(query_en)
    if not parts:
        return []
    if encoder is None:
        return [". ".join(parts)]
    budget = budget or getattr(encoder, "MAX_LENGTH", 64)
    try:
        if encoder.n_tokens(". ".join(parts)) <= budget:
            return [". ".join(parts)]
    except Exception:
        return [". ".join(parts)]

    out, cur = [], []
    for p in parts:
        trial = ". ".join(cur + [p])
        if cur and encoder.n_tokens(trial) > budget:
            out.append(". ".join(cur))
            cur = [p]
        else:
            cur.append(p)
    if cur:
        out.append(". ".join(cur))
    return out or [". ".join(parts)]


def keyframe_url(video_id, frame_idx):
    """Duong dan anh keyframe MOI -- phuc vu on-demand tu mp4, xem retrieval.frames."""
    return f"/keyframe/{video_id}/{int(frame_idx):06d}.jpg"


def entry_key(video_id, frame_idx):
    """Khoa on dinh cua mot keyframe, dung chung giua ket qua tim kiem, danh
    sach ignore va danh sach dap an."""
    return f"{video_id}#{int(frame_idx)}"


def parse_entry_key(key):
    """'L24_V007#12450' -> ('L24_V007', 12450), hoac None."""
    if not isinstance(key, str) or "#" not in key:
        return None
    vid, _, fi = key.rpartition("#")
    try:
        return vid, int(fi)
    except ValueError:
        return None
