import os

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from retrieval import service
from utils.logger_config import get_logger
from utils.models import (
    AutofillRequest,
    FeedbackRequest,
    KeyframeContextRequest,
    PanelSearchRequest,
    TextSearchRequest,
    TrakeRequest,
    QaRequest,
)

logger = get_logger(__name__)

# Khởi tạo FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Eagerly initialize fusion engine on startup to fail-fast
# Kaggle stateless backend only relies on this.
try:
    _fusion_state = service.get()
    # Nap SigLIP NGAY luc khoi dong, khong doi truy van dau tien.
    # Nap luoi nghia la vai truy van dau chay khong co kenh thi giac va nguoi
    # dung khong he biet -- ho chi thay ket qua te. Dat SIGLIP_PRELOAD=0 de bo
    # qua (chay che do chi-BM25 co y thuc).
    if os.environ.get("SIGLIP_PRELOAD", "1") != "0":
        info = service.preload(strict=True)
        logger.info("SigLIP san sang: %s chieu tren %s", info["dim"], info["device"])
    # Index object khong co thi /panel tra ve rong -- do la che do xuong cap co
    # y (tim kiem van chay), nhung phai NOI TO luc khoi dong. Tren Kaggle,
    # data/artifacts la mount chi doc nen index phai duoc nap kem dataset hoac
    # dung lai; xem retrieval/objects.py.
    _oix = _fusion_state["objects"]
    if _oix.ok:
        logger.info("Index object: %s lop, %s detection (%s)",
                    len(_oix.entities), len(_oix.det_ent), _oix.path)
    else:
        logger.warning("KHONG CO INDEX OBJECT -- /panel se tra ve rong. %s",
                       _oix.error)
    logger.info("Fusion engine initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize fusion engine: {e}")
    _fusion_state = {"error": str(e)}

def get_fusion():
    if "error" in _fusion_state:
        return None
    return _fusion_state

@app.post("/textsearch")
def fusion_search(request: TextSearchRequest):
    """query -> nhieu kenh -> rrf() -> top N video -> frame cua nhung video do."""
    svc = get_fusion()
    if svc is None:
        return []

    from retrieval.query import decompose

    dq = decompose(
        query_vi=request.query_vi or request.textquery,
        query_en=request.query_en,
        use_llm=request.decompose,
        kind=request.kind,          # nguoi dung ep loai -> uu tien cao nhat
    )

    ignore_gidx = request.ignore_idxs if request.ignore else None

    # --- Loc lai trong ket qua cua luot truoc ------------------------------
    # Ba tham so `filter`, `filtervideo`, `videos` da duoc giao dien gui len tu
    # lau nhung backend KHONG HE DOC -- nguoi dung bat "Trong ket qua cu" roi
    # nhan ket qua nam ngoai tap da loc, ma khong co gi bao la nut do khong noi
    # voi ai. Gio doc that.
    restrict = None
    restrict_gidx = None
    prev_time = {}          # video_id -> moc thoi gian cua frame tot nhat luot truoc
    if request.filter and request.videos:
        restrict, prev_frames = [], []
        for v in request.videos:
            vid = v.get("video_id")
            if not vid:
                continue
            restrict.append(vid)
            vi = v.get("video_info") or {}
            ts = [t for t in (vi.get("lst_pts_times") or []) if t is not None]
            if ts:
                prev_time[vid] = float(ts[0])
            for t in ts:
                prev_frames.append((vid, float(t)))
        restrict = restrict or None

        # Loc o cap FRAME, khong chi cap video.
        #
        # Bao cao tu nguoi dung: tim "nguoi dan ong ao xanh la tren xich lo" ->
        # bat "Trong ket qua cu" -> go tiep "xe buyt mau cam" -> ket qua mat han
        # phan nguoi dan ong. Dung: gioi han theo VIDEO thi luot hai tim xe buyt
        # o BAT KY dau trong nhung video do.
        #
        # Gio giu lai chinh nhung frame cua luot truoc, cong them mot cua so thoi
        # gian hai ben -- vi hai chi tiet cua cung mot canh thuong lech nhau vai
        # giay ("xe buyt vua chay vuot qua" xay ra sau).
        if prev_frames:
            import numpy as _np

            w = float(request.refine_window_sec)
            store = svc["store"]
            keep = set()
            by_v = {}
            for vid, t in prev_frames:
                by_v.setdefault(vid, []).append(t)
            for vid, ts in by_v.items():
                df = store.frames_of(vid)
                if df.empty:
                    continue
                pt = df["pts_time"].to_numpy(dtype=float)
                gi = df["gidx"].to_numpy(dtype=int)
                arr = _np.asarray(ts, dtype=float)
                near = (_np.abs(pt[:, None] - arr[None, :]) <= w).any(axis=1)
                keep.update(int(g) for g in gi[near])
            restrict_gidx = keep or None

    result = svc["engine"].search(
        query_en=dq.query_en,
        query_vi=dq.query_vi,
        video_topn=request.video_topn,
        frame_topk=request.k,
        weights=request.weights,
        channels=request.channels,
        ignore_gidx=ignore_gidx,
        kind=dq.kind,          # quyet dinh trong so giua thi giac va van ban
        modes=request.channel_modes,   # cong tac ASR/OCR do nguoi dung gat
        restrict_videos=restrict,      # "Trong ket qua cu" -- cap video
        restrict_gidx=restrict_gidx,   # ... va cap FRAME
    )

    # --- Thu hep theo HUONG thoi gian --------------------------------------
    # filtervideo: 0 khong loc | 1 chi giu frame SAU moc cu | 2 chi giu frame TRUOC.
    # Dung khi da chot dung video va biet canh can tim nam ve phia nao so voi
    # canh dang thay.
    if restrict and request.filtervideo in (1, 2) and prev_time:
        after = request.filtervideo == 1
        kept = []
        for v in result.get("videos", []):
            t0 = prev_time.get(v["video_id"])
            vi = v["video_info"]
            if t0 is None:
                kept.append(v)
                continue
            keep_i = [i for i, t in enumerate(vi.get("lst_pts_times") or [])
                      if t is not None and ((t > t0) if after else (t < t0))]
            if not keep_i:
                continue
            for key, arr in list(vi.items()):
                if isinstance(arr, list) and len(arr) >= max(keep_i) + 1:
                    vi[key] = [arr[i] for i in keep_i]
            kept.append(v)
        result["videos"] = kept
        result["loc_huong"] = ("chi frame SAU moc cu" if after
                               else "chi frame TRUOC moc cu")

    if restrict:
        result["loc_trong_ket_qua_cu"] = {
            "n_video": len(restrict),
            "n_frame_duoc_phep": len(restrict_gidx or ()),
            "cua_so_giay": request.refine_window_sec,
        }

    # Tra thang ket qua cua engine. `videos` da dung dang FE doc duoc
    # ({video_id, video_info:{lst_*}}) va con mang theo `explain` -- thu hang cua
    # video o tung kenh, tuc la LY DO no noi len. Ep ve mang thuan se vut mat
    # phan do, ma day chinh la cach nguoi dung phan biet cac video trong giong
    # het nhau.
    # --- dinh tuyen theo LOAI truy van -----------------------------------
    # Khong phai cu nhieu menh de la dong hang. Tieu chi la DO PHAN BIET:
    #
    #   anchored       co chi tiet hiem ("bang London Zoo", "doan Nhan Nghia
    #                  Duong") -- mot menh de da chot duoc video, DP chi ton them
    #                  thoi gian.
    #   generic_chain  moi menh de deu la canh pho bien ("dung duoi nuoc roi
    #                  den", "keo luoi ca") -- rieng le thi hang nghin video
    #                  khop, chi THU TU + khoang cach THOI GIAN moi phan biet
    #                  duoc. Day la luc DP an thua.
    # Nguoi dung ngoi truoc man hinh va nhin thay ket qua, bo phan loai thi
    # khong -- nen `align` do ho dat len se de len tren moi phan doan cua may.
    cfg = svc["config"]
    if request.align is True:
        want_dp = len(dq.clauses_en) >= 2      # van can >=2 menh de moi dong hang duoc
        dp_decided_by = "nguoi_dung"
    elif request.align is False:
        want_dp = False
        dp_decided_by = "nguoi_dung"
    else:
        want_dp = dq.needs_dp
        dp_decided_by = dq.kind_source

    dp_ran = False
    dp_skip_reason = None
    if not want_dp:
        dp_skip_reason = (
            "nguoi dung tat" if request.align is False
            else f"loai truy van la {dq.kind!r}" if dq.kind == "anchored"
            else "chi co mot menh de, khong co gi de dong hang")
    if cfg.dp_enabled and want_dp:
        try:
            svc["store"].assert_encoder_matches(svc["engine"].encoder)
            top_ids = [v["video_id"] for v in result.get("videos", [])]
            aligned = svc["engine"].align_videos(
                dq.clauses_en, top_ids[:cfg.dp_video_topn])
            if aligned:
                order = {a["video_id"]: i for i, a in enumerate(aligned)}
                info = {a["video_id"]: a for a in aligned}
                for v in result["videos"]:
                    a = info.get(v["video_id"])
                    if a:
                        v["dp_score"] = round(a["dp_score"], 4)
                        v["matched"] = a["matched"]
                        v["n_skipped"] = a["n_skipped"]
                # Xep lai theo diem dong hang; video khong dong hang duoc xuong cuoi.
                result["videos"].sort(
                    key=lambda v: order.get(v["video_id"], 1 << 30))
                dp_ran = True
        except Exception as e:
            # DP hong khong duoc lam sap truy van -- ket qua tim phang van dung.
            result.setdefault("errors", {})["align"] = f"{type(e).__name__}: {e}"

    if not cfg.dp_enabled and want_dp:
        dp_skip_reason = "dp_enabled=false trong config"

    result["mode"] = "fusion+align" if dp_ran else "fusion"
    result["aligned"] = dp_ran
    result["align_decided_by"] = dp_decided_by
    if not dp_ran and dp_skip_reason:
        result["align_skipped"] = dp_skip_reason
    result["query"] = dq.as_dict()
    result["query_en"] = dq.query_en
    result["query_vi"] = dq.query_vi
    return result

# ---------------------------------------------------------------------------
# Bon duong tim kiem phu cua UI. Truoc day deu 404, UI bao "Fetch failed".
# ---------------------------------------------------------------------------
@app.get("/data")
def data():
    """Du lieu khoi tao cho UI: tu vung lop object, danh sach video, trang thai kenh.

    UI truoc day go tag object tu mot danh sach COCO chep cung trong
    frontend/src/helper/icons.js ("person", "dog", ...). Du lieu object cua BTC
    lai la OpenImages viet hoa ("Person", "Dog") va co 545 lop -- go tag theo
    danh sach cu thi phan lon khong khop gi, ma UI khong bao gi ca. Lay tu vung
    THAT tu day.
    """
    svc = get_fusion()
    if svc is None:
        return {"error": _fusion_state["error"], "status_code": 503}
    ix = svc["objects"]
    return {
        "ok": True,
        "objects": [{"ten": e, "n": n} for e, n in (ix.vocabulary() if ix.ok else [])],
        "object_index": ix.status(),
        "videos": svc["store"].video_ids,
        "n_video": len(svc["store"].video_ids),
        "n_keyframe": int(len(svc["store"].meta)),
        "kenh_van_ban": list(svc["channels"].index),
        "che_do_kenh": {"gia_tri": ["auto", "on", "off"],
                        "kenh": ["siglip", "meta", "asr", "ocr"]},
    }


@app.post("/panel")
def panel(request: PanelSearchRequest):
    """Tim theo LOP object + VI TRI tren khung hinh + chu tren hinh + loi noi."""
    svc = get_fusion()
    if svc is None:
        return {"videos": [], "warnings": [_fusion_state["error"]], "status_code": 503}

    videos, warnings = svc["panels"].panel(
        tags=request.tags or [],
        drag_objects=request.dragObject or [],
        amount=request.amount,
        ocr=request.ocr,
        asr=request.asr,
        ids=request.id,
        useid=bool(request.useid),
        ignore_idxs=(request.ignore_idxs if request.ignore else None),
        k=request.k,
    )
    return {"videos": videos, "warnings": warnings,
            "n_video": len(videos),
            "n_frame": sum(len(v["video_info"]["lst_idxs"]) for v in videos)}


@app.get("/framerange")
def framerange(video_id: str, start: int, end: int, text_query: str = "",
               limit: int = 200):
    """Keyframe cua mot video trong dai [start, end] tinh theo frame_idx."""
    svc = get_fusion()
    if svc is None:
        return {"error": _fusion_state["error"], "status_code": 503}
    return svc["panels"].frame_range(video_id, start, end, text_query, limit)


@app.get("/imgsearch")
def imgsearch(imgid: str, k: int = 200, per_video: int = 0):
    """KNN tu vector cua mot keyframe. imgid la khoa 'video_id#frame_idx'."""
    svc = get_fusion()
    if svc is None:
        return []
    videos, err = svc["panels"].img_search(imgid, k=k, per_video=per_video)
    if err:
        return {"error": err, "status_code": 404}
    # UI cu doc thang mang -> giu nguyen dang mang.
    return videos


@app.get("/relatedimg")
def relatedimg(imgid: str = "", span: int = 6):
    """Boi canh cua anh dang xem toan man hinh: frame truoc/sau + link video goc."""
    svc = get_fusion()
    if svc is None or not imgid:
        return {}
    return svc["panels"].related(imgid, span=span)


@app.post("/getrec")
def getrec():
    """Goi y tag: da chuyen sang GET /data (tu vung lop that cua du lieu object)."""
    return []

@app.post("/feedback")
def feedback(request: FeedbackRequest):
    """Tim lai bang chinh cac ANH da cham dung/sai (phan hoi lien quan).

    Truoc day day la mot stub `return []` -- nut "Gui & tim lai" luon cho ra
    "khong co ket qua nao khop", ma nguoi dung khong the biet la chua ai lam.

    Cach lam (Rocchio, dang don gian nhat co the giai thich duoc):

        q_moi = trung binh(vector cac frame DUNG) - beta * trung binh(cac frame SAI)

    Roi tim bang chinh q_moi do tren toan corpus. Khong con truy van chu nao ca
    -- day la tim bang HINH ANH, nen no bat duoc nhung thu ma cau chu khong ta
    noi (mot kieu bo cuc, mot kieu do hoa ban tin).

    beta nho hon 1 co y: anh SAI chi de day ket qua ra xa, khong duoc lan at
    huong ma anh DUNG da chi ra.
    """
    svc = get_fusion()
    if svc is None:
        return []

    import numpy as np

    store = svc["store"]
    from retrieval.engine import entry_key, parse_entry_key

    look = {}
    for v, f, g in zip(store.meta["video_id"], store.meta["frame_idx"],
                       store.meta["gidx"]):
        look[entry_key(v, f)] = int(g)

    def vectors(keys):
        idx = []
        for it in (keys or []):
            k = it if isinstance(it, str) else str(it)
            g = look.get(k)
            if g is None and parse_entry_key(k):
                g = look.get(entry_key(*parse_entry_key(k)))
            if g is not None:
                idx.append(g)
        return store.X[idx] if idx else None

    pos = vectors(request.lst_pos_idxs)
    neg = vectors(request.lst_neg_idxs)
    if pos is None and neg is None:
        return []

    BETA = 0.35
    q = np.zeros(store.X.shape[1], dtype=np.float32)
    if pos is not None:
        q += pos.mean(axis=0)
    if neg is not None:
        q -= BETA * neg.mean(axis=0)
    n = float(np.linalg.norm(q))
    if n < 1e-9:
        return []
    q = q / n

    sims = store.X @ q
    topk = min(int(request.k or 200), sims.shape[0])
    part = np.argpartition(-sims, topk - 1)[:topk]
    part = part[np.argsort(-sims[part])]

    # Bo chinh nhung anh vua cham -- chung da nam trong tay nguoi dung roi.
    seen = set(request.lst_pos_idxs or []) | set(request.lst_neg_idxs or [])
    rows = []
    for g in part:
        r = store.meta.iloc[int(g)]
        key = entry_key(str(r["video_id"]), int(r["frame_idx"]))
        if key in seen:
            continue
        rows.append({"video_id": str(r["video_id"]),
                     "frame_idx": int(r["frame_idx"]),
                     "pts_time": float(r["pts_time"]),
                     "score": float(sims[int(g)])})

    from retrieval.panels import pack_videos

    order = {}
    for r in rows:
        order.setdefault(r["video_id"], len(order))
    return pack_videos(rows, order=order)

@app.post("/translate")
def translate(request: dict):
    """Dich truy van sang tieng Anh cho o hien thi tren UI.

    Tra lai nguyen van tieng Viet nhu truoc thi o dich tren UI vo nghia. Dung
    chung ham dich da co kiem chung cua retrieval.query: no tra ve rong khi
    dich vu tra ve trang loi, thay vi do nguyen trang loi ra man hinh.
    """
    from retrieval.query import _translate

    text = str(request.get("textquery") or "").strip()
    if not text:
        return ""
    return _translate(text) or ""

@app.get("/getvideoshot")
def getvideoshot(imgid: str, block_sec: float = 60.0):
    """Toan bo keyframe cua video chua `imgid`, nhom theo KHOI THOI GIAN.

    Truoc day day la stub tra ve rong -- trang "Ca video" mo ra chi thay
    "Collection:  Video id:" trong tron, va nguoi dung tuong trang bi treo.

    Khong nhom theo "shot" vi he nay KHONG co ranh gioi canh: bo keyframe duoc
    lay mau theo do troi ngu nghia, khong theo cat canh. Cot cos_to_prev cung
    khong dung duoc -- phan bo cua no da bi chinh nguong lay mau cat cut. Nhom
    theo tung phut la cach chia trung thuc va van duyet duoc bang mat.
    """
    svc = get_fusion()
    if svc is None:
        return {"error": _fusion_state["error"], "collection": "", "video_id": "",
                "video_name": "", "shots": {}, "selected_shot": "0"}

    from retrieval.engine import entry_key, keyframe_url, parse_entry_key

    store = svc["store"]
    p = parse_entry_key(imgid)
    if p:
        video_id, sel_frame = p
    else:
        # Client cu gui chi so toan cuc, hoac chi gui ten video.
        video_id, sel_frame = str(imgid), None
        try:
            g = int(imgid)
            r = store.meta.iloc[g]
            video_id, sel_frame = str(r["video_id"]), int(r["frame_idx"])
        except (ValueError, IndexError):
            pass

    df = store.frames_of(video_id)
    if df.empty:
        return {"error": f"Video {video_id!r} khong co trong index",
                "collection": "", "video_id": video_id, "video_name": video_id,
                "shots": {}, "selected_shot": "0"}

    # frame_idx duoc hoi co the KHONG phai keyframe cua ta (dap an cua BTC danh
    # so theo he cua ho). Lay keyframe gan nhat de van danh dau dung cho.
    if sel_frame is not None:
        j = (df["frame_idx"] - sel_frame).abs().idxmin()
        sel_frame = int(df.loc[j, "frame_idx"])

    shots, selected = {}, "0"
    for r in df.to_dict("records"):
        fi, pts = int(r["frame_idx"]), float(r["pts_time"])
        b = int(pts // block_sec)
        lo, hi = int(b * block_sec), int((b + 1) * block_sec)
        label = f"{lo // 60:02d}:{lo % 60:02d}-{hi // 60:02d}:{hi % 60:02d}"
        sh = shots.setdefault(label, {"lst_keyframe_paths": [], "lst_idxs": [],
                                      "lst_keyframe_idxs": [], "lst_pts_times": []})
        sh["lst_keyframe_paths"].append(keyframe_url(video_id, fi))
        sh["lst_idxs"].append(entry_key(video_id, fi))
        sh["lst_keyframe_idxs"].append(fi)
        sh["lst_pts_times"].append(round(pts, 2))
        if sel_frame is not None and fi == sel_frame:
            selected = label

    return {
        "collection": video_id.split("_")[0],
        "video_id": video_id,
        "video_name": video_id,
        "n_keyframe": int(len(df)),
        "shots": shots,
        "selected_shot": selected,
    }
# --------------------------------------------------


@app.post("/keyframe_context")
def keyframe_context(request: KeyframeContextRequest):
    """Object, OCR, ASR quanh mot keyframe."""
    svc = get_fusion()
    if svc is None:
        return {"error": _fusion_state["error"], "status_code": 503}
    return svc["bridge"].context_at(request.video_id, request.frame_idx)

@app.post("/autofill")
def autofill_endpoint(request: AutofillRequest):
    svc = get_fusion()
    if svc is None:
        return {"error": _fusion_state["error"], "status_code": 503}
    from retrieval.autofill import autofill

    # request.manual la list AnswerEntry (pydantic), khong phai dict --
    # autofill() doc bang .get() nen phai model_dump() truoc.
    manual = [m.model_dump() for m in request.manual]

    # Mo rong tap ung vien vuot ra ngoai luoi dang hien.
    #
    # Luoi tren man hinh co k anh (mac dinh 500) vi day la thu nguoi dung NHIN
    # duoc. Bai nop lai co 100 o va cham theo thu hang -- khong co ly do gi de
    # 100 o do bi gioi han boi so anh hien tren man hinh. Do duoc: pool 500 ->
    # Final 0.6000, pool 3000 -> 0.6286.
    cands = list(request.candidates or [])
    pool_note = None
    q_vi = request.query_vi or ""
    q_en = request.query_en or ""
    if q_vi.strip() or q_en.strip():
        try:
            from retrieval.query import decompose

            dq = decompose(query_vi=q_vi, query_en=q_en, use_llm=False,
                           kind=request.kind)
            pool_k = request.pool_topk or getattr(
                svc["config"], "autofill_pool_topk", 3000)
            wide = svc["engine"].search(
                query_en=dq.query_en, query_vi=dq.query_vi,
                frame_topk=pool_k, kind=dq.kind)
            extra = []
            for v in wide.get("videos", []):
                vi = v["video_info"]
                for j, fi in enumerate(vi["lst_keyframe_idxs"]):
                    extra.append({"video_id": v["video_id"], "frame_idx": int(fi),
                                  "score": (vi.get("lst_scores") or [None])[j]})
            # Ung vien tren man hinh giu nguyen thu tu o DAU; phan mo rong noi sau.
            have = {(c.get("video_id"), int(c.get("frame_idx")))
                    for c in cands if c.get("frame_idx") is not None}
            n_before = len(cands)
            cands += [e for e in extra if (e["video_id"], e["frame_idx"]) not in have]
            pool_note = (f"mo rong tap ung vien tu {n_before} len {len(cands)} "
                         f"(ngoai luoi dang hien)")
        except Exception as e:
            pool_note = f"khong mo rong duoc tap ung vien: {type(e).__name__}: {e}"

    out = autofill(
        svc["store"], manual, cands,
        config=svc["config"], ignore=request.ignore, target=request.target,
        head=request.head, tail_gap=request.tail_gap_sec,
    )
    n_manual = sum(1 for e in out if e["source"] == "manual")
    from retrieval.autofill import slot_weight

    return {
        "answers": out,
        "n": len(out),
        "n_manual": n_manual,
        "n_autofill": len(out) - n_manual,
        "n_ung_vien": len(cands),
        "ghi_chu": pool_note,
        # De UI hien duoc gia tri cua tung o theo cong thuc cham cua BTC.
        "gia_tri_o": {"1": slot_weight(1), "2-5": slot_weight(2),
                      "6-20": slot_weight(6), "21-50": slot_weight(21),
                      "51-100": slot_weight(51)},
    }

@app.get("/diagnostics")
def diagnostics():
    """Tự kiểm tra sức khoẻ hệ thống (Fusion tier only)."""
    if get_fusion() is not None:
        artifacts = _fusion_state["module"].diagnostics() if "module" in _fusion_state else service.diagnostics()
    else:
        artifacts = {"error": _fusion_state.get("error", "Unknown")}

    return {
        "ok": get_fusion() is not None,
        "artifacts": artifacts,
    }

@app.post("/trake")
def trake_endpoint(request: TrakeRequest):
    """Dong hang chuoi su kien bang DP tren truc thoi gian.

    Hai duong vao:
      video_id co   -> dong hang tren dung video do (duong cu, giu nguyen).
      video_id rong -> TU TIM video ung vien bang tang hop nhat roi dong hang
                       tren tung video, tra ve xep hang. Day moi la dang bai
                       TRAKE that: truy van -> video + day frame. Bat nguoi
                       dung phai biet truoc video thi endpoint gan nhu vo dung.
    """
    svc = get_fusion()
    if svc is None:
        return {"error": _fusion_state["error"], "status_code": 503}

    cfg = svc["config"]
    engine = svc["engine"]
    encoder = engine.encoder
    try:
        svc["store"].assert_encoder_matches(encoder)
    except Exception as e:
        # Kenh thi giac la bat buoc voi TRAKE -- khong co encoder thi khong the
        # cham diem su kien, va doan bua se cho ra chuoi frame vo nghia.
        return {"error": f"TRAKE can SigLIP encoder: {e}", "status_code": 503}

    # --- lay danh sach su kien ------------------------------------------
    events = [e.strip() for e in (request.events or []) if e and e.strip()]
    dq = None
    if not events:
        import re as _re

        from retrieval.query import _translate, decompose

        raw = request.query_vi or request.query_en
        if not raw.strip():
            return {"error": "Can `events` hoac `query_vi`", "status_code": 400}

        # BTC danh so su kien NGAY TRONG cau: "(1) ... (2) ... (3) ...".
        # Cat theo dung cac moc do truoc -- khong phai doan, va khong phu thuoc
        # vao LLM. decompose() cat theo dau cau nen gop het thanh MOT menh de,
        # va TRAKE mot menh de thi khong con la TRAKE.
        parts = _re.split(r"\(\s*\d+\s*\)", raw)
        numbered = [p.strip(" ,.;:\n") for p in parts[1:] if p.strip(" ,.;:\n")]
        if len(numbered) >= 2:
            events = [(_translate(e) or e) for e in numbered]
        else:
            dq = decompose(query_vi=request.query_vi, query_en=request.query_en,
                           use_llm=request.decompose, kind="generic_chain")
            events = [c for c in dq.clauses_en if c.strip()]
        if not events:
            return {"error": ("khong tach duoc menh de nao tu truy van -- nhap "
                              "thang danh sach `events` bang tieng Anh"),
                    "status_code": 400}

    delta = cfg.dp_delta_sec if request.delta is None else request.delta
    gamma = cfg.dp_gamma if request.gamma is None else request.gamma
    min_gap = (getattr(cfg, "dp_min_gap_sec", 0.0)
               if request.min_gap is None else request.min_gap)
    if min_gap > delta:
        return {"error": f"min_gap ({min_gap}) khong duoc lon hon delta ({delta})",
                "status_code": 400}

    # --- chon tap video ---------------------------------------------------
    if request.video_id:
        if request.video_id not in svc["store"].video_slice:
            return {"error": f"Video {request.video_id} khong co trong index",
                    "status_code": 404}
        video_ids = [request.video_id]
        chon_boi = "nguoi_dung"
    else:
        query_en = (dq.query_en if dq else request.query_en) or "\n".join(events)
        query_vi = (dq.query_vi if dq else request.query_vi)
        fused, _, _ = engine.rank_videos(query_en, query_vi, kind="generic_chain")
        video_ids = [v for v, _ in fused if v in svc["store"].video_slice
                     ][:max(1, request.video_topn)]
        chon_boi = "tang_hop_nhat"
        if not video_ids:
            return {"message": "Khong tim thay video ung vien nao.",
                    "events": events, "status_code": 404}

    aligned = engine.align_videos(events, video_ids, delta=delta, gamma=gamma,
                                  min_gap=min_gap, normalize=request.normalize,
                                  n_candidates=request.n_candidates)
    if not aligned:
        return {"message": "Khong dong hang duoc chuoi su kien tren video nao.",
                "events": events, "status_code": 404}

    for a in aligned:
        a["dp_score"] = round(a["dp_score"], 4)
        # Day frame de nop bai TRAKE, theo dung thu tu su kien.
        a["matched_frames"] = [m["frame_idx"] for m in a["matched"]]
        a["idxs"] = [None if m["frame_idx"] is None
                     else f"{a['video_id']}#{m['frame_idx']}" for m in a["matched"]]
        a["keyframe_paths"] = [None if m["frame_idx"] is None
                               else f"/keyframe/{a['video_id']}/{int(m['frame_idx']):06d}.jpg"
                               for m in a["matched"]]
        for ev in a.get("candidates") or []:
            for c in ev:
                c["path"] = (f"/keyframe/{a['video_id']}/"
                             f"{int(c['frame_idx']):06d}.jpg")
                c["id"] = f"{a['video_id']}#{c['frame_idx']}"

    best = aligned[0]
    return {
        # khoa cu, cho client da viet theo mot video
        "video_id": best["video_id"],
        "score": best["dp_score"],
        "matched": best["matched"],
        "matched_frames": best["matched_frames"],
        "n_skipped": best["n_skipped"],
        # xep hang day du
        "events": events,
        "videos": aligned,
        "n_video": len(aligned),
        "chon_video_boi": chon_boi,
        "tham_so": {"delta": delta, "gamma": gamma, "min_gap": min_gap,
                    "chuan_hoa_z": bool(request.normalize)},
        "ghi_chu": ("gamma tinh bang do lech chuan cua chinh su kien do tren "
                    "corpus, khong phai don vi cosine"
                    if request.normalize else
                    "chuan_hoa_z=false: diem la cosine tho, gamma gan nhu chac "
                    "chan qua lon nen DP se khong bao gio bo qua su kien nao"),
    }


@app.post("/qa")
def qa_endpoint(request: QaRequest):
    """Q/A KHONG chay o day -- endpoint that nam o socket_app.py (local).

    Ket luan tu chinh ghi chu trong ban truoc: VLM can ANH that de tra loi, ma
    anh chi trich duoc tu data/videos -- thu muc 65GB chi co o may local, Kaggle
    khong mount. Goi VLM o day thi luon phai truyen image=None, tuc la hoi mot
    model thi giac ma khong dua anh nao.

    Giu endpoint nay de client cu goi nham con nhan duoc chi dan ro rang.
    """
    return {
        "error": "Q/A chay o server local, khong phai Kaggle.",
        "goi_thay_vao": "POST {socket_url}/qa (socket_app.py)",
        "ly_do": ("VLM can anh that; anh trich tu data/videos chi co o may local. "
                  "Kaggle chi mount data/artifacts (features/keyframes/asr/ocr)."),
        "status_code": 501,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
