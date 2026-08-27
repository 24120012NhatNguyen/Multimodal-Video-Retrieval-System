import os

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from retrieval import service
from utils.logger_config import get_logger
from utils.models import (
    AutofillRequest,
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
    )

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
def feedback():
    return []

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
def getvideoshot(imgid: str):
    return {
        "collection": "",
        "video_id": "",
        "video_name": "",
        "shots": {},
        "selected_shot": "0",
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
    out = autofill(
        svc["store"], manual, request.candidates,
        config=svc["config"], ignore=request.ignore, target=request.target
    )
    n_manual = sum(1 for e in out if e["source"] == "manual")
    return {
        "answers": out,
        "n": len(out),
        "n_manual": n_manual,
        "n_autofill": len(out) - n_manual,
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
        from retrieval.query import decompose

        raw = request.query_vi or request.query_en
        if not raw.strip():
            return {"error": "Can `events` hoac `query_vi`", "status_code": 400}
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
                                  min_gap=min_gap, normalize=request.normalize)
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
