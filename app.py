import os

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from retrieval import service
from utils.auth import install_auth
from utils.logger_config import get_logger
from utils.models import (
    AutofillRequest,
    KeyframeContextRequest,
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
install_auth(app)

# Eagerly initialize fusion engine on startup to fail-fast
# Kaggle stateless backend only relies on this.
try:
    _fusion_state = service.get()
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

# --- Stub Endpoints to prevent frontend crashes ---
@app.post("/getrec")
def getrec():
    return []

@app.get("/relatedimg")
def relatedimg():
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
    """Dong hang chuoi su kien bang DP tren truc thoi gian cua mot video."""
    svc = get_fusion()
    if svc is None:
        return {"error": _fusion_state["error"], "status_code": 503}

    from retrieval.trake import dp_alignment, events_to_scores

    if not request.events:
        return {"error": "Cần ít nhất một sự kiện", "status_code": 400}
    if request.video_id not in svc["store"].video_slice:
        return {"error": f"Video {request.video_id} không có trong index",
                "status_code": 404}

    encoder = svc["engine"].encoder
    try:
        svc["store"].assert_encoder_matches(encoder)
    except Exception as e:
        # Kenh thi giac la bat buoc voi TRAKE -- khong co encoder thi khong the
        # cham diem su kien, va doan bua se cho ra chuoi frame vo nghia.
        return {"error": f"TRAKE cần SigLIP encoder: {e}", "status_code": 503}

    pts_times, frame_idxs, scores = events_to_scores(
        svc["store"], request.video_id, encoder, request.events)
    if not pts_times:
        return {"error": f"Video {request.video_id} không có keyframe nào",
                "status_code": 404}

    path, score = dp_alignment(pts_times, scores,
                               delta=request.delta, gamma=request.gamma)
    if not path:
        return {"message": "Không tìm thấy chuỗi sự kiện nào phù hợp.",
                "status_code": 404}

    matched = []
    for k, p in enumerate(path):
        if p == -1:
            matched.append({"event": k, "frame_idx": None, "pts_time": None,
                            "skipped": True})
        else:
            matched.append({"event": k, "frame_idx": int(frame_idxs[p]),
                            "pts_time": round(float(pts_times[p]), 3),
                            "score": round(float(scores[p, k]), 4),
                            "skipped": False})

    return {
        "video_id": request.video_id,
        "score": float(score),
        "events": request.events,
        "matched": matched,
        # giu khoa cu cho client da viet theo dinh dang truoc
        "matched_frames": [m["frame_idx"] for m in matched],
        "n_skipped": sum(1 for m in matched if m["skipped"]),
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
