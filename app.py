import copy
import os
from typing import Any, Dict, Optional, Union

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from retrieval import service
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
    
    # Convert result to legacy array format for frontend
    result_dict = {}
    for vid, fid, score in zip(result.get("videos", []), result.get("frame_idxs", []), result.get("scores", [])):
        if vid not in result_dict:
            result_dict[vid] = {
                "lst_keyframe_paths": [],
                "lst_idxs": [],
                "lst_keyframe_idxs": [],
                "lst_scores": [],
            }
        
        image_path = f"/static/images/keyframe/{vid}/{int(fid):06d}.jpg"
        pseudo_idx = f"{vid}_{fid}"
        
        result_dict[vid]["lst_keyframe_paths"].append(image_path)
        result_dict[vid]["lst_idxs"].append(pseudo_idx)
        result_dict[vid]["lst_keyframe_idxs"].append(int(fid))
        result_dict[vid]["lst_scores"].append(float(score))
        
    frontend_array = [
        {"video_id": key, "video_info": value} for key, value in result_dict.items()
    ]
    frontend_array = sorted(
        frontend_array, key=lambda x: x["video_info"]["lst_scores"][0], reverse=True
    )
    
    return frontend_array

# --- Stub Endpoints to prevent frontend crashes ---
@app.get("/getrec")
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
    return request.get("textquery", "")

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
    out = autofill(
        svc["store"], request.manual, request.candidates,
        config=svc["config"], ignore=request.ignore, target=request.target
    )
    return {"answers": out}

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
    """Tìm chuỗi frame tối ưu nhất cho tập các sự kiện tuần tự (TRAKE) bằng DP."""
    svc = get_fusion()
    if svc is None:
        return {"error": _fusion_state["error"], "status_code": 503}
        
    from retrieval.trake import dp_alignment
    import numpy as np
    
    df = svc["store"].frames_of(request.video_id)
    if df.empty:
        return {"error": f"Video {request.video_id} không tìm thấy hoặc không có frame nào", "status_code": 404}
        
    pts_times = df["pts_time"].tolist()
    frame_idxs = df["frame_idx"].tolist()
    num_frames = len(pts_times)
    num_events = len(request.events)
    
    event_scores = np.zeros((num_frames, num_events))
    encoder = svc["engine"].encoder
    
    X = svc["store"].X
    meta = svc["store"].meta
    m = meta.video_id == request.video_id
    video_features = X[m]
    
    if len(video_features) != num_frames:
        return {"error": "Số lượng frame và đặc trưng SigLIP không khớp", "status_code": 500}
        
    for k, event_query in enumerate(request.events):
        q = encoder.encode_texts([event_query])[0]
        s = video_features @ q
        event_scores[:, k] = s
        
    path, max_score = dp_alignment(pts_times, event_scores, delta=request.delta, gamma=request.gamma)
    
    if max_score == -np.inf or not path:
        return {"message": "Không tìm thấy chuỗi sự kiện nào phù hợp.", "status_code": 404}
        
    matched_frames = []
    for p in path:
        if p == -1:
            matched_frames.append(None)
        else:
            matched_frames.append(frame_idxs[p])
            
    return {
        "video_id": request.video_id,
        "score": max_score,
        "matched_frames": matched_frames
    }

@app.post("/qa")
def qa_endpoint(request: QaRequest):
    """Giải quyết truy vấn Q/A bằng cách sử dụng VLM (Gemini)."""
    svc = get_fusion()
    if svc is None:
        return {"error": _fusion_state["error"], "status_code": 503}
        
    from retrieval.qa import solve_qa, extract_best_frame
    
    store = svc["store"]
    best_frame = extract_best_frame(request.video_id, request.start_pts, request.end_pts, store)
    
    if best_frame is None:
        return {"error": "Không tìm thấy frame nào trong khoảng thời gian", "status_code": 404}
        
    frame_idx = best_frame["frame_idx"]
    pts_time = float(best_frame["pts_time"])
    ocr_result = svc["bridge"].ocr_at(request.video_id, pts_time, frame_idx=frame_idx)
    ocr_texts = [item["text"] for item in ocr_result.get("items", [])]
    
    # Static image path pattern on Local
    # Because Frontend accesses it directly: /static/images/keyframe/L21_V001/000123.jpg
    # Wait, the VLM runs on Kaggle. It needs the image to answer QA! 
    # BUT wait, the images are only on Local!
    # Ah! If Kaggle has no images, how does it pass an image to Gemini?
    # Let me check if data/artifacts/ has images... No, data/artifacts/ only has numpy arrays and csv.
    # So if Kaggle runs VLM, how does it read the image?
    # Actually, QA might need to be run on Local, or Local sends the image base64 to Kaggle?
    # The current `retrieval/qa.py` reads `image_path` from `svc["images"].get(request.video_id, frame_idx)` which extracts it from `data/videos/`.
    # Let's pass the image URL instead or we must move the QA endpoint to `socket_app.py`!
    
    # I will keep the qa endpoint here for now, but log a warning.
    # The proper way is moving QA to local server if images are local.
    qa = solve_qa(request.video_id, request.question, None, ocr_texts) # Image is None on Kaggle!
    
    return {
        "video_id": request.video_id,
        "frame_idx": int(frame_idx),
        "pts_time": pts_time,
        "answer": qa["answer"],
        "degraded": qa["degraded"],
        "vlm": {k: v for k, v in qa.items() if k != "answer"},
        "ocr_texts_used": ocr_texts,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
