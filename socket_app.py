"""Server LOCAL: dap an, anh keyframe, Q/A. Khong chay tren Kaggle.

Ranh gioi kien truc theo checklist:

  A3 - moi thu co TRANG THAI chay local. Kaggle session chet la mat sach dap an
       nguoi dung dang go do.
  A2 - trich anh tang tinh chay local, Kaggle chi tra pts_time. Kaggle chi mount
       artifacts (features/keyframes/asr/ocr), KHONG co data/videos (65GB), nen
       khong the trich anh o do.

Vi vay phan chia la:

  Kaggle  app.py :8080        tim kiem (SigLIP + BM25 + RRF), stateless
  Local   socket_app.py :8081 dap an + anh keyframe + Q/A (can anh that)

Dap an luu theo schema Viec 2, LIST CO THU TU:

    [{"video_id": "L24_V007", "frame_idx": 12450, "source": "manual"}, ...]

`source` phan biet nguoi chon voi may lap, de auto-fill khong day phan doan cua
nguoi dung xuong duoi.
"""

import io
import json
import os
import zipfile

import socketio
import uvicorn
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from retrieval.answers import (
    kis_csv,
    make,
    normalize,
    qa_csv,
    reorder as reorder_answers,
    trake_csv,
)
from utils.auth import install_auth
from utils.logger_config import get_logger
from utils.models import QaRequest, QuestionNameRequest, UsernameRequest, UserRequest

logger = get_logger(__name__)

app = FastAPI()

ALLOWED_ORIGINS = [
    "*",
    "https://*.ngrok-free.app",
    "https://*.ngrok-free.dev",
    "https://*.trycloudflare.com",
    "http://localhost:3000",
    "http://localhost",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)
install_auth(app)

sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins=ALLOWED_ORIGINS,
    ping_timeout=60,
    ping_interval=25,
)

###################### Trang thai ##############################
back_up_folder = "back_up"
os.makedirs(back_up_folder, exist_ok=True)


def _load(name, default):
    path = f"{back_up_folder}/{name}.json"
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("khong doc duoc %s: %s", path, e)
        return default


def _store(name, obj):
    path = f"{back_up_folder}/{name}.json"
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


# normalize() doc duoc ca dinh dang cu ({"video","frames","idxs"}) lan moi.
AnswerDict = {k: normalize(v) for k, v in _load("answer", {}).items()}
AnswerIgnoreDict = _load("answer_ignore", {})
UserDict = _load("user", {})
ReorderStatus = _load("reorder_status", {})


def store_answer():
    _store("answer", AnswerDict)


def store_ignore():
    _store("answer_ignore", AnswerIgnoreDict)


def store_user():
    _store("user", UserDict)


def store_status():
    _store("reorder_status", ReorderStatus)


####################### Tang anh (local) #######################
# Nap lazy: khong co data/videos thi cac phan con lai van chay binh thuong.
_media = {"svc": None, "error": None}


def get_media():
    if _media["svc"] is None and _media["error"] is None:
        try:
            from retrieval.frames import KeyframeImages
            from retrieval.store import ArtifactStore

            store = ArtifactStore()
            _media["store"] = store
            _media["svc"] = KeyframeImages(store)
            logger.info("tang anh local: %s", store)
        except Exception as e:
            _media["error"] = f"{type(e).__name__}: {e}"
            logger.exception("khong nap duoc tang anh local")
    return _media["svc"]


####################### Helper #################################
def entry_key(video_id, frame_idx):
    """Khoa on dinh cho mot dong dap an.

    He thong cu dung chi so keyframe toan cuc trong dict/id2img.json; thu muc do
    da bi xoa nen khoa gio la cap (video_id, frame_idx) ghep lai.
    """
    return f"{video_id}#{int(frame_idx)}"


def parse_key(key):
    """'L24_V007#12450' -> ('L24_V007', 12450). None neu khong parse duoc."""
    if not isinstance(key, str) or "#" not in key:
        return None
    vid, _, fi = key.rpartition("#")
    try:
        return vid, int(fi)
    except ValueError:
        return None


def entry_from_payload(data):
    """Doc (video_id, frame_idx) tu payload FE, chap nhan nhieu cach goi ten."""
    if not isinstance(data, dict):
        parsed = parse_key(data)
        return parsed
    vid = data.get("video_id") or data.get("video")
    fi = data.get("frame_idx")
    if fi is None and isinstance(data.get("frames"), list) and data["frames"]:
        fi = data["frames"][0]
    if vid is None or fi is None:
        return parse_key(data.get("idx") or data.get("key"))
    try:
        return str(vid), int(fi)
    except (TypeError, ValueError):
        return None


def keyframe_path(video_id, frame_idx):
    """Duong dan anh keyframe. FE tu ghep tien to media_url (server local nay)."""
    return f"/keyframe/{video_id}/{int(frame_idx):06d}.jpg"


def index2info(lst_answers):
    """Dung dang du lieu ma FE dang doc (submit.jsx, index.jsx)."""
    info = {
        "lst_idxs": [],
        "lst_keyframe_idxs": [],
        "lst_keyframe_paths": [],
        "lst_video_idxs": [],
        "lst_answers": [],
        "lst_sources": [],
    }
    for ans in lst_answers:
        vid = ans["video_id"]
        fi = int(ans["frame_idx"])
        info["lst_idxs"].append(entry_key(vid, fi))
        info["lst_keyframe_idxs"].append(fi)
        info["lst_keyframe_paths"].append(keyframe_path(vid, fi))
        info["lst_video_idxs"].append(vid)
        info["lst_answers"].append(ans.get("answer"))
        info["lst_sources"].append(ans.get("source", "manual"))
    return info


def add_submit(ques_name, data):
    parsed = entry_from_payload(data)
    if parsed is None:
        logger.warning("submit thieu video_id/frame_idx: %s", data)
        return False
    vid, fi = parsed

    lst = AnswerDict.setdefault(ques_name, [])
    for a in lst:
        if a["video_id"] == vid and int(a["frame_idx"]) == fi:
            # Da co roi: nang len manual va cap nhat dap an chu neu co.
            a["source"] = "manual"
            if data.get("answer") is not None:
                a["answer"] = data["answer"]
            store_answer()
            return True

    lst.append(make(vid, fi, source="manual", answer=data.get("answer"),
                    frames=data.get("frames")))
    ReorderStatus.setdefault(ques_name, {"status": False, "owner": ""})
    store_status()
    store_answer()
    return True


def clear_submit_helper(ques_name, payload):
    lst = AnswerDict.get(ques_name)
    if not lst:
        logger.warning("Question name: %s not exist", ques_name)
        return
    parsed = entry_from_payload(payload)
    if parsed is None:
        return
    vid, fi = parsed
    AnswerDict[ques_name] = [
        a for a in lst
        if not (a["video_id"] == vid and int(a["frame_idx"]) == fi)
    ]
    store_answer()


def add_ignore(ques_name, payload, auto_ignore):
    items = payload if isinstance(payload, list) else [payload]
    cur = AnswerIgnoreDict.setdefault(ques_name, [])
    for it in items:
        parsed = entry_from_payload(it)
        if parsed is None:
            continue
        key = entry_key(*parsed)
        if key not in cur:
            cur.append(key)
        elif not auto_ignore:
            # bam lai lan nua = bo ignore
            cur.remove(key)
    store_ignore()


def clear_ignore_helper(ques_name, payload):
    cur = AnswerIgnoreDict.get(ques_name)
    if not cur:
        logger.warning("Question name: %s not exist", ques_name)
        return
    parsed = entry_from_payload(payload)
    if parsed is None:
        return
    key = entry_key(*parsed)
    if key in cur:
        cur.remove(key)
    store_ignore()


def add_user(user, ques_name):
    if not user:
        return
    lst = UserDict.setdefault(user, [])
    if ques_name not in lst:
        lst.append(ques_name)
        UserDict[user] = sorted(lst)
    store_user()


def check_owned_all(username):
    owned = set(UserDict.get(username) or ())
    return [{"question": q, "owned": q in owned} for q in sorted(AnswerDict)]


##################### Socket events ############################
@sio.event
async def submit(sid, data):
    logger.info("submit")
    ques_name = data["questionName"]
    ok = add_submit(ques_name, data)
    add_user(data.get("user"), ques_name)
    result = {
        "questionName": ques_name,
        "data": index2info(AnswerDict.get(ques_name, [])),
    }
    if not ok:
        result["error"] = "payload thieu video_id/frame_idx"
    await sio.emit("submit", result)


@sio.event
async def clearsubmit(sid, data):
    logger.info("clear submit")
    ques_name = data["questionName"]
    clear_submit_helper(ques_name, data)
    await sio.emit("clearsubmit", {
        "questionName": ques_name,
        "data": index2info(AnswerDict.get(ques_name, [])),
    })


@sio.event
async def setanswers(sid, data):
    """Thay ca danh sach dap an -- dung cho nut Auto-fill 100 dong.

    FE goi app.py /autofill (noi co ArtifactStore va ma tran dac trung) roi day
    ket qua sang day. Server nay khong tinh toan gi, dung nghia server trang thai.
    """
    logger.info("set answers")
    ques_name = data["questionName"]
    AnswerDict[ques_name] = normalize(data.get("answers") or [])
    store_answer()
    add_user(data.get("user"), ques_name)
    await sio.emit("submit", {
        "questionName": ques_name,
        "data": index2info(AnswerDict[ques_name]),
    })


@sio.event
async def ignore(sid, data):
    logger.info("ignore")
    ques_name = data["questionName"]
    add_ignore(ques_name, data.get("idx", data), data.get("autoIgnore", False))
    await sio.emit("ignore", {
        "questionName": ques_name,
        "data": AnswerIgnoreDict.get(ques_name, []),
    })


@sio.event
async def clearignore(sid, data):
    logger.info("clear ignore")
    ques_name = data["questionName"]
    clear_ignore_helper(ques_name, data.get("idx", data))
    await sio.emit("ignore", {
        "questionName": ques_name,
        "data": AnswerIgnoreDict.get(ques_name, []),
    })


@sio.event
async def reorder(sid, data):
    """Sap xep lai theo thu tu FE keo tha. Danh sach dap an la CO THU TU nen
    day chinh la thao tac quyet dinh thu hang nop bai."""
    logger.info("re order")
    ques_name = data["questionName"]
    order = (data.get("data") or {}).get("lst_idxs") or []
    keys = [k for k in (parse_key(x) for x in order) if k]
    if AnswerDict.get(ques_name) and keys:
        AnswerDict[ques_name] = reorder_answers(AnswerDict[ques_name], keys)
        store_answer()

    ReorderStatus.setdefault(ques_name, {"status": False, "owner": ""})
    ReorderStatus[ques_name].update(status=False, owner="")
    store_status()

    await sio.emit("reorder", {
        "questionName": ques_name,
        "data": index2info(AnswerDict.get(ques_name, [])),
    })


@sio.event
async def activereorder(sid, data):
    logger.info("active reorder")
    ques_name = data["questionName"]
    user = data.get("user", "")
    status = {"ques_name": ques_name, "user": user, "is_accepted": False}
    ReorderStatus.setdefault(ques_name, {"status": False, "owner": ""})
    if ReorderStatus[ques_name]["status"] and not data.get("isAdmin"):
        logger.error("Reorder an active question error")
    else:
        ReorderStatus[ques_name].update(status=True, owner=user)
        store_status()
        status["is_accepted"] = True
    await sio.emit("activereorder", status)


@sio.event
async def viewsubmitted(sid, data):
    logger.info("view submitted")
    ques_name = data["questionName"]
    if AnswerDict.get(ques_name):
        await sio.emit("viewsubmitted", {
            "questionName": ques_name,
            "data": index2info(AnswerDict[ques_name]),
        })
    else:
        await sio.emit("viewsubmitted", {})


##################### Anh keyframe (local) #####################
@app.get("/keyframe/{video_id}/{name}")
def keyframe_image(video_id: str, name: str):
    """Trich anh keyframe tu data/videos va cache lai.

    Chay o LOCAL vi Kaggle khong mount video goc. Chi 3,3% keyframe moi co san
    JPG cua BTC nen phai trich thang tu mp4 de anh dung voi frame_idx nop bai.
    """
    svc = get_media()
    if svc is None:
        return {"error": _media["error"], "status_code": 503}
    try:
        frame_idx = int(os.path.splitext(name)[0])
    except ValueError:
        return {"error": f"ten anh khong hop le: {name!r}", "status_code": 400}

    path = svc.get(video_id, frame_idx)
    if not path:
        return {"error": f"khong trich duoc {video_id}#{frame_idx}",
                "status_code": 404}
    return FileResponse(path, media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"})


##################### Q/A (local, can anh that) ################
@app.post("/qa")
def qa_endpoint(request: QaRequest):
    """Q/A bang VLM tren frame net nhat trong [start_pts, end_pts].

    Chay local vi buoc nay CAN anh that -- ma anh chi trich duoc o day.
    """
    svc = get_media()
    if svc is None:
        return {"error": _media["error"], "status_code": 503}

    from retrieval.qa import extract_best_frame, solve_qa

    store = _media["store"]
    best = extract_best_frame(request.video_id, request.start_pts,
                              request.end_pts, store)
    if best is None:
        return {"error": "Không tìm thấy frame nào trong khoảng thời gian",
                "status_code": 404}

    frame_idx = int(best["frame_idx"])
    pts_time = float(best["pts_time"])
    image_path = svc.get(request.video_id, frame_idx)
    if not image_path:
        return {"error": "Không thể trích xuất ảnh", "status_code": 404}

    from retrieval.bridge import ContextBridge

    if "bridge" not in _media:
        _media["bridge"] = ContextBridge(store)
    ocr = _media["bridge"].ocr_at(request.video_id, pts_time, frame_idx=frame_idx)
    ocr_texts = [i["text"] for i in ocr.get("items", [])]

    qa = solve_qa(request.video_id, request.question, image_path, ocr_texts)
    return {
        "video_id": request.video_id,
        "frame_idx": frame_idx,
        "pts_time": pts_time,
        "answer": qa["answer"],
        "degraded": qa["degraded"],
        "vlm": {k: v for k, v in qa.items() if k != "answer"},
        "ocr_texts_used": ocr_texts,
    }


##################### HTTP #####################################
@app.get("/health")
async def health():
    svc = get_media()
    return {
        "ok": True,
        "role": "local state + media server",
        "n_question": len(AnswerDict),
        "n_answer": sum(len(v) for v in AnswerDict.values()),
        "media": ("san sang" if svc else f"TAT - {_media['error']}"),
        "back_up": os.path.abspath(back_up_folder),
    }


@app.get("/submit")
async def submit_get(item: str = "", frame: str = "", session: str = ""):
    if not item or not frame:
        return {"description": "Missing item or frame", "status": "error"}
    return {
        "description": f"Submitted item={item}, frame={frame}, session={session}",
        "status": "success",
    }


@app.get("/getallques")
async def get_all_ques():
    return sorted(AnswerDict)


@app.post("/getsubmitques")
async def get_submit_ques(request: UserRequest):
    return UserDict.get(request.user, [])


@app.post("/getquestions")
async def get_questions(request: UsernameRequest):
    return check_owned_all(request.username)


@app.post("/getignoredquestions")
async def get_ignored_questions():
    return list(AnswerIgnoreDict)


@app.post("/getignore")
async def get_ignore(request: QuestionNameRequest):
    return {
        "questionName": request.questionName,
        "data": AnswerIgnoreDict.get(request.questionName, []),
    }


##################### Export CSV/ZIP ###########################
@app.get("/export/kis")
async def export_kis(questionName: str = Query(...)):
    answers = AnswerDict.get(questionName, [])
    if not answers:
        return Response(
            content=json.dumps(
                {"error": f"Không có dữ liệu cho câu hỏi '{questionName}'"},
                ensure_ascii=False),
            status_code=400, media_type="application/json")
    if len(answers) > 100:
        return Response(
            content=json.dumps({"error": (
                f"Câu hỏi '{questionName}' có {len(answers)} dòng, vượt quá giới "
                f"hạn 100 dòng/file của cuộc thi. Vui lòng xoá bớt trên UI trước "
                f"khi export.")}, ensure_ascii=False),
            status_code=400, media_type="application/json")
    return Response(
        content=kis_csv(answers), media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{questionName}.csv"'})


@app.get("/export/submission_zip")
async def export_submission_zip():
    if not AnswerDict:
        return Response(
            content=json.dumps({"error": "Không có dữ liệu trong AnswerDict"},
                               ensure_ascii=False),
            status_code=400, media_type="application/json")

    buf = io.BytesIO()
    warnings = []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for key in sorted(AnswerDict):
            answers = AnswerDict.get(key, [])
            if not answers:
                warnings.append(f"{key}: không có dữ liệu, đã bỏ qua.")
                continue
            if key.endswith("-qa"):
                content = qa_csv(answers)
            elif key.endswith("-trake"):
                content = trake_csv(answers)
            else:
                if len(answers) > 100:
                    msg = (f"{key}: có {len(answers)} dòng (vượt quá giới hạn "
                           f"100), đã bỏ qua.")
                    warnings.append(msg)
                    logger.warning("Export ZIP: %s", msg)
                    continue
                content = kis_csv(answers)
            zf.writestr(f"submission/{key}.csv", content)
        if warnings:
            zf.writestr("submission/_WARNINGS.txt", "\n".join(warnings))

    buf.seek(0)
    return Response(
        content=buf.read(), media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="submission.zip"'})


socket_app = socketio.ASGIApp(sio, app)

if __name__ == "__main__":
    uvicorn.run("socket_app:socket_app", host="0.0.0.0",
                port=int(os.environ.get("PORT", "8081")))
