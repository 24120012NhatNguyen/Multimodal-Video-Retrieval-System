import json
import os
import socketio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from utils.logger_config import get_logger
from utils.models import QuestionNameRequest, UsernameRequest, UserRequest, QaRequest

logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Create Socket.IO server
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins=["*"],
    ping_timeout=60,
    ping_interval=25,
)

###################### Initialize dict ########################
back_up_folder = "back_up"
if not os.path.exists(back_up_folder):
    os.mkdir(back_up_folder)

def load_json_or_default(path, default):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return default

AnswerDict = load_json_or_default(f"{back_up_folder}/answer.json", {})
UserDict = load_json_or_default(f"{back_up_folder}/user.json", {})
ReorderStatus = load_json_or_default(f"{back_up_folder}/reorder_status.json", {})
AnswerIgnoreDict = load_json_or_default(f"{back_up_folder}/answer_ignore.json", {})
###############################################################


####################### Helper Utils ##########################
def store_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)

def store_answer(): store_json(f"{back_up_folder}/answer.json", AnswerDict)
def store_user(): store_json(f"{back_up_folder}/user.json", UserDict)
def store_status(): store_json(f"{back_up_folder}/reorder_status.json", ReorderStatus)
def store_ignore(): store_json(f"{back_up_folder}/answer_ignore.json", AnswerIgnoreDict)

def build_image_path(video_id, frame_idx):
    # Dựa trên phản hồi: frontend/public/static/images/keyframe
    # Cấu trúc phổ biến: L21_V001 / 001234.jpg
    return f"/static/images/keyframe/{video_id}/{int(frame_idx):06d}.jpg"


def add_submit(ques_name, data):
    video = data.get("video")
    frame_idx = data.get("frame_idx")
    answer_text = data.get("answer", None)
    
    # Giả sử FE cũ vẫn gửi idx, ta có thể sinh ra 1 ID giả (vd: hash)
    # Nhưng theo yêu cầu, hệ thống mới xài trực tiếp video_id + frame_idx.
    # Ta dùng f"{video}_{frame_idx}" làm khóa duy nhất nếu cần.
    pseudo_idx = f"{video}_{frame_idx}"
    
    new_answer = {
        "video": video,
        "frames": [int(frame_idx)],
        "answer": answer_text,
        "idxs": [pseudo_idx] # Fake ID
    }

    if ques_name not in AnswerDict:
        AnswerDict[ques_name] = [new_answer]
    else:
        exists = False
        for ans in AnswerDict[ques_name]:
            if isinstance(ans, dict) and ans.get("video") == video and ans.get("frames") == [int(frame_idx)]:
                exists = True
                ans["answer"] = answer_text
                break
        if not exists:
            AnswerDict[ques_name].append(new_answer)
            
    if ques_name not in ReorderStatus:
        ReorderStatus[ques_name] = {"status": False, "owner": ""}
        store_status()
    store_answer()


def add_ignore(ques_name, video, frame_idx, autoIgnore):
    pseudo_idx = f"{video}_{frame_idx}"
    if ques_name not in AnswerIgnoreDict:
        AnswerIgnoreDict[ques_name] = [pseudo_idx]
    else:
        if pseudo_idx not in AnswerIgnoreDict[ques_name]:
            AnswerIgnoreDict[ques_name].append(pseudo_idx)
        elif not autoIgnore:
            AnswerIgnoreDict[ques_name].remove(pseudo_idx)
    store_ignore()


def add_user(user, ques_name):
    if user not in UserDict:
        UserDict[user] = [ques_name]
    else:
        if ques_name not in UserDict[user]:
            UserDict[user].append(ques_name)
            UserDict[user] = sorted(UserDict[user])
    store_user()


def clear_submit_helper(ques_name, video, frame_idx):
    pseudo_idx = f"{video}_{frame_idx}"
    if ques_name in AnswerDict:
        for ans in AnswerDict[ques_name]:
            if isinstance(ans, dict):
                if pseudo_idx in ans.get("idxs", []):
                    AnswerDict[ques_name].remove(ans)
                    break
    store_answer()


def clear_ignore_helper(ques_name, video, frame_idx):
    pseudo_idx = f"{video}_{frame_idx}"
    if ques_name in AnswerIgnoreDict:
        if pseudo_idx in AnswerIgnoreDict[ques_name]:
            AnswerIgnoreDict[ques_name].remove(pseudo_idx)
    store_ignore()


def index2info(lst_answers):
    info = {
        "lst_idxs": [],
        "lst_keyframe_idxs": [],
        "lst_keyframe_paths": [],
        "lst_video_idxs": [],
        "lst_answers": [],
    }
    for ans in lst_answers:
        if isinstance(ans, dict):
            video = ans.get("video")
            frame_idx = ans.get("frames")[0] if ans.get("frames") else 0
            pseudo_idx = f"{video}_{frame_idx}"
            image_path = build_image_path(video, frame_idx)

            info["lst_idxs"].append(pseudo_idx)
            info["lst_keyframe_idxs"].append(frame_idx)
            info["lst_keyframe_paths"].append(image_path)
            info["lst_video_idxs"].append(video)
            info["lst_answers"].append(ans.get("answer", None))
    return info


def check_owned_all(username):
    all_ques = sorted(list(AnswerDict.keys()))
    checked_ques = []

    if username not in UserDict:
        return [{"question": q, "owned": False} for q in all_ques]

    for ques in all_ques:
        checked_ques.append({"question": ques, "owned": (ques in UserDict[username])})
    return checked_ques


##################### API Routes cho QA (VLM) ##################
@app.post("/qa")
def qa_endpoint_local(request: QaRequest):
    """
    QA endpoint được đưa về Local để có thể truy cập ảnh thực tế.
    """
    from retrieval.llm_client import get_client
    # Gỉa định rằng OCR đã được Kaggle xử lý trước, hoặc nếu không ta dùng tạm.
    # Tuy nhiên Frontend gọi QA sau khi đã xem ảnh.
    image_path = f"../frontend/public/static/images/keyframe/{request.video_id}/{int(request.frame_idx):06d}.jpg"
    if not os.path.exists(image_path):
        # Thu hồi về đường dẫn chuẩn nếu cần
        image_path = None
    
    # Prompt đơn giản cho QA
    prompt = f"Câu hỏi: {request.question}\nTrả lời ngắn gọn dựa trên hình ảnh:"
    client = get_client()
    result = client.generate(
        prompt,
        images=[image_path] if image_path else None,
        tier="flash",
    )
    
    if result.ok:
        return {
            "answer": result.text,
            "degraded": False,
            "source": "vlm_local",
        }
    return {
        "answer": None,
        "degraded": True,
        "error": result.error,
    }


##################### Web Sockets ##############################
@sio.event
async def submit(sid, data):
    ques_name = data.get("questionName")
    user = data.get("user")
    add_submit(ques_name, data)
    add_user(user, ques_name)
    result = {"questionName": ques_name, "data": index2info(AnswerDict[ques_name])}
    await sio.emit("submit", result)


@sio.event
async def clearsubmit(sid, data):
    ques_name = data.get("questionName")
    video = data.get("video")
    frame_idx = data.get("frame_idx")
    clear_submit_helper(ques_name, video, frame_idx)
    result = {
        "questionName": ques_name,
        "data": index2info(AnswerDict.get(ques_name, [])),
    }
    await sio.emit("clearsubmit", result)


@sio.event
async def ignore(sid, data):
    ques_name = data.get("questionName")
    video = data.get("video")
    frame_idx = data.get("frame_idx")
    add_ignore(ques_name, video, frame_idx, data.get("autoIgnore", False))
    result = {"questionName": ques_name, "data": AnswerIgnoreDict[ques_name]}
    await sio.emit("ignore", result)


@sio.event
async def clearignore(sid, data):
    ques_name = data.get("questionName")
    video = data.get("video")
    frame_idx = data.get("frame_idx")
    clear_ignore_helper(ques_name, video, frame_idx)
    result = {
        "questionName": ques_name,
        "data": AnswerIgnoreDict.get(ques_name, []),
    }
    await sio.emit("ignore", result)


@sio.event
async def reorder(sid, data):
    ques_name = data.get("questionName")
    lst_answers = data.get("data", {}).get("lst_answers", [])
    # Cần logic reorder cụ thể nếu frontend thay đổi
    pass


@sio.event
async def viewsubmitted(sid, data):
    ques_name = data.get("questionName")
    if ques_name in AnswerDict:
        result = {"questionName": ques_name, "data": index2info(AnswerDict[ques_name])}
        await sio.emit("viewsubmitted", result)
    else:
        await sio.emit("viewsubmitted", {})


@app.get("/getallques")
async def get_all_ques():
    return sorted(list(AnswerDict.keys()))

@app.post("/getsubmitques")
async def get_submit_ques(request: UserRequest):
    return UserDict.get(request.user, [])

@app.post("/getquestions")
async def get_questions(request: UsernameRequest):
    return check_owned_all(request.username)


if __name__ == "__main__":
    import uvicorn
    asgi_app = socketio.ASGIApp(sio, app)
    uvicorn.run(asgi_app, host="0.0.0.0", port=5000)
