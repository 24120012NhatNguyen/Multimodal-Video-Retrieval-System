from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, field_validator


def _int_or_default(value: Any, default: int) -> int:
    """Ép giá trị về int, tự lùi về `default` khi ô nhập bị bỏ trống.

    Các ô số trên UI (K, range, search space) gửi chuỗi rỗng khi người dùng xoá
    trắng. Nếu không xử lý, pydantic trả 422 "Input should be a valid integer"
    và chặn nguyên lượt tìm kiếm.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return default
        try:
            return int(float(text))
        except ValueError:
            return default
    if isinstance(value, (int, float)):
        return int(value)
    return default


# Define Pydantic models for request validation
class TextSearchRequest(BaseModel):
    search_space: int = 0
    k: int = 500
    nomic: bool = True
    clipv2: bool = False
    textquery: str = ""
    range_filter: int = 3
    filter: bool = False
    id: Optional[List[str]] = None
    ignore: Optional[bool] = False
    ignore_idxs: Optional[List[str]] = None
    filtervideo: int = 0
    videos: Optional[List[Dict[str, Any]]] = None

    # --- Viec 1: tang hop nhat cap video ---------------------------------
    # Co bat/tat. Tat (mac dinh) thi endpoint chay nguyen duong cu (xep hang
    # frame truc tiep), de so sanh duoc hai che do.
    fusion: bool = False
    # Hai chuoi truy van la CO Y: SigLIP an tieng Anh, BM25 an tieng Viet.
    # Thieu query_en thi lay textquery; thieu query_vi cung vay.
    query_en: str = ""
    query_vi: str = ""
    video_topn: Optional[int] = None
    channels: Optional[List[str]] = None
    weights: Optional[Dict[str, float]] = None
    # Phan ra truy van bang LLM (tier "pro"). Tat -> dung truy van tho.
    # Da tu nhap query_en thi co nay bi bo qua.
    decompose: bool = True
    # Dong hang chuoi su kien bang DP -- NGUOI DUNG NOI GI THI NGHE NAY:
    #   True  = ep BAT du bo phan loai cho la khong can
    #   False = ep TAT (de so sanh hai che do)
    #   None  = de he thong tu quyet dinh theo loai truy van
    align: Optional[bool] = None
    # Ep loai truy van, ghi de ca LLM lan bo do anchor.
    # "generic_chain" | "anchored" | None (tu quyet dinh)
    kind: Optional[str] = None

    @field_validator("query_en", "query_vi", "textquery", mode="before")
    @classmethod
    def _qtext(cls, v: Any) -> str:
        return "" if v is None else str(v)

    @field_validator("video_topn", mode="before")
    @classmethod
    def _video_topn(cls, v: Any) -> Optional[int]:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        return _int_or_default(v, 30)

    @field_validator("k", mode="before")
    @classmethod
    def _k(cls, v: Any) -> int:
        return _int_or_default(v, 500)

    @field_validator("range_filter", mode="before")
    @classmethod
    def _range_filter(cls, v: Any) -> int:
        return _int_or_default(v, 3)

    @field_validator("search_space", "filtervideo", mode="before")
    @classmethod
    def _zero_default(cls, v: Any) -> int:
        return _int_or_default(v, 0)

    @field_validator("videos", mode="before")
    @classmethod
    def videos_accept_dict_or_list(cls, v: Any) -> Optional[List[Dict[str, Any]]]:
        if v is None:
            return None
        if isinstance(v, list):
            return v
        if isinstance(v, dict):
            return None  # dict (vd. {}) từ FE → coi như không có kết quả trước
        return None


class PanelSearchRequest(BaseModel):
    k: int = 500
    search_space: int = 0
    useid: bool = True
    id: Optional[List[int]] = None
    ignore: Optional[bool] = False
    ignore_idxs: Optional[List[int]] = None
    ocr: str = ""
    asr: str = ""
    dragObject: Optional[List[Dict[str, Any]]] = []
    tags: Optional[List[str]] = []
    amount: Optional[str] = ""

    @field_validator("k", mode="before")
    @classmethod
    def _k(cls, v: Any) -> int:
        return _int_or_default(v, 500)

    @field_validator("search_space", mode="before")
    @classmethod
    def _search_space(cls, v: Any) -> int:
        return _int_or_default(v, 0)

    @field_validator("ocr", "asr", "amount", mode="before")
    @classmethod
    def _text(cls, v: Any) -> str:
        return "" if v is None else str(v)


class FeedbackRequest(BaseModel):
    k: int = 500
    # FE gửi state `videos` là mảng kết quả (group_result_by_video); giữ Dict để
    # tương thích ngược với các client cũ.
    videos: Union[List[Dict[str, Any]], Dict[str, Any]] = []
    lst_pos_idxs: List[int] = []
    lst_neg_idxs: List[int] = []

    @field_validator("k", mode="before")
    @classmethod
    def _k(cls, v: Any) -> int:
        return _int_or_default(v, 500)


class TagRequest(BaseModel):
    text: str = ""


class TranslateRequest(BaseModel):
    textquery: str = ""


# Define Pydantic models for request bodies
class UserRequest(BaseModel):
    user: str = ""


class UsernameRequest(BaseModel):
    username: str = ""


class QuestionNameRequest(BaseModel):
    questionName: str = ""


class AnswerEntry(BaseModel):
    """Mot dong dap an -- schema Viec 2."""

    video_id: str
    frame_idx: int
    source: str = "manual"
    answer: Optional[str] = None
    frames: Optional[List[int]] = None

    @field_validator("frame_idx", mode="before")
    @classmethod
    def _frame_idx(cls, v: Any) -> int:
        return _int_or_default(v, 0)

    @field_validator("source", mode="before")
    @classmethod
    def _source(cls, v: Any) -> str:
        return v if v in ("manual", "autofill") else "manual"


class AutofillRequest(BaseModel):
    """Viec 3 -- lap day 100 dong tren tap ket qua da co san."""

    manual: List[AnswerEntry] = []
    # Tap ung vien dang hien tren luoi: [{"video_id", "frame_idx", "score"}]
    candidates: List[Dict[str, Any]] = []
    ignore: List[Dict[str, Any]] = []
    target: int = 100
    mmr_lambda: Optional[float] = None
    max_per_video: Optional[int] = None

    @field_validator("target", mode="before")
    @classmethod
    def _target(cls, v: Any) -> int:
        return _int_or_default(v, 100)


class KeyframeContextRequest(BaseModel):
    """Viec 4 -- panel object / OCR / ASR cua mot keyframe."""

    video_id: str
    frame_idx: int

    @field_validator("frame_idx", mode="before")
    @classmethod
    def _frame_idx(cls, v: Any) -> int:
        return _int_or_default(v, 0)

class TrakeRequest(BaseModel):
    """Yêu cầu dóng hàng sự kiện TRAKE."""
    video_id: str
    events: List[str]  # Danh sách các câu truy vấn text cho từng sự kiện
    delta: float = 5.0
    gamma: float = 0.5

class QaRequest(BaseModel):
    """Yêu cầu giải quyết câu hỏi QA thông qua VLM."""
    video_id: str
    question: str
    start_pts: float
    end_pts: float
