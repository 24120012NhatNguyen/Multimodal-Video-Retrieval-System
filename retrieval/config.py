"""Cau hinh tang hop nhat. Moi con so co the hieu chinh bang bo eval deu nam o day."""

import json
import os
from dataclasses import dataclass, field, asdict

CONFIG_PATH = os.environ.get("FUSION_CONFIG", "config/fusion.json")

# Thu muc artifacts. Moi pack la mot thu muc con chua features/ keyframes/ asr/ ocr/.
ARTIFACT_ROOT = os.environ.get("ARTIFACT_ROOT", "data/artifacts")
# Metadata cua BTC (khong nam trong cau truc pack).
META_DIR = os.environ.get(
    "META_DIR", "data/artifacts/media-info-aic25-b1/media-info"
)
VIDEO_DIR = os.environ.get("VIDEO_DIR", "data/videos")
KEYFRAME_CACHE = os.environ.get("KEYFRAME_CACHE", "data/keyframe_cache")

# Model PHAI khop voi BUILD.json. Features la 1152-d cua siglip-so400m; dung
# encoder khac chieu (hoac khac khong gian) thi cosine van ra so dep nhung vo
# nghia -- xem ArtifactStore.assert_encoder_matches().
SIGLIP_MODEL = os.environ.get("SIGLIP_MODEL", "google/siglip-so400m-patch14-384")


# ---------------------------------------------------------------------------
# LLM / VLM -- KHONG hardcode model ID o bat ky dau.
#
# Chu ky khai tu cua Google rat nhanh (gemini-2.0-flash-001 dong 01/06/2026,
# cum 2.5 dong 16/10/2026), nen model ID phai la tham so van hanh, khong phai
# hang so trong ma nguon.
#
# CO Y de trong khi chua dat: mot ID doan bua se tra 404 giua luc thi -- dung
# cai loi ma muc nay dang muon chan. Chua dat -> he thong chay che do khong-LLM
# (truy van tho, khong phan ra), xem retrieval/llm_client.py.
#
# Dat gia tri that lay tu chinh tai khoan cua ban:
#     export GEMINI_API_KEY=...
#     python -m retrieval.llm_client --list      # in ra ID kem ban co that
#     export GEMINI_MODEL_FLASH=gemini-3.7-flash
#     export GEMINI_MODEL_PRO=gemini-2.5-pro

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")
LLM_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
# flash: tac vu nhanh (Q/A tren mot frame). pro: phan ra truy van kho.
LLM_MODEL_FLASH = os.environ.get("GEMINI_MODEL_FLASH") or None
LLM_MODEL_PRO = os.environ.get("GEMINI_MODEL_PRO") or None

LLM_TIMEOUT_SEC = float(os.environ.get("LLM_TIMEOUT_SEC", "20"))
LLM_MAX_RETRY = int(os.environ.get("LLM_MAX_RETRY", "2"))
# Sau ngan nay lan hong lien tiep thi ngung goi API trong LLM_COOLDOWN_SEC.
# Muc dich: mot API chet khong duoc keo dai do tre cua tung truy van suot buoi thi.
LLM_BREAKER_THRESHOLD = int(os.environ.get("LLM_BREAKER_THRESHOLD", "3"))
LLM_COOLDOWN_SEC = float(os.environ.get("LLM_COOLDOWN_SEC", "60"))


@dataclass
class FusionConfig:
    # --- Viec 1: trong so RRF (mac dinh 1.0 cho tat ca) -------------------
    weights: dict = field(default_factory=lambda: {
        "siglip": 1.0,
        "meta": 1.0,
        "meta_fold": 1.0,
        "asr": 1.0,
        "ocr": 1.0,
    })
    rrf_k: int = 60
    # So video moi kenh tra ve truoc khi gop.
    channel_topn: int = 100
    # So video giu lai sau khi gop, truoc khi xuong tang frame.
    video_topn: int = 30
    # So frame tra ve cho luoi ket qua.
    frame_topk: int = 200

    # --- Dong hang chuoi su kien (DP) -------------------------------------
    # Chi chay khi truy van la "chuoi hanh dong chung chung": tung menh de deu
    # pho bien, chi THU TU + khoang cach THOI GIAN moi phan biet duoc video.
    # Truy van co anchor (ten rieng, chu tren bien hieu) thi tim phang da du.
    dp_enabled: bool = True
    # Khoang thoi gian toi da giua hai su kien lien tiep. 30s vua du cho mot
    # chuoi canh trong ban tin; nam trong bang hang so can DO chu khong doan.
    dp_delta_sec: float = 30.0
    dp_gamma: float = 0.5
    # So video dua vao DP (DP chay ~6ms/video nen day khong phai nut co chai).
    dp_video_topn: int = 30

    # --- Viec 3: auto-fill ------------------------------------------------
    autofill_target: int = 100
    mmr_lambda: float = 0.7          # lambda trong cong thuc MMR
    max_per_video: int = 5           # m: so frame toi da moi video dong gop
    min_gap_sec: float = 2.0         # hai frame cung video cach nhau it nhat
    neighbour_window_sec: float = 3.0  # mo rong lan can quanh frame thu cong

    # --- Viec 4: dung sai cau noi ----------------------------------------
    # Nguoi/xe/do vat hiem khi bien mat trong 3 giay.
    object_tolerance_sec: float = 3.0
    # Lower-third chi hien ~4 giay, do hoa doi lien tuc.
    ocr_tolerance_sec: float = 1.5
    # ASR neo theo thoi gian -> khong phai cau noi, chi noi rong bien doan.
    asr_pad_sec: float = 2.0

    def save(self, path=CONFIG_PATH):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path=CONFIG_PATH):
        cfg = cls()
        if not os.path.exists(path):
            return cfg
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            return cfg
        for k, v in raw.items():
            if not hasattr(cfg, k):
                continue
            # weights gop vao mac dinh de them kenh moi khong lam mat kenh cu
            if k == "weights" and isinstance(v, dict):
                cfg.weights.update(v)
            else:
                setattr(cfg, k, v)
        return cfg
