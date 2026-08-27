"""Cau hinh tang hop nhat. Moi con so co the hieu chinh bang bo eval deu nam o day."""

import json
import os
from dataclasses import dataclass, field, asdict

CONFIG_PATH = os.environ.get("FUSION_CONFIG", "config/fusion.json")

# Thu muc artifacts. Moi pack la mot thu muc con chua features/ keyframes/ asr/ ocr/.
ARTIFACT_ROOT = os.environ.get("ARTIFACT_ROOT", "data/artifacts")
# Metadata + object cua BTC: nam NGANG HANG voi cac pack, ben trong
# ARTIFACT_ROOT. PHAI suy tu ARTIFACT_ROOT chu khong hardcode duong dan tuong
# doi -- tren Kaggle ARTIFACT_ROOT tro sang /kaggle/input/... nen duong dan
# "data/artifacts/..." khong ton tai, va hai kenh meta/meta_fold CHET AM THAM
# (log chi con MetaIndex[asr] va MetaIndex[ocr]).
def _under_root(env_key, *parts):
    v = os.environ.get(env_key)
    if v:
        return v
    return os.path.join(ARTIFACT_ROOT, *parts)


META_DIR = _under_root("META_DIR", "media-info-aic25-b1", "media-info")
OBJECT_DIR = _under_root("OBJECT_DIR", "objects-aic25-b1", "objects")
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
#     export ANTHROPIC_API_KEY=...
#     python -m retrieval.llm_client --list

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic")

if LLM_PROVIDER == "anthropic":
    LLM_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
    # ID cua Anthropic la DAY DU nhu the nay, KHONG them hau to ngay.
    # Haiku 4.5: 200K context, re va nhanh -- du cho phan ra truy van va Q/A.
    LLM_MODEL_FLASH = os.environ.get("CLAUDE_MODEL_FLASH", "claude-haiku-4-5")
    LLM_MODEL_PRO = os.environ.get("CLAUDE_MODEL_PRO", "claude-haiku-4-5")
else:
    LLM_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
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
    # Trong so san cho kenh ma NGUOI DUNG bat cong tac sang "on". Khong co no
    # thi nut bat la nut gia: trong so theo loai truy van dang la 0 nen kenh
    # duoc bat van khong bo phieu duoc.
    channel_on_weight: float = 1.0
    # --- Xep lai theo bang chung frame -----------------------------------
    # RRF xep video theo "may kenh bo phieu"; no khong biet ben trong video co
    # khoanh khac nao that su giong truy van khong. Xep lai theo diem thi giac
    # cao nhat trong video thi biet. Do tren bo eval: xem retrieval/engine.py.
    # Cach chia han muc frame cho luoi ket qua:
    #   "global"       sort toan cuc theo diem roi cat -- video khong co frame
    #                  diem cao BIEN MAT khoi luoi (loc am tham theo frame)
    #   "round_robin"  moi video top duoc 1 frame moi vong -- thu hang video
    #                  luon nhin thay duoc, nhung video dung chi co 1 frame/vong
    #   "hybrid"       bao dam 1 frame cho top (han muc / 3) video, con lai theo diem
    # Con so o day PHAI do bang eval/run_eval.py --no-llm (phan ra bang LLM
    # khong tat dinh, chay hai lan ra hai ket qua khac nhau -- khong so sanh duoc).
    frame_alloc: str = "global"
    rerank_by_frame: bool = False
    # Trong so cua thu hang RRF khi xep lai. 0 = xep hoan toan theo frame.
    rerank_rrf_weight: float = 0.0
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
    # gamma tinh bang DON VI DO LECH CHUAN, khong phai don vi cosine: diem su
    # kien da duoc chuan hoa z-score theo phan bo cua chinh truy van tren corpus
    # (retrieval.trake.event_stats). Ban truoc de 0.5 tren thang cosine (bien do
    # +/-0.13) nen nhanh "bo qua" cua DP khong bao gio duoc chon -- DP luon ep
    # khop du moi su kien ke ca su kien khong he co trong video.
    # Nguong z de coi mot su kien la CO MAT trong video. Diem khop tro thanh
    # (z - tau) nen su kien chi hoi giong se AM va bi bo qua -- do la co che duy
    # nhat lam nhanh "bo qua" cua DP song. Do tren corpus: p99 cua z ~2,8.
    dp_tau: float = 2.0
    # Khoan phat cau truc them cho moi su kien bi bo trong. tau da lam phan
    # nguong, nen mac dinh 0.
    dp_gamma: float = 0.0
    # Khoang cach TOI THIEU giua hai su kien lien tiep. 0 = chi doi hoi frame
    # sau muon hon frame truoc (co the la hai keyframe lien ke, ~2-3s).
    dp_min_gap_sec: float = 0.0
    # So video dua vao DP (DP chay ~6ms/video nen day khong phai nut co chai).
    dp_video_topn: int = 30

    # --- Viec 3: auto-fill ------------------------------------------------
    autofill_target: int = 100
    # So o dau GIU NGUYEN thu hang tim kiem. 20 chinh la nguong k = 20 trong
    # cong thuc cham diem: cac o 1-20 dang 0,60-1,00 diem moi o, khong danh cuoc
    # chung vao da dang hoa. Do: xem ghi chu dau retrieval/autofill.py.
    autofill_head: int = 20
    # Tu o head+1 tro di, hai frame cung video phai cach nhau it nhat ngan nay.
    # Vi pham thi bi day xuong cuoi, khong bi loai.
    #
    # 4s khong phai so bua: dap an duoc tinh la dung khi frame nam trong dung sai
    # +/-2s, tuc moi frame PHU mot cua so rong 4s. Lat dung 4s thi hai o lien
    # tiep phu hai khoang khong chong nhau va khong ho -- moi o mua duoc nhieu
    # dien tich thoi gian nhat. Do: gap 4s -> Final 0.6286; gap 8/15/30s -> 0.5714.
    autofill_tail_gap_sec: float = 4.0
    # So ung vien lay ve de lap 100 dong. LON HON so frame hien tren luoi: bai
    # nop khong nen bi gioi han boi thu nguoi dung dang nhin thay. Do: pool 500
    # -> Final 0.6000, pool 3000 -> 0.6286 (Q5 lot vao o hang 87).
    autofill_pool_topk: int = 3000
    # --- cac hang so cua co che MMR cu, khong con duoc dung -----------------
    # Giu lai de cau hinh cu doc vao khong bao loi. MMR da bi go: do duoc no lam
    # Final Score tut tu 0.5714 xuong 0.3429.
    mmr_lambda: float = 0.7
    max_per_video: int = 5
    min_gap_sec: float = 2.0
    neighbour_window_sec: float = 3.0

    # --- Viec 4: dung sai cau noi ----------------------------------------
    # Object cua BTC neo vao keyframe cua HO, khong trung keyframe cua ta; bac
    # cau qua pts_time lay tu map-keyframes. 2.5s: nguoi/xe/do vat hiem khi bien
    # mat trong khoang do, va khoang cach giua hai keyframe BTC lien tiep do
    # duoc phan lon < 5s.
    object_tolerance_sec: float = 2.5
    # Lower-third chi hien ~4 giay, do hoa doi lien tuc.
    ocr_tolerance_sec: float = 1.5
    # ASR neo theo thoi gian -> khong phai cau noi, chi noi rong bien doan.
    asr_pad_sec: float = 2.0
    # Nua do rong cua clip ngan quanh mot keyframe: [pts - w, pts + w].
    clip_window_sec: float = 2.0

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
