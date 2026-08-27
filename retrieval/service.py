"""Cac doi tuong dung chung giua app.py va socket_app.py, nap mot lan."""

import glob
import os

from retrieval.bridge import ContextBridge
from retrieval.config import ARTIFACT_ROOT, META_DIR, FusionConfig
from retrieval.encoder import SigLipTextEncoder
from retrieval.engine import FusionEngine
from retrieval.frames import KeyframeImages
from retrieval.objects import ObjectIndex
from retrieval.panels import PanelSearch
from retrieval.store import ArtifactStore
from retrieval.textindex import TextChannels

_state = {}


def get():
    if "engine" not in _state:
        cfg = FusionConfig.load()
        store = ArtifactStore()
        channels = TextChannels()
        encoder = SigLipTextEncoder()
        objects = ObjectIndex()
        _state.update(
            config=cfg,
            store=store,
            channels=channels,
            encoder=encoder,
            engine=FusionEngine(store, channels, encoder, cfg),
            bridge=ContextBridge(store, cfg),
            images=KeyframeImages(store),
            objects=objects,
            panels=PanelSearch(store, objects, channels, cfg, encoder),
        )
    return _state


def preload(strict=True):
    """Nap DU model truoc khi mo cong.

    Tren Kaggle, nap lazy nghia la vai truy van dau chay KHONG co kenh thi giac
    -- nguoi dung nhan ket qua chi xep bang BM25 ma khong biet. Nap truoc, va
    strict=True thi that bai luc khoi dong con hon im lang tra ket qua sai.
    """
    s = get()
    enc = s["encoder"]
    try:
        v = enc.encode_texts(["warmup"])
        s["store"].assert_encoder_matches(enc)
        return {"ok": True, "dim": int(v.shape[1]), "device": enc.device,
                "model": enc.model_name}
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if strict:
            raise RuntimeError(
                f"Khong nap duoc SigLIP text encoder: {msg}\n"
                f"Kenh thi giac se TAT va ket qua chi con xep bang BM25. "
                f"Dat SIGLIP_PRELOAD=0 de bo qua kiem tra nay."
            ) from e
        return {"ok": False, "error": msg}


def reload_config():
    cfg = FusionConfig.load()
    _state["config"] = cfg
    if "engine" in _state:
        _state["engine"].cfg = cfg
        _state["bridge"].cfg = cfg
        _state["panels"].cfg = cfg
    return cfg


def _llm_status():
    """Trang thai LLM/VLM. Chua cau hinh KHONG phai loi -- he thong van chay
    che do khong-LLM, chi mat phan ra truy van va Q/A."""
    try:
        from retrieval import llm_client

        st = llm_client.status()
        st["che_do_khong_llm"] = not any(st.get("kha_dung", {}).values())
        return st
    except Exception as e:
        return {"kha_dung": {}, "che_do_khong_llm": True,
                "error": f"{type(e).__name__}: {e}"}


# --------------------------------------------------------------------------
def diagnostics():
    """Viec 5 -- phan biet "he thong loi" voi "video nay chua nam trong index"."""
    s = get()
    store, channels, encoder, bridge = (
        s["store"], s["channels"], s["encoder"], s["bridge"])

    meta_videos = {
        os.path.basename(f)[:-5]
        for f in glob.glob(os.path.join(META_DIR, "*.json"))
    }
    indexed = set(store.video_ids)

    # Video BTC co metadata nhung chua co features -> "chua nam trong index",
    # khong phai he thong loi.
    missing = sorted(meta_videos - indexed)

    per_pack = []
    for pack, n in sorted(store.packs.items()):
        sub = store.meta[store.meta["pack"] == pack]
        per_pack.append({
            "pack": pack,
            "n_video": n,
            "n_keyframe": int(len(sub)),
            "co_asr": len(glob.glob(os.path.join(ARTIFACT_ROOT, pack, "asr", "*.json"))),
            "co_ocr": len(glob.glob(os.path.join(ARTIFACT_ROOT, pack, "ocr", "*.json"))),
        })

    return {
        "build_id": store.build_id,
        "build_id_thieu": store.build.get("_missing"),
        "build": {k: v for k, v in store.build.items() if k != "_path"},
        "packs": per_pack,
        "n_pack": len(store.packs),
        "n_video_da_phu": len(indexed),
        "n_keyframe": int(len(store.meta)),
        "embed_dim": int(store.X.shape[1]) if store.X.size else 0,
        "metadata": {
            "dir": META_DIR,
            "n_video": len(meta_videos),
            "n_thieu_artifacts": len(missing),
            "video_thieu_artifacts": missing[:50],
            "ghi_chu": ("Video co metadata nhung chua co features -- tim khong ra "
                        "la dung, khong phai loi he thong."),
        },
        "video_bi_bo_qua": [
            {"video_id": v, "ly_do": r} for v, r in store.skipped[:50]
        ],
        "kenh_van_ban": channels.status(),
        "siglip": encoder.status(),
        "llm": _llm_status(),
        "object_bridge": bridge.object_status(),
        "object_index": s["objects"].status(),
        "anh_keyframe": s["images"].stats(),
        "config": {
            "weights": s["config"].weights,
            "rrf_k": s["config"].rrf_k,
            "mmr_lambda": s["config"].mmr_lambda,
            "max_per_video": s["config"].max_per_video,
            "min_gap_sec": s["config"].min_gap_sec,
            "object_tolerance_sec": s["config"].object_tolerance_sec,
            "ocr_tolerance_sec": s["config"].ocr_tolerance_sec,
        },
    }
