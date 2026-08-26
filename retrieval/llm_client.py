"""Lop abstraction mong cho LLM/VLM.

Ba rang buoc cua muc A1, deu duoc cuong che o day chu khong de cho noi goi:

1. KHONG hardcode model ID. Moi ID den tu retrieval.config (bien moi truong).
   Chua dat ID -> client bao khong kha dung, KHONG doan bua mot ID nao.

2. KHONG dung alias thieu so phien ban. Alias bi hot-swap va co the doi hanh vi
   giua luc thi; `looks_pinned()` canh bao khi ID khong co hau to phien ban.

3. FALLBACK BAT BUOC. Moi loi -- thieu key, thieu SDK, 404, timeout, het quota --
   deu tra ve LLMResult(ok=False) chu KHONG NEM EXCEPTION. Noi goi doc `.ok` va
   tu xuong che do khong-LLM. Them mot circuit breaker: API chet thi ngung goi
   trong mot khoang, de khong keo dai do tre cua moi truy van sau do.

Doi nha cung cap = them mot lop con LLMClient, khong sua logic truy xuat.
"""

import re
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

from retrieval import config as cfg

# ID cua Google co dang "<ho>-<phien ban>-<bien the>-<NNN>", vd. ...-flash-001.
# Thieu hau to so la alias di dong.
_PINNED = re.compile(r"-\d{3}$|@\d+$")

# Cac loi coi la "API dang hong" -> ha circuit breaker.
_DEGRADE_HINTS = (
    "404", "not found", "timeout", "timed out", "deadline",
    "quota", "429", "resource_exhausted", "rate limit",
    "503", "unavailable", "500", "internal",
)


def looks_pinned(model_id: Optional[str]) -> bool:
    return bool(model_id and _PINNED.search(model_id))


@dataclass
class LLMResult:
    ok: bool
    text: Optional[str] = None
    error: Optional[str] = None
    # Vi sao khong dung duoc -- de /diagnostics phan biet "chua cau hinh" voi "API hong".
    reason: Optional[str] = None
    model: Optional[str] = None
    latency_ms: Optional[int] = None

    def __bool__(self):
        return self.ok


@dataclass
class _Breaker:
    threshold: int
    cooldown: float
    fails: int = 0
    open_until: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def is_open(self):
        with self.lock:
            return time.time() < self.open_until

    def record(self, ok):
        with self.lock:
            if ok:
                self.fails = 0
                self.open_until = 0.0
            else:
                self.fails += 1
                if self.fails >= self.threshold:
                    self.open_until = time.time() + self.cooldown

    def status(self):
        with self.lock:
            remain = max(0.0, self.open_until - time.time())
            return {"fails_lien_tiep": self.fails,
                    "dang_ngat": remain > 0,
                    "con_lai_giay": round(remain, 1)}


class LLMClient:
    """Giao dien chung. Khong bao gio nem exception ra ngoai."""

    name = "base"

    def available(self) -> bool:
        raise NotImplementedError

    def generate(self, prompt, images=None, tier="flash", timeout=None) -> LLMResult:
        raise NotImplementedError

    def status(self) -> dict:
        raise NotImplementedError


class NullClient(LLMClient):
    """Dung khi chua cau hinh gi. Moi loi goi deu tra ok=False ngay lap tuc."""

    name = "null"

    def __init__(self, reason):
        self.reason = reason

    def available(self):
        return False

    def generate(self, prompt, images=None, tier="flash", timeout=None):
        return LLMResult(ok=False, reason=self.reason, error=self.reason)

    def status(self):
        return {"provider": self.name, "kha_dung": False, "ly_do": self.reason}


class GeminiClient(LLMClient):
    """Google GenAI SDK hop nhat (`from google import genai`).

    SDK cu `google.generativeai` da ngung phat trien va khong duoc dung o day.
    """

    name = "gemini"

    def __init__(self, api_key=None, model_flash=None, model_pro=None,
                 timeout=None, max_retry=None):
        self.api_key = api_key if api_key is not None else cfg.LLM_API_KEY
        self.models = {
            "flash": model_flash if model_flash is not None else cfg.LLM_MODEL_FLASH,
            "pro": model_pro if model_pro is not None else cfg.LLM_MODEL_PRO,
        }
        self.timeout = cfg.LLM_TIMEOUT_SEC if timeout is None else timeout
        self.max_retry = cfg.LLM_MAX_RETRY if max_retry is None else max_retry
        self.breaker = _Breaker(cfg.LLM_BREAKER_THRESHOLD, cfg.LLM_COOLDOWN_SEC)
        self._client = None
        # init_error: KHONG the ket noi (thieu API key, thieu SDK). Chan ca
        # list_models().
        self.init_error = None
        # config_note: ket noi duoc nhung chua du de generate() (thieu model ID).
        # KHONG duoc chan list_models() -- do chinh la lenh de lay model ID.
        self.config_note = None

        self.warnings = [
            f"model {tier!r} = {mid!r} khong co so phien ban -- alias bi hot-swap, "
            f"hanh vi co the doi giua luc thi"
            for tier, mid in self.models.items()
            if mid and not looks_pinned(mid)
        ]

    # ------------------------------------------------------------------
    def _ensure(self):
        if self._client is not None or self.init_error is not None:
            return
        if not self.api_key:
            self.init_error = "chua dat GEMINI_API_KEY"
            return
        try:
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        except ImportError as e:
            self.init_error = f"thieu SDK google-genai: {e}"
        except Exception as e:
            self.init_error = f"{type(e).__name__}: {e}"

    def model_for(self, tier):
        return self.models.get(tier) or self.models.get("flash")

    def available(self, tier="flash"):
        self._ensure()
        return bool(self._client and self.model_for(tier))

    # ------------------------------------------------------------------
    def generate(self, prompt, images=None, tier="flash", timeout=None):
        t0 = time.time()
        model = self.model_for(tier)

        if not model:
            return LLMResult(ok=False, reason="chua_dat_model_id", error=(
                f"chua dat model ID cho tier {tier!r}. Dat GEMINI_MODEL_FLASH / "
                f"GEMINI_MODEL_PRO; xem retrieval/config.py"))

        self._ensure()
        if self._client is None:
            return LLMResult(ok=False, reason="chua_cau_hinh",
                             error=self.init_error, model=model)

        if self.breaker.is_open():
            return LLMResult(ok=False, reason="circuit_breaker", model=model,
                             error="API dang bi ngat tam thoi sau nhieu lan hong")

        parts = [prompt]
        for im in (images or []):
            loaded = self._load_image(im)
            if loaded is not None:
                parts.append(loaded)

        last = None
        for attempt in range(max(1, self.max_retry + 1)):
            try:
                resp = self._client.models.generate_content(
                    model=model,
                    contents=parts,
                    config={"http_options": {
                        "timeout": int((timeout or self.timeout) * 1000)}},
                )
                text = (getattr(resp, "text", None) or "").strip()
                if not text:
                    last = "model tra ve rong"
                    continue
                self.breaker.record(True)
                return LLMResult(ok=True, text=text, model=model,
                                 latency_ms=int((time.time() - t0) * 1000))
            except Exception as e:
                last = f"{type(e).__name__}: {e}"
                # 404 / thieu quyen la loi vinh vien -> thu lai vo ich
                if any(h in last.lower() for h in ("404", "not found",
                                                   "permission", "api key")):
                    break
                if attempt < self.max_retry:
                    time.sleep(0.5 * (2 ** attempt))

        self.breaker.record(False)
        low = (last or "").lower()
        reason = next((h for h in _DEGRADE_HINTS if h in low), "loi_api")
        return LLMResult(ok=False, reason=reason, error=last, model=model,
                         latency_ms=int((time.time() - t0) * 1000))

    @staticmethod
    def _load_image(im):
        if isinstance(im, str):
            try:
                from PIL import Image

                return Image.open(im)
            except Exception:
                return None
        return im

    # ------------------------------------------------------------------
    def list_models(self):
        """Liet ke model that su co tren tai khoan -- dung de chon ID dung."""
        self._ensure()
        if self._client is None:
            return {"ok": False, "error": self.init_error}
        try:
            out = []
            for m in self._client.models.list():
                mid = getattr(m, "name", "") or ""
                mid = mid.split("/")[-1]
                out.append({"id": mid, "pinned": looks_pinned(mid)})
            return {"ok": True, "models": sorted(out, key=lambda x: x["id"])}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    def status(self):
        self._ensure()
        return {
            "provider": self.name,
            "co_api_key": bool(self.api_key),
            "sdk": "google-genai" if self._client else None,
            "init_error": self.init_error,
            "config_note": self.config_note,
            "models": dict(self.models),
            "models_co_so_phien_ban": {t: looks_pinned(m)
                                       for t, m in self.models.items()},
            "canh_bao": self.warnings,
            "kha_dung": {t: self.available(t) for t in ("flash", "pro")},
            "circuit_breaker": self.breaker.status(),
            "timeout_sec": self.timeout,
        }


# --------------------------------------------------------------------------
_client = None
_client_lock = threading.Lock()


def get_client():
    """Client dung chung. Luon tra ve mot doi tuong dung duoc -- NullClient khi
    chua cau hinh -- de noi goi khong bao gio phai bat exception."""
    global _client
    with _client_lock:
        if _client is None:
            if cfg.LLM_PROVIDER == "gemini":
                _client = GeminiClient()
                # Thieu model ID KHONG phai loi ket noi: van phai mo duoc SDK de
                # chay list_models() -- lenh dung de lay chinh model ID do.
                missing = [n for n, v in (
                    ("GEMINI_MODEL_FLASH", cfg.LLM_MODEL_FLASH),
                    ("GEMINI_MODEL_PRO", cfg.LLM_MODEL_PRO),
                ) if not v]
                if missing:
                    _client.config_note = (
                        "chua dat: " + ", ".join(missing)
                        + " -- che do khong-LLM. Lay ID that bang: "
                        "python -m retrieval.llm_client --list")
            else:
                _client = NullClient(
                    f"chua ho tro LLM_PROVIDER={cfg.LLM_PROVIDER!r}")
        return _client


def reset_client():
    global _client
    with _client_lock:
        _client = None


def status():
    return get_client().status()


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Kiem tra cau hinh LLM")
    ap.add_argument("--list", action="store_true",
                    help="liet ke model that su co tren tai khoan")
    a = ap.parse_args()

    c = get_client()
    print(json.dumps(c.status(), ensure_ascii=False, indent=2))
    if a.list:
        # Chi can API KEY. Thieu model ID khong chan lenh nay -- day chinh la
        # lenh de lay model ID.
        r = c.list_models() if hasattr(c, "list_models") else {
            "ok": False, "error": f"provider {getattr(c, 'name', '?')} khong ho tro"}
        if r.get("ok"):
            pinned = [m for m in r["models"] if m["pinned"]]
            print(f"\n{len(r['models'])} model tren tai khoan "
                  f"({len(pinned)} co so phien ban).")
            print("Chon dong [pinned] -- alias tran bi hot-swap giua luc thi:\n")
            for m in r["models"]:
                print(f"  {'[pinned]' if m['pinned'] else '[alias] '} {m['id']}")
            if pinned:
                print("\nVi du dat bien:")
                print(f'  os.environ["GEMINI_MODEL_FLASH"] = "{pinned[0]["id"]}"')
                print(f'  os.environ["GEMINI_MODEL_PRO"]   = "{pinned[-1]["id"]}"')
        else:
            print("\nKhong liet ke duoc:", r.get("error"))
            if not cfg.LLM_API_KEY:
                print("  -> Dat GEMINI_API_KEY truoc. KHONG can model ID cho lenh nay.")
