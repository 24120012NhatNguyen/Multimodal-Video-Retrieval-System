"""Chay backend TIM KIEM tren Kaggle (app.py :8080).

CHI tim kiem chay o day. Dap an, anh keyframe va Q/A chay o may local bang
socket_app.py -- xem ghi chu dau file do:

  A3  moi thu co trang thai phai o local, Kaggle session chet la mat sach.
  A2  Kaggle khong mount data/videos (65GB) nen khong trich anh duoc.

KHONG dan chuoi bi mat vao file nay -- no nam trong git. Dat bi mat bang mot
trong hai cach duoi, script tu doc duoc ca hai:

  1. Kaggle Secrets (nen dung).  Add-ons -> Secrets, them cac khoa:
         ANTHROPIC_API_KEY   NGROK_TOKEN
     Bi mat khong bao gio xuat hien trong notebook lan output.

  2. Dat bang bien moi truong trong mot cell TRUOC khi chay:
         import os
         os.environ["NGROK_TOKEN"]      = "..."
         os.environ["ANTHROPIC_API_KEY"] = "..."
     Roi:  %run kaggle_run.py
     Luu y: cell nay se duoc luu lai trong notebook. Notebook de private,
     hoac dung cach 1.

Copy URL ngrok in ra, dan vao frontend/.env.local (NEXT_PUBLIC_WEB_URL).
"""

import os
import subprocess
import sys

# ===================== CAU HINH ==========================================
# Doc tu Kaggle Secrets neu co, khong thi lay bien moi truong. Dan thang chuoi
# bi mat vao day se bi commit len git.
def secret(name, default=""):
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret(name)
    except Exception:
        return os.environ.get(name, default)


def mask(v):
    """Che bot de output cell khong lo bi mat.

    Output cua cell duoc LUU vao notebook. In nguyen token ra day thi bat ky ai
    xem duoc notebook la co token -- dung cai ma xac thuc tunnel dinh chan.
    """
    if not v:
        return "(chua dat)"
    return v[:3] + "..." + v[-2:] if len(v) > 8 else "***"


ARTIFACT_ROOT = os.environ.get(
    "ARTIFACT_ROOT", "/kaggle/input/artifacts-dataset/data/artifacts")

NGROK_TOKEN = secret("NGROK_TOKEN")

# --- Nha cung cap LLM ---------------------------------------------------
# Mac dinh ANTHROPIC, khop voi retrieval/config.py. File nay truoc day chi doc
# GEMINI_API_KEY trong khi config.py da chuyen sang doc ANTHROPIC_API_KEY --
# ket qua la tren Kaggle, LLM TAT HOAN TOAN ma khong bao gi ca: van tim kiem
# duoc nhung mat phan ra truy van, ma nguoi dung chi thay ket qua te di.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic")

ANTHROPIC_API_KEY = secret("ANTHROPIC_API_KEY")
GEMINI_API_KEY = secret("GEMINI_API_KEY")      # chi dung khi LLM_PROVIDER=gemini

# --- Model ID -----------------------------------------------------------
# ANTHROPIC: ID on dinh va tu no da la ban ghim (claude-haiku-4-5), khong co
# hau to ngay, nen KHONG phai dien tay. De trong -> retrieval/config.py lay mac
# dinh. Muon doi thi dat CLAUDE_MODEL_FLASH / CLAUDE_MODEL_PRO.
CLAUDE_MODEL_FLASH = os.environ.get("CLAUDE_MODEL_FLASH", "")
CLAUDE_MODEL_PRO = os.environ.get("CLAUDE_MODEL_PRO", "")

# GEMINI: DE TRONG CO Y. ID cua Google co chu ky khai tu ngan va alias khong so
# phien ban bi hot-swap giua buoi thi -- mot ID doan bua se tra 404 dung luc
# dang thi. Lay ID that tu chinh tai khoan:
#     GEMINI_API_KEY=... python -m retrieval.llm_client --list
# roi dien ID co [pinned].
GEMINI_MODEL_PRO = os.environ.get("GEMINI_MODEL_PRO", "")
GEMINI_MODEL_FLASH = os.environ.get("GEMINI_MODEL_FLASH", "")

# Index object cho /panel. Mac dinh tim o $ARTIFACT_ROOT/object_index.npz --
# tuc la no di kem luon khi ban upload thu muc artifacts len Kaggle. Chi phai
# dat bien nay khi de file o cho khac. Xem retrieval/objects.py.
OBJECT_INDEX = os.environ.get("OBJECT_INDEX", "")
# =========================================================================

print("Installing requirements...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                "requirements.txt"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyngrok",
                "uvicorn"], check=True)

# 2. Thiết lập biến môi trường
os.environ["ARTIFACT_ROOT"] = ARTIFACT_ROOT
os.environ["LLM_PROVIDER"] = LLM_PROVIDER
for _k, _v in (("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY),
               ("GEMINI_API_KEY", GEMINI_API_KEY),
               ("CLAUDE_MODEL_FLASH", CLAUDE_MODEL_FLASH),
               ("CLAUDE_MODEL_PRO", CLAUDE_MODEL_PRO),
               ("GEMINI_MODEL_FLASH", GEMINI_MODEL_FLASH),
               ("GEMINI_MODEL_PRO", GEMINI_MODEL_PRO),
               ("OBJECT_INDEX", OBJECT_INDEX)):
    if _v:
        os.environ[_k] = _v

# --- Kiem tra du lieu truoc khi mo tunnel -------------------------------
# In bang mot lenh duy nhat + flush: Kaggle gom dem stdout, nhieu lenh print
# lien tiep co the bi nuot dong hoac tron thu tu voi log cua thu vien khac.
def say(*lines):
    print("\n".join(str(x) for x in lines), flush=True)


say("", f"ARTIFACT_ROOT = {ARTIFACT_ROOT}")
if not os.path.isdir(ARTIFACT_ROOT):
    say("  KHONG TON TAI. Tim thu muc dung bang:",
        "     import os; os.listdir('/kaggle/input')")
    sys.exit(1)

subdirs = sorted(d for d in os.listdir(ARTIFACT_ROOT)
                 if os.path.isdir(os.path.join(ARTIFACT_ROOT, d)))


def _count(pack, sub):
    d = os.path.join(ARTIFACT_ROOT, pack, sub)
    return len(os.listdir(d)) if os.path.isdir(d) else 0


packs = [d for d in subdirs if os.path.isdir(os.path.join(ARTIFACT_ROOT, d, "features"))]
n_feat = sum(_count(p, "features") for p in packs)
n_asr = sum(_count(p, "asr") for p in packs)
n_ocr = sum(_count(p, "ocr") for p in packs)

say(f"  {len(packs)} pack, {n_feat} features, {n_asr} asr, {n_ocr} ocr")
if n_feat == 0:
    say("  Khong co file features nao -> tim kiem se tra ve rong. Dung lai.")
    sys.exit(1)

# --- Thu muc phu: thieu cai nao thi kenh do CHET AM THAM ----------------
# Moi dong duoi la mot chuc nang cu the se hong, khong phai canh bao chung chung.
side = []
for label, rel, hong in (
    ("metadata", os.path.join("media-info-aic25-b1", "media-info"),
     "kenh meta/meta_fold tat -> tim theo ten dai, tieu de, tu khoa khong chay"),
    ("object", os.path.join("objects-aic25-b1", "objects"),
     "panel object va /keyframe_context muc objects tat"),
    ("map-keyframes", "map-keyframes",
     "object khong co moc thoi gian de bac cau -> kenh object tat du co du lieu"),
):
    d = os.path.join(ARTIFACT_ROOT, rel)
    alt = os.path.join(ARTIFACT_ROOT, os.path.basename(rel))
    got = d if os.path.isdir(d) else (alt if os.path.isdir(alt) else None)
    side.append((label, got, len(os.listdir(got)) if got else 0, hong))

for label, got, n, hong in side:
    if got:
        say(f"  {label:14} = {n} file")
    else:
        say(f"  [thieu] {label} -> {hong}")

# --- Video co metadata nhung KHONG co features --------------------------
# Tim khong ra nhung video nay la DUNG, khong phai loi he thong -- nhung phai
# biet truoc con so, chu khong phat hien giua buoi thi.
_meta_dir = next((g for l, g, _, _ in side if l == "metadata" and g), None)
if _meta_dir:
    n_meta = len([f for f in os.listdir(_meta_dir) if f.endswith(".json")])
    if n_meta > n_feat:
        say("",
            f"  [chu y] {n_meta} video co metadata nhung chi {n_feat} co features.",
            f"          {n_meta - n_feat} video ({100*(n_meta-n_feat)/n_meta:.0f}%) "
            f"KHONG tim ra duoc bang kenh thi giac.",
            "          Bo artifacts tren Kaggle cu hon bo o may local? Xem "
            "GET /diagnostics muc metadata.")

say("",
    f"  NGROK_TOKEN       = {mask(NGROK_TOKEN)}",
    f"  LLM_PROVIDER      = {LLM_PROVIDER}")
if LLM_PROVIDER == "anthropic":
    say(f"  ANTHROPIC_API_KEY = {mask(ANTHROPIC_API_KEY)}")
else:
    say(f"  GEMINI_API_KEY    = {mask(GEMINI_API_KEY)}",
        f"  GEMINI_MODEL_PRO   = {GEMINI_MODEL_PRO or '(chua dat)'}",
        f"  GEMINI_MODEL_FLASH = {GEMINI_MODEL_FLASH or '(chua dat)'}")

# --- Index object cho /panel -------------------------------------------
# Thieu thi /panel tra ve rong. Day la che do xuong cap CO Y (tim kiem van
# chay), nhung phai noi to o day chu khong de nguoi dung tu phat hien.
from retrieval.objects import ObjectIndex as _OIX

_oix = _OIX()
if _oix.ok:
    say(f"  Index object      = {len(_oix.entities)} lop, "
        f"{len(_oix.det_ent)} detection ({_oix.path})")
else:
    say("",
        "  [chu y] KHONG CO INDEX OBJECT -> /panel se tra ve RONG.",
        f"          {_oix.error}",
        "          Cach sua: o MAY LOCAL chay",
        "              python -m retrieval.objects build",
        "          roi chep data/artifacts/object_index.npz vao thu muc "
        "artifacts",
        "          truoc khi upload len Kaggle (hoac dat bien OBJECT_INDEX).")

# --- Kiem tra LLM that su tra loi duoc ---------------------------------
# Co API key khong dong nghia goi duoc: thieu SDK, sai key, het quota, sai ten
# model deu hong theo kieu khac nhau. Thu mot lan o day, con hon phat hien giua
# buoi thi.
say("", "Kiem tra LLM...")
from retrieval.llm_client import get_client as _get_llm

_llm = _get_llm()
_probe = _llm.generate(
    "Chi in ra dung hai ky tu OK, khong them bat ky chu nao khac.",
    tier="flash")
if _probe.ok:
    say(f"  LLM san sang: {_probe.model} ({_probe.latency_ms}ms) "
        f"-> {(_probe.text or '').strip()[:30]!r}")
else:
    _err = str(_probe.error or "")
    say(f"  LLM KHONG DUNG DUOC: {_probe.reason} -- {_err}")
    if "No module named" in _err or "thieu SDK" in _err:
        # Thieu goi thi sua duoc ngay tai cho, khong phai chay lai ca notebook.
        say("  -> Thieu goi Python. Chay trong mot cell roi %run lai file nay:",
            f"         !pip install -q anthropic",
            "     (requirements.txt da co san tu ban nay tro di.)")
    say("  He thong chay CHE DO KHONG-LLM: van tim duoc bang SigLIP + BM25 +",
        "  dich may, chi mat phan ra truy van va Q/A. Day khong phai loi chet.")


# --- Nap model TRUOC khi mo tunnel --------------------------------------
# Mo cong roi moi tai model nghia la nhung truy van dau tien chay khong co kenh
# thi giac, va nguoi dung khong he biet -- ho chi thay ket qua te.
print("\nNap SigLIP (lan dau tai ~3.5GB, cac lan sau doc tu cache)...")
import time as _t
_t0 = _t.time()
try:
    from retrieval import service as _svc
    _info = _svc.preload(strict=True)
    print(f"  SigLIP san sang: {_info['dim']} chieu tren {_info['device']} "
          f"({_t.time()-_t0:.0f}s)")
except Exception as _e:
    print(f"  KHONG NAP DUOC SigLIP: {_e}")
    print("  Dung lai -- mo backend luc nay se cho ket qua chi xep bang BM25.")
    sys.exit(1)

# --- Mo tunnel ----------------------------------------------------------
from pyngrok import ngrok

if NGROK_TOKEN:
    ngrok.set_auth_token(NGROK_TOKEN)
public_url = ngrok.connect(8080).public_url

print("\n" + "=" * 68)
print(f"BACKEND TIM KIEM (Kaggle): {public_url}")
print("=" * 68)
print("Dan vao frontend/.env.local roi khoi dong lai `npm run dev`:")
print(f"  NEXT_PUBLIC_WEB_URL={public_url}")
print("\nO MAY LOCAL nho chay them server trang thai + anh:")
print("  python socket_app.py")
print("=" * 68 + "\n")

# --- Chay server NGAY TRONG tien trinh nay ------------------------------
# KHONG dung subprocess.run(["uvicorn", ...]).
#
# Ly do, do duoc tu log that: subprocess sinh mot tien trinh Python MOI, no
# import lai app.py va nap LAI tu dau ca ArtifactStore lan SigLIP. Trong khi do
# tien trinh cha van song (dang cho con chay xong) va van giu ban cua no. Ket
# qua: HAI ban cung ton tai suot phien.
#
#     store.X  = 154.640 x 1152 float32 = 713 MB moi ban  -> thua 713 MB RAM
#     SigLIP   = mot ban thua nam tren GPU suot buoi
#     va toan bo 88s nap lan dau la nem di
#
# Chay trong cung tien trinh thi `service` da nho san ket qua trong `_state`,
# nen `import app` dung lai dung nhung doi tuong vua kiem tra o tren -- nap MOT
# lan, va thu da kiem chinh la thu dem ra phuc vu.
import uvicorn

from app import app as _fastapi_app

uvicorn.run(_fastapi_app, host="0.0.0.0", port=8080)
