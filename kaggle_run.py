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

print(f"Checking ARTIFACT_ROOT: {ARTIFACT_ROOT}")

# --- Kiem tra du lieu truoc khi mo tunnel -------------------------------
print(f"\nARTIFACT_ROOT = {ARTIFACT_ROOT}")
if not os.path.isdir(ARTIFACT_ROOT):
    print("  KHONG TON TAI. Tim thu muc dung bang:")
    print("     import os; os.listdir('/kaggle/input')")
    sys.exit(1)

packs = sorted(d for d in os.listdir(ARTIFACT_ROOT)
               if os.path.isdir(os.path.join(ARTIFACT_ROOT, d)))
n_feat = sum(len(os.listdir(os.path.join(ARTIFACT_ROOT, p, "features")))
             for p in packs
             if os.path.isdir(os.path.join(ARTIFACT_ROOT, p, "features")))
print(f"  {len(packs)} thu muc, {n_feat} file features")
if n_feat == 0:
    print("  Khong co file features nao -> tim kiem se tra ve rong. Dung lai.")
    sys.exit(1)

print(f"  NGROK_TOKEN       = {mask(NGROK_TOKEN)}")
print(f"  LLM_PROVIDER      = {LLM_PROVIDER}")
if LLM_PROVIDER == "anthropic":
    print(f"  ANTHROPIC_API_KEY = {mask(ANTHROPIC_API_KEY)}")
else:
    print(f"  GEMINI_API_KEY    = {mask(GEMINI_API_KEY)}")
    print(f"  GEMINI_MODEL_PRO   = {GEMINI_MODEL_PRO or '(chua dat)'}")
    print(f"  GEMINI_MODEL_FLASH = {GEMINI_MODEL_FLASH or '(chua dat)'}")

# --- Index object cho /panel -------------------------------------------
# Thieu thi /panel tra ve rong. Day la che do xuong cap CO Y (tim kiem van
# chay), nhung phai noi to o day chu khong de nguoi dung tu phat hien.
from retrieval.objects import ObjectIndex as _OIX

_oix = _OIX()
if _oix.ok:
    print(f"  Index object      = {len(_oix.entities)} lop, "
          f"{len(_oix.det_ent)} detection ({_oix.path})")
else:
    print("\n  [chu y] KHONG CO INDEX OBJECT -> /panel se tra ve RONG.")
    print(f"          {_oix.error}")
    print("          Dat object_index.npz vao thu muc artifacts truoc khi upload")
    print("          len Kaggle, hoac dat OBJECT_INDEX tro toi file do.")
    print("          Dung index: python -m retrieval.objects build")

# --- Kiem tra LLM that su tra loi duoc ---------------------------------
# Co API key khong dong nghia goi duoc: sai key, het quota, sai ten model deu
# ra 4xx. Thu mot lan o day, con hon phat hien giua buoi thi.
print("\nKiem tra LLM...")
from retrieval.llm_client import get_client as _get_llm

_llm = _get_llm()
_probe = _llm.generate("Tra loi dung mot tu: OK", tier="flash")
if _probe.ok:
    print(f"  LLM san sang: {_probe.model} ({_probe.latency_ms}ms) "
          f"-> {(_probe.text or '').strip()[:30]!r}")
else:
    print(f"  LLM KHONG DUNG DUOC: {_probe.reason} -- {_probe.error}")
    print("  He thong chay CHE DO KHONG-LLM: van tim duoc bang SigLIP + BM25 +")
    print("  dich may, chi mat phan ra truy van va Q/A. Day khong phai loi chet.")


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

subprocess.run(["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"])
