"""Chay backend TIM KIEM tren Kaggle (app.py :8080).

CHI tim kiem chay o day. Dap an, anh keyframe va Q/A chay o may local bang
socket_app.py -- xem ghi chu dau file do:

  A3  moi thu co trang thai phai o local, Kaggle session chet la mat sach.
  A2  Kaggle khong mount data/videos (65GB) nen khong trich anh duoc.

KHONG dan chuoi bi mat vao file nay -- no nam trong git. Dat bi mat bang mot
trong hai cach duoi, script tu doc duoc ca hai:

  1. Kaggle Secrets (nen dung).  Add-ons -> Secrets, them cac khoa:
         GEMINI_API_KEY   NGROK_TOKEN   AIC_API_TOKEN
     Bi mat khong bao gio xuat hien trong notebook lan output.

  2. Dat bang bien moi truong trong mot cell TRUOC khi chay:
         import os
         os.environ["NGROK_TOKEN"]   = "..."
         os.environ["GEMINI_API_KEY"] = "..."
         os.environ["AIC_API_TOKEN"]  = "..."
         os.environ["GEMINI_MODEL_PRO"]   = "..."
         os.environ["GEMINI_MODEL_FLASH"] = "..."
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
GEMINI_API_KEY = secret("GEMINI_API_KEY")

# Token xac thuc tunnel (muc C5). URL ngrok la public -- khong co token thi doi
# khac doan ra URL la goi vao duoc. PHAI trung voi NEXT_PUBLIC_API_TOKEN o FE.
AIC_API_TOKEN = secret("AIC_API_TOKEN", "doi-chuoi-nay-di")

# --- Model ID -----------------------------------------------------------
# DE TRONG CO Y. Mot ID doan bua se tra 404 giua luc thi -- dung cai loi ma muc
# A1 dang muon chan. Lay ID that tu chinh tai khoan cua ban:
#
#     GEMINI_API_KEY=... python -m retrieval.llm_client --list
#
# roi dien ID co [pinned] (co hau to so phien ban, vd ...-001). Alias khong co
# so phien ban bi hot-swap va co the doi hanh vi giua buoi thi.
# Chua dien -> he thong chay che do khong-LLM: van tim kiem duoc bang BM25 +
# dich may, chi mat phan ra truy van bang LLM.
GEMINI_MODEL_PRO = os.environ.get("GEMINI_MODEL_PRO", "")
GEMINI_MODEL_FLASH = os.environ.get("GEMINI_MODEL_FLASH", "")
# =========================================================================

print("Installing requirements...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                "requirements.txt"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyngrok",
                "uvicorn"], check=True)

# 2. Thiết lập biến môi trường
os.environ["ARTIFACT_ROOT"] = ARTIFACT_ROOT
if GEMINI_API_KEY:
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
if GEMINI_MODEL_PRO:
    os.environ["GEMINI_MODEL_PRO"] = GEMINI_MODEL_PRO
if GEMINI_MODEL_FLASH:
    os.environ["GEMINI_MODEL_FLASH"] = GEMINI_MODEL_FLASH

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

print(f"  GEMINI_API_KEY  = {mask(GEMINI_API_KEY)}")
print(f"  NGROK_TOKEN     = {mask(NGROK_TOKEN)}")
print(f"  AIC_API_TOKEN   = {mask(AIC_API_TOKEN)}")
print(f"  GEMINI_MODEL_PRO   = {GEMINI_MODEL_PRO or '(chua dat)'}")
print(f"  GEMINI_MODEL_FLASH = {GEMINI_MODEL_FLASH or '(chua dat)'}")

if not GEMINI_MODEL_FLASH and not GEMINI_MODEL_PRO:
    print("\n  [chu y] Chua dat GEMINI_MODEL_*. He thong chay che do khong-LLM:")
    print("          van tim duoc bang BM25 + dich may, mat phan ra truy van.")
    print("          Lay ID that: python -m retrieval.llm_client --list")

if AIC_API_TOKEN == "doi-chuoi-nay-di":
    print("\n  [chu y] AIC_API_TOKEN dang la gia tri mau. Doi di truoc khi thi.")

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
print(f"  NEXT_PUBLIC_API_TOKEN=<token ban da dat>   ({mask(AIC_API_TOKEN)})")
print("\nO MAY LOCAL nho chay them server trang thai + anh:")
print("  AIC_API_TOKEN=<cung token do> python socket_app.py")
print("=" * 68 + "\n")

subprocess.run(["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"])
