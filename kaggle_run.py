import os
import subprocess
import sys
import time
from pyngrok import ngrok

# 1. Cài đặt các thư viện cần thiết
print("Installing requirements...")
subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "pyngrok", "uvicorn"], check=True)

# 2. Thiết lập biến môi trường
os.environ["ARTIFACT_ROOT"] = "/kaggle/input/artifacts-dataset/data/artifacts"
os.environ["GEMINI_API_KEY"] = "YOUR_GEMINI_API_KEY_HERE"
os.environ["GEMINI_MODEL_PRO"] = "gemini-2.5-pro"
os.environ["GEMINI_MODEL_FLASH"] = "gemini-3.7-flash-lite"

# 3. Mở Ngrok Tunnel
NGROK_TOKEN = "3IBfV7LWs5xyMp3ZpgrgdBB0n4a_4SZfZ1AaV3naoVUiMk1Uc"
ngrok.set_auth_token(NGROK_TOKEN)
public_url = ngrok.connect(8080).public_url
print(f"===========================================================")
print(f"KAGGLE BACKEND IS LIVE AT: {public_url}")
print(f"Hãy copy URL trên dán vào cấu hình của Frontend/Local Server!")
print(f"===========================================================")

# 4. Chạy app.py bằng Uvicorn
subprocess.run(["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"])
