#!/bin/bash

# Cấu hình API Key cho Claude LLM
export ANTHROPIC_API_KEY="YOUR_API_KEY_HERE"

# Dừng tất cả các tiến trình con khi dừng script chính (Ctrl+C)
cleanup() {
    echo -e "\n\033[0;33mĐang tắt các dịch vụ...\033[0m"
    kill $BACKEND_PID $SOCKET_PID $FRONTEND_PID 2>/dev/null
    exit
}
trap cleanup SIGINT SIGTERM

echo "========================================="
echo -e "\033[1;32mKHỞI CHẠY HỆ THỐNG TRUY VẤN VIDEO\033[0m"
echo "========================================="

# 1. Khởi động Backend Search (Port 8080)
echo "1. Đang khởi động Backend Search (Port 8080)..."
./venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8080 > backend.log 2>&1 &
BACKEND_PID=$!

# 2. Khởi động Socket App (Port 8081)
echo "2. Đang khởi động Socket App (Port 8081)..."
./venv/bin/python socket_app.py > socket.log 2>&1 &
SOCKET_PID=$!

# 3. Khởi động Frontend (Port 3000)
echo "3. Đang khởi động Frontend (Port 3000)..."
cd frontend
npm run dev > ../frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

echo "-----------------------------------------"
echo "Hệ thống đang khởi chạy trong nền:"
echo "- Log của Backend: backend.log"
echo "- Log của Socket: socket.log"
echo "- Log của Frontend: frontend.log"
echo "-----------------------------------------"
echo "Đang kiểm tra trạng thái khởi động..."
sleep 3

if ps -p $BACKEND_PID > /dev/null; then
    echo -e "  [\033[0;32mOK\033[0m] Backend Search đang hoạt động."
else
    echo -e "  [\033[0;31mLỖI\033[0m] Backend Search thất bại. Kiểm tra backend.log"
fi

if ps -p $SOCKET_PID > /dev/null; then
    echo -e "  [\033[0;32mOK\033[0m] Socket App đang hoạt động."
else
    echo -e "  [\033[0;31mLỖI\033[0m] Socket App thất bại. Kiểm tra socket.log"
fi

if ps -p $FRONTEND_PID > /dev/null; then
    echo -e "  [\033[0;32mOK\033[0m] Frontend đang hoạt động."
else
    echo -e "  [\033[0;31mLỖI\033[0m] Frontend thất bại. Kiểm tra frontend.log"
fi

# Đọc cổng thực tế mà Next.js đang chạy từ file log
FRONTEND_PORT=$(grep -oE "(localhost|0.0.0.0):[0-9]+" frontend.log | head -n 1 | cut -d: -f2)
if [ -z "$FRONTEND_PORT" ]; then
    FRONTEND_PORT=3000
fi

echo "-----------------------------------------"
echo -e "\033[1;36mTruy cập giao diện tại: http://localhost:${FRONTEND_PORT}\033[0m"
echo "Nhấn Ctrl+C để tắt toàn bộ dịch vụ cùng lúc."
echo "========================================="

# Chờ các tiến trình con
wait
