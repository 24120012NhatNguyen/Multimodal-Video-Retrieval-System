"""Xac thuc token tinh tren tunnel (muc C5 cua checklist).

URL ngrok/cloudflare la public: bat ky ai doan ra cung goi duoc vao backend
dang chay tren Kaggle. Mot token tinh trong header la du cho mot cuoc thi keo
dai vai gio -- khong can OAuth.

    export AIC_API_TOKEN=chuoi-bi-mat-nao-do

Khong dat bien nay -> xac thuc TAT (tien cho luc phat trien o localhost).
Server se ghi log canh bao de khong ai vo tinh mo cong ra Internet ma khong biet.
"""

import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from utils.logger_config import get_logger

logger = get_logger(__name__)

HEADER = "x-aic-token"

# Nhung duong dan luon cho qua: preflight CORS va health check phai goi duoc
# truoc khi FE biet token, con /docs de tu kiem tra bang tay.
OPEN_PATHS = ("/health", "/docs", "/openapi.json", "/redoc", "/docs/oauth2-redirect")


def get_token():
    return os.environ.get("AIC_API_TOKEN") or ""


class TokenAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, token):
        super().__init__(app)
        self.token = token

    async def dispatch(self, request, call_next):
        # OPTIONS la preflight CORS, trinh duyet khong gan header tuy bien vao.
        if request.method == "OPTIONS":
            return await call_next(request)

        path = request.url.path
        if path in OPEN_PATHS or path.startswith("/socket.io"):
            return await call_next(request)

        sent = request.headers.get(HEADER) or request.query_params.get("token") or ""
        if sent != self.token:
            return JSONResponse(
                {"error": f"thieu hoac sai header {HEADER}", "status_code": 401},
                status_code=401,
            )
        return await call_next(request)


def install_auth(app):
    """Gan middleware neu co AIC_API_TOKEN. Tra ve True neu da bat."""
    token = get_token()
    if not token:
        logger.warning(
            "AIC_API_TOKEN chua dat -> KHONG xac thuc. Chap nhan duoc o "
            "localhost, nhung neu dang mo tunnel ra Internet thi bat ky ai co "
            "URL deu goi duoc. Dat AIC_API_TOKEN de bat."
        )
        return False
    app.add_middleware(TokenAuthMiddleware, token=token)
    logger.info("Xac thuc token BAT (header %s)", HEADER)
    return True
