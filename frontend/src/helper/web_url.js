// Cau hinh endpoint. Doc tu bien moi truong NEXT_PUBLIC_* -> doi tunnel khong
// phai sua code nua. Xem frontend/.env.local.example.
//
// Kien truc (theo checklist A2/A3):
//
//   web_url    Kaggle, app.py :8080     tim kiem. Stateless, co GPU.
//   socket_url LOCAL, socket_app.py     dap an + ignore + export (co trang thai)
//   media_url  LOCAL, socket_app.py     anh keyframe trich tu data/videos
//
// Anh PHAI lay o local: Kaggle chi mount artifacts, khong co data/videos (65GB).

const env = (key, fallback) => {
  const v = process.env[key];
  return v && v.trim() ? v.trim().replace(/\/+$/, "") : fallback;
};

// Backend tim kiem tren Kaggle. Doi moi lan mo tunnel moi.
export const web_url = env("NEXT_PUBLIC_WEB_URL", "http://localhost:8080");

// Server trang thai chay ngay tren may nay.
export const socket_url = env("NEXT_PUBLIC_SOCKET_URL", "http://localhost:8081");

// Anh keyframe: mac dinh cung server voi socket_url.
export const media_url = env("NEXT_PUBLIC_MEDIA_URL", socket_url);

// Server nop bai cua BTC.
export const server = env("NEXT_PUBLIC_SUBMIT_URL", `${socket_url}/submit`);
export const session = env("NEXT_PUBLIC_SESSION", "1");

// Token xac thuc tunnel (muc C5). De trong = tat xac thuc.
export const api_token = env("NEXT_PUBLIC_API_TOKEN", "");

// ---------------------------------------------------------------------------
/** Header dung cho moi request len backend. */
export function apiHeaders(extra) {
  const h = {
    "ngrok-skip-browser-warning": "69420",
    "Content-Type": "application/json",
    ...(extra || {}),
  };
  if (api_token) h["x-aic-token"] = api_token;
  return new Headers(h);
}

/** Bien duong dan anh tra ve tu backend thanh URL day du.
 *
 * Backend tra ve duong dan TUONG DOI:
 *   "/keyframe/L24_V007/012450.jpg"  -> server media (local)
 *   "/static/images/Keyframes/..."   -> anh cu, Next.js tu phuc vu tu public/
 *
 * Khong xu ly buoc nay thi trinh duyet se di tim /keyframe/... ngay tren
 * Next.js va nhan 404 -- day la nguyen nhan vo anh toan bo.
 */
export function imageUrl(path) {
  if (!path) return "";
  if (/^https?:\/\//i.test(path)) return path;
  if (path.startsWith("/keyframe/")) {
    const sep = api_token ? `?token=${encodeURIComponent(api_token)}` : "";
    return `${media_url}${path}${sep}`;
  }
  return path;
}
