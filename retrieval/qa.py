"""Q/A tren mot doan video: VLM doc NHIEU khung hinh + OCR + ASR de tra loi.

Khong con goi SDK truc tiep va khong con model ID nao trong file nay -- tat ca
di qua retrieval.llm_client. VLM hong thi tra ve cau tra loi suy tu OCR kem
`degraded=True`, KHONG nem exception va khong lam sap endpoint.

Ban truoc co ba diem yeu, ban nay sua het:

  1. Chi gui MOT anh. Cau hoi Q/A cua BTC thuong la "nguoi do dang lam gi",
     "co bao nhieu chiec xe di qua" -- khong tra loi duoc bang mot anh tinh.
     Gio gui mot day anh trong cua so thoi gian, danh so ro rang.

  2. Chon anh chi theo DO NET (lap_var). Anh net nhat trong doan thuong la
     canh tinh, chua chac lien quan den CAU HOI. Gio cham diem tong hop
     do net + do khop ngu nghia giua frame va cau hoi (SigLIP), khi co encoder.

  3. Khong dua ASR. Loi thoai la nguon tra loi manh nhat trong ban tin --
     ten nguoi, con so, dia danh deu duoc doc len chu khong hien tren hinh.
"""

import numpy as np

from retrieval.llm_client import get_client

# Tier "flash": Q/A tren mot doan ngan la tac vu nhanh, khong can model manh nhat.
QA_TIER = "flash"

# Trong so khi chon frame dua cho VLM: do khop cau hoi quan trong hon do net,
# nhung anh mo thi VLM doc sai chu tren man hinh nen do net van phai co mat.
W_RELEVANCE = 1.0
W_SHARPNESS = 0.35


def build_prompt(question, ocr_texts, asr_texts=(), n_frames=1, window=None):
    ocr_context = ", ".join(t for t in ocr_texts if t) or "khong co text nao tren hinh"
    asr_context = " ".join(t.strip() for t in asr_texts if t and t.strip())

    if n_frames > 1:
        head = (f"Ban dang xem {n_frames} khung hinh lien tiep cat ra tu mot doan "
                f"video tin tuc tieng Viet")
    else:
        head = "Ban dang xem mot khung hinh cat ra tu video tin tuc tieng Viet"
    if window:
        head += f" (khoang {window[0]:.1f}s - {window[1]:.1f}s)"

    parts = [head + ".\n"]
    parts.append(f"Text OCR doc duoc tren hinh (co the sai dau): [{ocr_context}]\n")
    if asr_context:
        parts.append(f"Loi noi trong doan nay (ASR, co the sai): \"{asr_context}\"\n")
    parts.append(f"\nCau hoi: {question}\n\n")
    parts.append(
        "Tra loi that ngan gon bang tieng Viet, chi dua tren nhung gi nhin thay "
        "trong anh va hai nguon text tren. Neu khong du thong tin de tra loi "
        "chac chan, tra loi dung hai chu: Khong ro."
    )
    return "".join(parts)


def solve_qa(video_id, question, image_paths, ocr_texts, asr_texts=(), window=None):
    """-> dict {answer, degraded, source, ...}. Khong bao gio nem exception.

    image_paths: mot duong dan hoac danh sach duong dan (theo thu tu thoi gian).

    Fallback bat buoc: API tra 404 / timeout / het quota -> tra ve phan OCR va
    ASR thu duoc de nguoi dung tu doc, thay vi de endpoint sap.
    """
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    images = [p for p in (image_paths or []) if p]

    client = get_client()
    result = client.generate(
        build_prompt(question, ocr_texts, asr_texts,
                     n_frames=len(images) or 1, window=window),
        images=images or None,
        tier=QA_TIER,
    )

    if result.ok:
        return {
            "answer": result.text,
            "degraded": False,
            "source": "vlm",
            "model": result.model,
            "n_frame": len(images),
            "latency_ms": result.latency_ms,
        }

    # --- che do khong-LLM -------------------------------------------------
    joined = ", ".join(t for t in ocr_texts if t)
    spoken = " ".join(t.strip() for t in asr_texts if t and t.strip())
    return {
        "answer": joined or spoken or None,
        "degraded": True,
        "source": "ocr_fallback" if joined else ("asr_fallback" if spoken else "none"),
        "reason": result.reason,
        "error": result.error,
        "model": result.model,
        "n_frame": len(images),
        "ghi_chu": (
            "VLM khong dung duoc nen day chi la text OCR/ASR doc duoc quanh "
            "frame, chua phai cau tra loi. Xem /diagnostics muc llm."
        ),
    }


# ---------------------------------------------------------------------------
def pick_frames(video_id, start_pts, end_pts, store, n=3, question=None,
                encoder=None):
    """Chon toi da n frame trong [start_pts, end_pts] de dua cho VLM.

    Xep hang bang do khop voi CAU HOI (SigLIP) cong do net. Chi lay do net nhu
    ban truoc thi hay chon canh tinh dep ma khong lien quan cau hoi.

    Khong co frame nao trong khoang -> lay frame gan start_pts nhat, de nguoi
    dung van nhan duoc mot cau tra loi thay vi mot loi.
    """
    df = store.frames_of(video_id)
    if df.empty:
        return []

    sub = df[(df["pts_time"] >= start_pts) & (df["pts_time"] <= end_pts)]
    if sub.empty:
        return [df.loc[(df["pts_time"] - start_pts).abs().idxmin()]]

    n = max(1, int(n))
    if len(sub) <= n:
        return [sub.iloc[i] for i in range(len(sub))]

    score = np.zeros(len(sub), dtype=np.float64)

    lap = sub["lap_var"].to_numpy(dtype=float)
    if np.isfinite(lap).any():
        lo, hi = np.nanmin(lap), np.nanmax(lap)
        if hi > lo:
            score += W_SHARPNESS * np.nan_to_num((lap - lo) / (hi - lo))

    if question and encoder is not None:
        try:
            store.assert_encoder_matches(encoder)
            gid = sub["gidx"].to_numpy(dtype=int)
            v = encoder.encode_texts([question])[0]
            rel = store.X[gid] @ v
            lo, hi = rel.min(), rel.max()
            if hi > lo:
                score += W_RELEVANCE * (rel - lo) / (hi - lo)
        except Exception:
            # Khong cham diem ngu nghia duoc thi van con do net -- khong hong ca Q/A.
            pass

    # Lay n frame diem cao nhat roi sap lai THEO THOI GIAN: thu tu thoi gian la
    # thong tin that voi cau hoi kieu "sau do chuyen gi xay ra".
    top = np.argsort(-score)[:n]
    return [sub.iloc[i] for i in sorted(top)]


def extract_best_frame(video_id, start_pts, end_pts, store, question=None,
                       encoder=None):
    """Mot frame duy nhat -- giu cho client cu goi theo ten ham nay."""
    got = pick_frames(video_id, start_pts, end_pts, store, n=1,
                      question=question, encoder=encoder)
    return got[0] if got else None
