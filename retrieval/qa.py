"""Q/A tren mot frame: VLM doc anh + OCR de tra loi cau hoi.

Khong con goi SDK truc tiep va khong con model ID nao trong file nay -- tat ca
di qua retrieval.llm_client. VLM hong thi tra ve cau tra loi suy tu OCR kem
`degraded=True`, KHONG nem exception va khong lam sap endpoint.
"""

from retrieval.llm_client import get_client

# Tier "flash": Q/A tren mot frame la tac vu nhanh, khong can model manh nhat.
QA_TIER = "flash"


def build_prompt(question, ocr_texts):
    ocr_context = ", ".join(t for t in ocr_texts if t) or "Khong co text nao tren hinh."
    return (
        "Ban dang xem mot khung hinh cat ra tu video tin tuc tieng Viet.\n"
        f"Text OCR doc duoc trong khung hinh (co the sai dau): [{ocr_context}]\n\n"
        f"Cau hoi: {question}\n\n"
        "Tra loi that ngan gon bang tieng Viet, chi dua tren nhung gi nhin thay "
        "trong anh va text OCR. Neu khong du thong tin, tra loi dung hai chu: "
        "Khong ro."
    )


def solve_qa(video_id, question, image_path, ocr_texts):
    """-> dict {answer, degraded, source, ...}. Khong bao gio nem exception.

    Fallback bat buoc: API tra 404 / timeout / het quota -> tra ve phan OCR thu
    duoc de nguoi dung tu doc, thay vi de endpoint sap.
    """
    client = get_client()
    result = client.generate(
        build_prompt(question, ocr_texts),
        images=[image_path] if image_path else None,
        tier=QA_TIER,
    )

    if result.ok:
        return {
            "answer": result.text,
            "degraded": False,
            "source": "vlm",
            "model": result.model,
            "latency_ms": result.latency_ms,
        }

    # --- che do khong-LLM -------------------------------------------------
    joined = ", ".join(t for t in ocr_texts if t)
    return {
        "answer": joined or None,
        "degraded": True,
        "source": "ocr_fallback",
        "reason": result.reason,
        "error": result.error,
        "model": result.model,
        "ghi_chu": (
            "VLM khong dung duoc nen day chi la text OCR doc duoc quanh frame, "
            "chua phai cau tra loi. Xem /diagnostics muc llm."
        ),
    }


def extract_best_frame(video_id, start_pts, end_pts, store):
    """Frame net nhat trong [start_pts, end_pts] -- mot dong cua store.meta.

    Chon theo lap_var (do net Laplacian) vi anh mo thi VLM doc sai chu tren man
    hinh. Khong co frame nao trong khoang thi lay frame gan start_pts nhat.
    """
    df = store.frames_of(video_id)
    if df.empty:
        return None

    subset = df[(df["pts_time"] >= start_pts) & (df["pts_time"] <= end_pts)]

    if subset.empty:
        return df.loc[(df["pts_time"] - start_pts).abs().idxmin()]

    if "lap_var" in subset.columns and subset["lap_var"].notna().any():
        return subset.loc[subset["lap_var"].idxmax()]
    return subset.iloc[len(subset) // 2]
