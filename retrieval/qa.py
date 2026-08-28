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

     KEM THEO hai rao can ma ban truoc khong co:
       · cau hoi phai duoc DICH sang tieng Anh truoc. SigLIP huan luyen chu yeu
         tieng Anh; dua thang tieng Viet vao thi cosine van ra so dep nhung vo
         nghia -- dung loai loi im lang ma he thong nay phai tranh.
       · cau hoi phai duoc CHIA cho vua 64 token. Do tren bo eval: cau hoi tieng
         Viet dai 194-536 token, tuc tokenizer vut di phan lon noi dung.
     Khong dich duoc -> BO HAN phan cham diem ngu nghia, chi con do net.

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


def build_prompt(question, ocr_by_frame, asr_texts=(), timestamps=None,
                 window=None):
    """Prompt cho VLM.

    ocr_by_frame: list[list[str]] -- OCR CUA TUNG khung hinh, dung thu tu thoi gian
    timestamps:   list[float] | None -- moc thoi gian tung khung hinh

    Bon dieu duoc sua so voi ban truoc, moi dieu deu co bang chung:

    1. Viet CO DAU. Ban truoc viet khong dau nhung lai nhet ocr_context CO DAU
       vao giua roi doi tra loi CO DAU -- ba tin hieu mau thuan trong cung mot
       prompt. Khong dau la quy uoc cho log console, khong phai cho prompt.

    2. DANH SO khung hinh va noi ro thu tu thoi gian. Cau hoi that cua BTC co
       dang "con so hien thi CUOI CUNG tren can" -- khong danh so thi model
       khong biet frame nao truoc frame nao sau.

    3. OCR nhom THEO TUNG FRAME. Ban truoc gop phang bang ", ".join() nen mat
       han thong tin chuoi nao thuoc frame nao -- dung thu quyet dinh cho cau
       hoi kieu "cuoi cung".

    4. Bo qua chu thuoc giao dien kenh. Bang chung tu log that: khi VLM hong,
       phan OCR do ra toan la "HTV9, 06:53:27, Binh Duong: Triet xoa duong day
       ca do bong da qua mang" -- ten dai, dong ho, dong chay chan man hinh.
    """
    ocr_by_frame = list(ocr_by_frame or [])
    n = len(ocr_by_frame)
    parts = []

    if n > 1:
        head = (f"Bạn đang xem {n} khung hình cắt ra từ một đoạn video tin tức "
                f"tiếng Việt, sắp theo thứ tự thời gian tăng dần "
                f"(khung hình 1 là sớm nhất, khung hình {n} là muộn nhất)")
    else:
        head = "Bạn đang xem một khung hình cắt ra từ video tin tức tiếng Việt"
    if window:
        head += f", trong khoảng {window[0]:.1f}s - {window[1]:.1f}s của video"
    parts.append(head + ".\n\n")

    parts.append("Text OCR đọc được trên hình (có thể sai chính tả, sai dấu, "
                 "hoặc nhầm ký tự giống nhau):\n")
    for i, texts in enumerate(ocr_by_frame, 1):
        ctx = ", ".join(t for t in texts if t) or "không có text"
        stamp = ""
        if timestamps and i - 1 < len(timestamps):
            stamp = f" ({timestamps[i - 1]:.1f}s)"
        parts.append(f"  Khung hình {i}{stamp}: [{ctx}]\n")

    asr = " ".join(t.strip() for t in asr_texts if t and t.strip())
    if asr:
        parts.append(f'\nLời nói trong đoạn này (ASR, có thể sai): "{asr}"\n')

    parts.append(f"\nCâu hỏi: {question}\n\n")
    parts.append(
        "Hướng dẫn trả lời:\n"
        "- Chỉ dựa vào hình ảnh và hai nguồn text ở trên. Không suy đoán từ "
        "kiến thức bên ngoài.\n"
        "- Khi các nguồn mâu thuẫn: với chữ và số hiện trên màn hình, tin vào "
        "HÌNH ẢNH trước, OCR chỉ là gợi ý. Với tên riêng và địa danh, ASR "
        "thường đáng tin hơn OCR.\n"
        "- Bỏ qua chữ thuộc giao diện kênh: tên đài, dòng chạy chân màn hình, "
        "đồng hồ, logo. Chỉ dùng chữ thuộc về cảnh đang được hỏi.\n"
        "- Nếu câu hỏi nhắc tới thứ tự thời gian (đầu tiên, cuối cùng, sau đó), "
        "dựa vào số thứ tự khung hình ở trên.\n"
        "- Trả lời bằng tiếng Việt CÓ DẤU, dưới 100 ký tự. Chỉ đưa ra đáp án, "
        "không nhắc lại câu hỏi, không giải thích, không dùng markdown. Hỏi số "
        "thì trả số kèm đơn vị. Hỏi tên thì trả tên.\n"
        "- Nếu suy ra được một đáp án hợp lý thì đưa ra, kể cả khi chưa chắc "
        'chắn hoàn toàn. Chỉ trả lời "Không rõ" khi thật sự không có manh mối nào.'
    )
    return "".join(parts)


def solve_qa(video_id, question, image_paths, ocr_by_frame, asr_texts=(),
             window=None, timestamps=None):
    """-> dict {answer, degraded, source, ...}. Khong bao gio nem exception.

    image_paths: mot duong dan hoac danh sach duong dan (theo thu tu thoi gian).

    Fallback bat buoc: API tra 404 / timeout / het quota -> tra ve phan OCR va
    ASR thu duoc de nguoi dung tu doc, thay vi de endpoint sap.
    """
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    images = [p for p in (image_paths or []) if p]

    # Nhan ca dang cu (mot danh sach chuoi phang) de noi goi cu khong vo.
    if ocr_by_frame and isinstance(ocr_by_frame[0], str):
        ocr_by_frame = [list(ocr_by_frame)]

    client = get_client()
    result = client.generate(
        build_prompt(question, ocr_by_frame, asr_texts,
                     timestamps=timestamps, window=window),
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
    joined = ", ".join(t for fr in ocr_by_frame for t in fr if t)
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
                encoder=None, question_en=None):
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

    qen = (question_en or "").strip()
    if not qen and question:
        # Dich (co bo nho dem tren dia nen tat dinh va khong ton mang lan hai).
        try:
            from retrieval.query import _translate

            qen = (_translate(question) or "").strip()
        except Exception:
            qen = ""

    if qen and encoder is not None:
        try:
            store.assert_encoder_matches(encoder)
            gid = sub["gidx"].to_numpy(dtype=int)
            # encode_query, KHONG phai encode_texts: tu dong chia cho vua 64 token.
            v = encoder.encode_query(qen)
            if v is not None:
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
                       encoder=None, question_en=None):
    """Mot frame duy nhat -- giu cho client cu goi theo ten ham nay."""
    got = pick_frames(video_id, start_pts, end_pts, store, n=1,
                      question=question, encoder=encoder, question_en=question_en)
    return got[0] if got else None
