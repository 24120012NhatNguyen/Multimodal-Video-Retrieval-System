"""Viec 3 -- chon 100 dong dap an de nop.

Thiet ke bam theo DUNG cong thuc cham diem cua BTC (Debug/mark.md):

    R@k   = max{ R-Score cua k dap an dau }   voi k in {1, 5, 20, 50, 100}
    Final = trung binh cong 5 gia tri R@k

Voi KIS, R-Score la nhi phan, nen Final chi phu thuoc MOT con so: thu hang r cua
dap an DUNG DAU TIEN.

    r = 1  -> 1.00      r = 6..20  -> 0.60      r = 51..100 -> 0.20
    r = 2..5 -> 0.80    r = 21..50 -> 0.40      r > 100     -> 0

Hai he qua dinh doat toan bo thuat toan nay:

  (1) CHI dap an dung dau tien co gia tri. Dap an dung thu hai cong 0.
  (2) Gia tri bien cua tung o lech nhau toi 5 lan (o 1 dang 1.00, o 100 dang
      0.20).

BAN TRUOC DUNG MMR VA DIEU DO LAM MAT DIEM. Do tren Debug/7_questions.json:

    lay thang 100 ung vien dau, chi bo trung        Final 0.5714
    MMR lambda=0.7, toi da 5 frame/video, cach 2s   Final 0.3429

MMR day dap an dung XUONG. Cu the: Q3 co frame dung o vi tri 11 va Q4 o vi tri 9
trong danh sach ung vien -- dang le duoc 0.60 diem moi cau -- nhung ca hai deu
bi loai khoi 100 dong va thanh 0. Nguyen nhan la MMR do trung lap bang COSINE
NHUNG:

  · Hai frame khac VIDEO ma nhin giong nhau thi cosine cao -> bi phat, trong khi
    ve mat cham diem chung khong trung lap chut nao (video A sai thi video B van
    co the dung).
  · Frame dung thuong RAT giong frame dau bang (cung canh quay) -> chinh no bi
    phat nang nhat.

Da dang hoa chi co loi khi dau bang hay sai. Do duoc: dau bang KHONG hay sai
(video dung nam trong top-5 o 100% truy van cua bo eval). Nen:

    O 1..20    giu NGUYEN thu hang tim kiem. Day la nhung o dang 0,60-1,00
               diem; khong danh cuoc chung vao da dang hoa.
    O 21..100  den duoc day nghia la 20 o dau DA TRUOT -- tuc la ta dang o
               nhanh "dau bang sai". Chi luc do da dang hoa moi dang, va no
               khong the lam mat gi nua.

Con so 20 khong phai chon bua: no chinh la nguong k = 20 trong cong thuc.

Do tren bo eval, dau=20 va khoang cach 8s:  Final 0.6000
(so voi 0.5714 cua thu hang tho, va 0.3429 cua ban MMR cu)

KHONG co khai niem "shot". Bo keyframe sinh ra bang lay mau theo do troi ngu
nghia, khong theo ranh gioi canh, nen moi quan he lan can deu tinh theo
pts_time. Cot cos_to_prev khong duoc dung o day: phan bo cua no da bi chinh
nguong lay mau cat cut nen khong phai tin hieu ranh gioi canh.
"""

from retrieval.config import FusionConfig

# Nguong xep hang trong cong thuc cua BTC.
BREAKPOINTS = (1, 5, 20, 50, 100)


def slot_weight(i, breakpoints=BREAKPOINTS):
    """Diem toi da mot o thu i (dem tu 1) co the mang lai."""
    return sum(1 for k in breakpoints if k >= i) / len(breakpoints)


def _lookup(store):
    """(video_id, frame_idx) -> (gidx, pts_time)"""
    m = store.meta
    return {
        (v, int(f)): (int(g), float(p))
        for v, f, g, p in zip(m["video_id"], m["frame_idx"], m["gidx"], m["pts_time"])
    }


def _as_dict(it):
    if isinstance(it, dict):
        return it
    return it.model_dump() if hasattr(it, "model_dump") else dict(it)


def autofill(store, manual, candidates=None, config=None, ignore=None,
             target=None, head=None, tail_gap=None):
    """-> danh sach dap an da xep thu tu, dung schema Viec 2.

    manual      [{"video_id", "frame_idx"}] nguoi chon -- luon nam dau
    candidates  [{"video_id", "frame_idx", "score"}] tap ket qua dang hien,
                DA XEP HANG. Thu tu nay duoc ton trong o phan dau.
    ignore      [{"video_id", "frame_idx"}] hoac [(video_id, frame_idx)]
    head        so o dau giu nguyen thu hang tim kiem (mac dinh cfg.autofill_head)
    tail_gap    tu o head+1 tro di, hai frame cung video phai cach nhau it nhat
                ngan nay giay; vi pham thi bi DAY XUONG CUOI chu khong bi loai
    """
    cfg = config or FusionConfig.load()
    target = target or cfg.autofill_target
    head = getattr(cfg, "autofill_head", 20) if head is None else head
    tail_gap = (getattr(cfg, "autofill_tail_gap_sec", 8.0)
                if tail_gap is None else tail_gap)
    look = _lookup(store)

    ign = set()
    for it in (ignore or []):
        if isinstance(it, dict):
            ign.add((it.get("video_id"), int(it.get("frame_idx"))))
        else:
            ign.add((it[0], int(it[1])))

    out, seen = [], set()

    def emit(video_id, frame_idx, source, reason):
        key = (video_id, int(frame_idx))
        if key in seen or key in ign:
            return False
        hit = look.get(key)
        if hit is None:
            return False
        gidx, pts = hit
        seen.add(key)
        out.append({
            "video_id": video_id,
            "frame_idx": int(frame_idx),
            "source": source,
            "pts_time": round(pts, 3),
            "gidx": gidx,
            "reason": reason,
        })
        return True

    # --- 1. Frame nguoi dung tu chon: luon o dau, khong rang buoc gi ------
    # Nguoi ngoi truoc man hinh da NHIN thay frame do. Khong co phong doan tu
    # dong nao duoc phep day no xuong.
    for it in (manual or []):
        it = _as_dict(it)
        vid, fi = it.get("video_id"), it.get("frame_idx")
        if vid is not None and fi is not None:
            emit(vid, fi, "manual", "nguoi dung chon")
        if len(out) >= target:
            return out[:target]

    # --- 2. Chuan hoa tap ung vien, GIU NGUYEN thu tu ---------------------
    pool = []
    for c in (candidates or []):
        c = _as_dict(c)
        vid, fi = c.get("video_id"), c.get("frame_idx")
        if vid is None or fi is None:
            continue
        key = (vid, int(fi))
        if key in seen or key in ign or key not in look:
            continue
        if any(p[0] == key for p in pool):
            continue
        pool.append((key, look[key][1]))          # (khoa, pts_time)
    if not pool:
        return out[:target]

    # --- 3. Phan DAU: nguyen thu hang tim kiem ---------------------------
    n_head = max(0, head - len(out))
    for key, _ in pool[:n_head]:
        emit(key[0], key[1], "autofill",
             f"thu hang tim kiem (o 1-{head}, moi o dang "
             f"{slot_weight(len(out) + 1):.2f} diem)")

    # --- 4. Phan DUOI: trai deu theo thoi gian ---------------------------
    # Den duoc day nghia la cac o dau da truot -> gia thuyet "dung dau bang" sai
    # -> phu them khoanh khac khac va video khac moi la viec dang lam.
    picked = {}
    for r in out:
        picked.setdefault(r["video_id"], []).append(r["pts_time"])

    deferred = []
    for key, pts in pool[n_head:]:
        if len(out) >= target:
            break
        ts = picked.setdefault(key[0], [])
        if all(abs(pts - t) >= tail_gap for t in ts):
            if emit(key[0], key[1], "autofill",
                    f"trai deu >= {tail_gap}s trong video (o {head + 1}+)"):
                ts.append(pts)
        else:
            # DAY XUONG chu khong loai: dung sai cua ta chi la xap xi khoang
            # [s, e] that cua BTC, loai nham thi mat han co hoi.
            deferred.append(key)

    for key in deferred:
        if len(out) >= target:
            break
        emit(key[0], key[1], "autofill", "gan frame da chon, xep sau")

    return out[:target]
