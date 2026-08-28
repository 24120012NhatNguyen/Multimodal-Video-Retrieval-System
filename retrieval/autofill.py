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

import numpy as np

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


def densify(store, video_id, frame_idx, step=25, max_n=8):
    """Frame nam trong KHOANG TRONG hai ben mot keyframe da chon.

    Vi sao can -- do tren eval/kis_ground_truth.csv:

        khoang cach hai keyframe lien tiep   trung vi  75 frame (3.0s)
        do rong khoang dap an [s, e]         trung vi  75 frame

    Hai con so bang nhau, nen moi khoang dap an chi chua 0-2 keyframe cua ta, va
    co khoang khong chua cai nao. sample_10 la vi du: video xep hang #1, dung
    video, ma van 0 diem -- khoang [11500, 11625] khong co keyframe nao.

    Bai nop nhan `frame_id` BAT KY -- khong dong nao bat frame phai la keyframe
    cua ta, va may local trich duoc frame bat ky tu mp4. Do duoc: lay mau buoc
    25 frame (1.0s) phu 11/11 khoang dap an.

    Lap dung KHOANG TRONG chu khong buoc co dinh: neu khong co keyframe nao roi
    vao khoang dap an thi theo dinh nghia dap an nam GIUA hai keyframe, nen cho
    dang tim la hai khoang trong ke ben. Buoc co dinh +/-2 lan thi voi xa nhat
    50 frame, khong toi -- sample_10 can toi +76.

    Tra ve danh sach frame, XEN KE hai phia va gan truoc xa sau.
    """
    df = store.frames_of(video_id)
    if df.empty:
        return []
    fi = df["frame_idx"].to_numpy()
    f = int(frame_idx)
    prev = fi[fi < f].max() if (fi < f).any() else f
    nxt = fi[fi > f].max(initial=f) if False else (fi[fi > f].min() if (fi > f).any() else f)

    left, right = [], []
    d = step
    while len(left) + len(right) < 2 * max_n and d <= max(f - prev, nxt - f) + step:
        if f - d > prev:
            left.append(f - d)
        if f + d < nxt:
            right.append(f + d)
        d += step

    out, i = [], 0
    while i < max(len(left), len(right)) and len(out) < 2 * max_n:
        if i < len(right):
            out.append(int(right[i]))
        if i < len(left):
            out.append(int(left[i]))
        i += 1
    return out


def autofill(store, manual, candidates=None, config=None, ignore=None,
             target=None, head=None, tail_gap=None, densify_n=None,
             densify_step=None, densify_top=None):
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
    densify_n = (getattr(cfg, "autofill_densify_n", 2)
                 if densify_n is None else densify_n)
    densify_step = (getattr(cfg, "autofill_densify_step", 25)
                    if densify_step is None else densify_step)
    densify_top = (getattr(cfg, "autofill_densify_top", 5)
                   if densify_top is None else densify_top)
    look = _lookup(store)

    # Bien frame cua tung video, de khong sinh ra frame nam ngoai video.
    _bounds = {}

    def bounds(vid):
        if vid not in _bounds:
            df = store.frames_of(vid)
            _bounds[vid] = ((int(df["frame_idx"].min()), int(df["frame_idx"].max()))
                            if not df.empty else (None, None))
        return _bounds[vid]

    ign = set()
    for it in (ignore or []):
        if isinstance(it, dict):
            ign.add((it.get("video_id"), int(it.get("frame_idx"))))
        else:
            ign.add((it[0], int(it[1])))

    out, seen = [], set()

    def emit(video_id, frame_idx, source, reason, allow_between=False):
        key = (video_id, int(frame_idx))
        if key in seen or key in ign:
            return False
        hit = look.get(key)
        if hit is None:
            if not allow_between:
                return False
            # Frame nam GIUA hai keyframe: khong co trong bang, nhung van nop
            # duoc va van xem duoc (server anh trich thang tu mp4 theo fps).
            fps = store.fps.get(video_id) or 25.0
            hit = (-1, float(frame_idx) / fps)
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

    # --- 3. Phan DAU: NGUYEN thu hang tim kiem ---------------------------
    # KHONG xen gi vao day. Da thu va do: xen frame giua vao 20 o dau lam Final
    # Score tut tu 0.3455 xuong 0.1818 -- no day chinh nhung keyframe dang trung
    # ra ngoai. Cac o dau dang 0,60-1,00 diem moi o; khong danh cuoc chung.
    n_head = max(0, head - len(out))
    for key, _ in pool[:n_head]:
        emit(key[0], key[1], "autofill",
             f"thu hang tim kiem (o 1-{head}, o nay dang "
             f"{slot_weight(len(out) + 1):.2f} diem)")

    # --- 4. Phan DUOI: trai deu theo thoi gian ---------------------------
    # Den duoc day nghia la cac o dau da truot -> gia thuyet "dung dau bang" sai
    # -> phu them khoanh khac khac va video khac moi la viec dang lam.
    # --- 3b. Xen frame GIUA, ngay sau phan dau ---------------------------
    # Den duoc o 21 nghia la 20 keyframe dau da truot. Gia thuyet kha di nhat
    # con lai: dung vung, sai khoanh khac -- vi bo keyframe thua thot ngang voi
    # do hep cua dap an (xem ghi chu ham densify). Vay thi lang gieng +/-1s cua
    # cac moc dau bang la nuoc di dung, va no khong cuop o cua ai nua.
    if densify_n > 0 and densify_top > 0:
        anchors = [k for k, _ in pool[:densify_top]]
        plans = [(vid, fi, densify(store, vid, fi, densify_step, densify_n))
                 for vid, fi in anchors]
        # Vong tron qua cac moc: moc dau bang duoc frame gan nhat truoc, roi moi
        # den luot moc thu hai -- de o tot nhat khong bi mot moc chiem het.
        for i in range(2 * densify_n):
            for vid, fi, fr in plans:
                if len(out) >= target:
                    break
                if i < len(fr):
                    emit(vid, fr[i], "autofill",
                         f"lap khoang trong {fr[i] - fi:+d} frame quanh {vid}#{fi}",
                         allow_between=True)

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
