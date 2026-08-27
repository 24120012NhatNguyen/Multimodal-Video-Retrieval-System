"""DP dong hang chuoi su kien tren truc thoi gian cua mot video.

Bai toan: trong mot video, tim day frame TANG DAN THEO THOI GIAN sao cho tong
diem khop chuoi su kien la lon nhat, cho phep bo qua su kien voi mot khoan phat.

Ban truoc co ba loi (muc B4 cua checklist), da sua het:

  B4#2  dp[:,0] = score[:,0] khien su kien DAU khong bao gio bo qua duoc.
        Sua: them cot 0 la TRANG THAI RONG (= 0), moi su kien deu bo qua duoc.

  B4#3  Chi bo qua duoc MOT su kien lien tiep (chi nhin dp[j][k-2]).
        Sua: buoc "bo qua E_k" day dp[i][k-1] - gamma sang cot k. Vi no lai co
        the tiep tuc bi day sang cot k+1, so su kien bo qua lien tiep la khong
        gioi han, phat cong don dung nhu mong doi.

  B4#4  Ba vong lap long nhau -> O(N^2 K).
        Sua: frame da sap theo thoi gian nen cua so [t_i - delta, t_i - min_gap]
        truot mot chieu -> deque don dieu lay max trong O(1) khau hao, O(N K).

LOI THU TU (do duoc, ban nay sua):

  gamma SAI CA THANG DO LAN HINH DANG.

  Thang do: event_scores la cosine SigLIP, do tren du lieu that co bien do
  [-0.064, +0.135]. gamma mac dinh 0.5 lon gap ~4 lan diem khop tot nhat co the
  dat -> nhanh "bo qua" khong bao gio duoc chon. Sua bang chuan hoa z-score theo
  phan bo cua chinh truy van tren corpus (event_stats).

  Hinh dang: chuan hoa xong VAN khong bo qua duoc. Do lai voi mot su kien co y
  dat sai ("tau vu tru ha canh xuong sao Hoa" trong ban tin tieng Viet): gamma
  = 0, 1, 2, 4, 8 deu cho ket qua y het, khong bo qua lan nao. Ly do: su kien
  vang mat VAN co mot frame nao do trong video cham diem z duong, ma so sanh la
  "khop (+z) hay bo qua (-gamma)" nen khop luon thang khi z > -gamma. z hiem khi
  xuong duoi -3.

  Cach dung: dat NGUONG tau -- muc z ma tu do tro len moi coi la su kien CO MAT.
  Diem khop thanh (z - tau), nen su kien chi hoi giong (z = 1) se am va bi bo
  qua, su kien that su co (z = 3) van duong. Do tren corpus: p99 cua z la ~2,8,
  nen tau = 2,0 la ranh gioi hop ly. gamma con lai la khoan phat cau truc cho
  viec de trong, mac dinh 0.

Ghi chu: KHONG dung cot cos_to_prev de suy ra ranh gioi canh -- phan bo cua no
da bi chinh nguong lay mau cat cut. Moi quan he thoi gian o day deu tinh bang
pts_time.
"""

from collections import deque
from typing import List, Optional, Sequence, Tuple

import numpy as np

NEG = -np.inf

# Buoc lay mau corpus de uoc luong trung binh/do lech cua tung su kien.
# 154.640 keyframe / 37 = ~4.180 mau -- du on dinh, ton ~5ms.
STATS_STRIDE = 37

# Nguong z de coi mot su kien la CO MAT. Do tren corpus: z cua p99 la ~2,8 va
# z cua frame tot nhat toan corpus la ~3,8. 2,0 la ranh gioi giua "hoi giong"
# va "dung la canh nay".
DEFAULT_TAU = 2.0


def dp_alignment(
    pts_times: Sequence[float],
    event_scores: np.ndarray,
    delta: float = 30.0,
    gamma: float = 1.0,
    min_gap: float = 0.0,
    candidates: Optional[Sequence[Sequence[int]]] = None,
) -> Tuple[List[int], float]:
    """Tra ve (path, max_score).

    path co dung `num_events` phan tu: path[k] la chi so frame khop su kien k,
    hoac -1 neu su kien do bi bo qua.

    pts_times     thoi gian cua tung frame, PHAI tang dan
    event_scores  (num_frames, num_events) -- nen la z-score, xem event_stats
    delta         khoang thoi gian TOI DA giua hai su kien lien tiep
    gamma         diem phat cho moi su kien bi bo qua (don vi: do lech chuan)
    min_gap       khoang thoi gian TOI THIEU giua hai su kien lien tiep. 0 =
                  chi doi hoi frame sau muon hon frame truoc.
    candidates    tuy chon: candidates[k] la tap frame duoc phep khop su kien k
    """
    t = np.asarray(pts_times, dtype=np.float64)
    S = np.asarray(event_scores, dtype=np.float64)
    if S.ndim != 2:
        raise ValueError(f"event_scores phai la ma tran 2 chieu, dang {S.shape}")
    n, K = S.shape
    if n == 0 or K == 0:
        return [], NEG
    if len(t) != n:
        raise ValueError(f"pts_times ({len(t)}) khong khop so frame ({n})")
    if np.any(np.diff(t) < 0):
        raise ValueError("pts_times phai tang dan")
    if min_gap > delta:
        raise ValueError(f"min_gap ({min_gap}) khong duoc lon hon delta ({delta})")

    allowed = None
    if candidates is not None:
        allowed = [np.zeros(n, dtype=bool) for _ in range(K)]
        for k in range(K):
            idx = np.asarray(list(candidates[k]), dtype=int)
            if idx.size:
                allowed[k][idx] = True

    # Trang thai = "frame KHOP GAN NHAT", khong phai "frame hien tai". Bo qua
    # mot su kien KHONG lam trang thai tien len -- do la mau chot de bo qua
    # nhieu su kien lien tiep roi van khop tiep vao frame bat ky.
    #
    # Rieng trang thai "chua khop gi" khong gan voi frame nao, nen giu rieng
    # thanh mot so vo huong: tu do co the khop vao BAT KY frame nao (ke ca frame
    # dau video), khong bi rang buoc delta.
    dp = np.full((n, K + 1), NEG, dtype=np.float64)
    par = np.full((n, K + 1), -1, dtype=np.int64)      # frame khop truoc do
    skipped = np.zeros((n, K + 1), dtype=bool)         # su kien k co bi bo qua?
    from_none = np.zeros((n, K + 1), dtype=bool)       # tien nhiem la trang thai rong

    none_prev = 0.0        # chua khop su kien nao, chua ton phat nao

    for k in range(1, K + 1):
        prev = dp[:, k - 1].copy()

        # --- khop su kien k ------------------------------------------------
        # max{prev[j] : t_i - delta <= t_j <= t_i - min_gap, j < i}
        # Ca hai bien deu don dieu theo i -> deque don dieu, O(1) khau hao.
        dq = deque()
        p = 0                      # con tro nap: frame da du xa ve truoc
        for i in range(n):
            while p < i and t[p] <= t[i] - min_gap:
                if prev[p] != NEG:
                    while dq and prev[dq[-1]] <= prev[p]:
                        dq.pop()
                    dq.append(p)
                p += 1
            while dq and t[dq[0]] < t[i] - delta:
                dq.popleft()

            if allowed is not None and not allowed[k - 1][i]:
                continue

            best_prev, best_j, via_none = NEG, -1, False
            if dq and prev[dq[0]] != NEG:
                best_prev, best_j = prev[dq[0]], dq[0]
            # Tu trang thai rong: khong doi hoi frame som hon, khong rang buoc delta.
            if none_prev > best_prev:
                best_prev, best_j, via_none = none_prev, -1, True

            if best_prev == NEG:
                continue
            val = S[i, k - 1] + best_prev
            if val > dp[i, k]:
                dp[i, k] = val
                par[i, k] = best_j
                skipped[i, k] = False
                from_none[i, k] = via_none

        # --- bo qua su kien k: giu nguyen frame khop gan nhat ---------------
        for i in range(n):
            if prev[i] == NEG:
                continue
            val = prev[i] - gamma
            if val > dp[i, k]:
                dp[i, k] = val
                par[i, k] = i
                skipped[i, k] = True
                from_none[i, k] = False

        none_prev -= gamma      # bo qua ca su kien k khi chua khop gi

    # Ket qua tot nhat: hoac ket thuc o mot frame, hoac bo qua sach moi su kien.
    best_end = int(np.argmax(dp[:, K])) if n else -1
    best_frame_score = float(dp[best_end, K]) if n else NEG

    if none_prev >= best_frame_score:
        # Bo qua toan bo -- van la mot loi giai hop le, chi la rat te.
        if not np.isfinite(none_prev):
            return [], NEG
        return [-1] * K, float(none_prev)

    if not np.isfinite(best_frame_score):
        return [], NEG

    # --- lan nguoc ------------------------------------------------------
    path: List[int] = []
    i, k = best_end, K
    while k >= 1:
        if skipped[i, k]:
            path.append(-1)
            k -= 1
            continue
        path.append(int(i))
        if from_none[i, k]:
            # Tien nhiem la trang thai rong -> moi su kien con lai deu bi bo qua.
            k -= 1
            while k >= 1:
                path.append(-1)
                k -= 1
            break
        i = int(par[i, k])
        k -= 1
    path.reverse()
    return path, best_frame_score


# ---------------------------------------------------------------------------
def event_stats(store, encoder, events, stride=STATS_STRIDE):
    """(Q, mu, sigma) -- vector su kien va phan bo diem cua chung tren corpus.

    Vi sao can: cosine cua tung cau truy van co MUC NEN rieng. Do tren corpus
    that, bon menh de cua cung mot truy van co trung binh tu -0.015 den +0.009
    va do lech tu 0.024 den 0.033. Cong thang cosine tho lai nghia la menh de
    "de an diem" duoc tinh nang hon menh de kho, va khoan phat gamma thi khong
    the dat chung cho ca hai. Doi sang z-score thi ca hai van de bien mat.

    Tinh mot lan cho ca truy van roi dung lai cho moi video.
    """
    Q = encoder.encode_texts(list(events))                 # (K, D)
    sample = store.X[::max(1, int(stride))]
    if sample.shape[0] < 32:
        sample = store.X
    S = sample @ Q.T
    mu = S.mean(axis=0)
    sigma = S.std(axis=0)
    # Do lech 0 nghia la moi frame cham diem y het -- chia se ra vo cuc.
    sigma = np.where(sigma > 1e-6, sigma, 1.0)
    return Q, mu.astype(np.float32), sigma.astype(np.float32)


def events_to_scores(store, video_id, encoder, events, stats=None,
                     normalize=True, tau=DEFAULT_TAU):
    """(pts_times, frame_idxs, score_matrix) cho mot video.

    Tach rieng khoi dp_alignment de DP thuan tuy la thuat toan, khong dinh gi
    den tang du lieu -- nho vay test duoc bang ma tran dung san.

    normalize=True -> tra ve (z - tau): duong nghia la su kien CO MAT, am nghia
                      la khong. Nho vay nhanh "bo qua" cua DP moi kich hoat duoc.
    stats          -> ket qua event_stats() dung lai giua cac video.
    """
    df = store.frames_of(video_id)
    if df.empty:
        return [], [], np.zeros((0, len(events)))

    if stats is None:
        if normalize:
            stats = event_stats(store, encoder, events)
        else:
            stats = (encoder.encode_texts(list(events)), None, None)
    Q, mu, sigma = stats

    sl = store.video_slice[video_id]
    S = store.X[sl[0]:sl[1]] @ Q.T
    if normalize and mu is not None:
        S = (S - mu) / sigma - float(tau)
    return (df["pts_time"].tolist(), df["frame_idx"].tolist(), S)


def fill_skipped(path, pts_times, scores, delta):
    """Doan frame cho nhung su kien bi DP bo qua.

    Bai TRAKE cua BTC doi MOT frame cho MOI su kien -- khong nop duoc o trong.
    Bo qua o day la cong cu cham diem ("su kien nay khong that su co trong
    video"), khong phai cau tra loi cuoi. Nen van dua ra frame kha di nhat cho
    tung su kien bi bo, co danh dau `weak` de nguoi dung biet la doan.

    Chon frame tot nhat trong khoang thoi gian hop le giua hai su kien da khop
    hai ben, giu dung thu tu thoi gian.
    """
    t = np.asarray(pts_times, dtype=np.float64)
    S = np.asarray(scores, dtype=np.float64)
    n, K = S.shape
    out = list(path)
    for k in range(K):
        if out[k] != -1:
            continue
        lo_i = next((out[j] for j in range(k - 1, -1, -1) if out[j] != -1), None)
        hi_i = next((out[j] for j in range(k + 1, K) if out[j] != -1), None)
        lo = t[lo_i] if lo_i is not None else t[0] - 1.0
        hi = t[hi_i] if hi_i is not None else t[-1] + 1.0
        if lo_i is not None:
            lo = max(lo, t[lo_i])
            hi = min(hi, t[lo_i] + delta)
        m = (t > lo) & (t < hi) if lo_i is not None else (t < hi)
        if not m.any():
            m = np.ones(n, dtype=bool)
        idx = np.flatnonzero(m)
        out[k] = int(idx[int(np.argmax(S[idx, k]))])
    return out
