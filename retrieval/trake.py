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
        Sua: frame da sap theo thoi gian nen cua so [t_i - delta, t_i) truot mot
        chieu -> dung deque don dieu lay max trong O(1) khau hao, tong O(N K).

Ghi chu: KHONG dung cot cos_to_prev de suy ra ranh gioi canh -- phan bo cua no
da bi chinh nguong lay mau cat cut. Moi quan he thoi gian o day deu tinh bang
pts_time.
"""

from collections import deque
from typing import List, Optional, Sequence, Tuple

import numpy as np

NEG = -np.inf


def dp_alignment(
    pts_times: Sequence[float],
    event_scores: np.ndarray,
    delta: float = 5.0,
    gamma: float = 0.5,
    candidates: Optional[Sequence[Sequence[int]]] = None,
) -> Tuple[List[int], float]:
    """Tra ve (path, max_score).

    path co dung `num_events` phan tu: path[k] la chi so frame khop su kien k,
    hoac -1 neu su kien do bi bo qua.

    pts_times     thoi gian cua tung frame, PHAI tang dan
    event_scores  (num_frames, num_events)
    delta         khoang thoi gian toi da giua hai su kien lien tiep
    gamma         diem phat cho moi su kien bi bo qua
    candidates    tuy chon: candidates[k] la tap frame duoc phep khop su kien k
                  (thu hep khong gian tim kiem; None = xet moi frame)
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
        # max{prev[j] : t_i - delta <= t_j < t_i} bang deque don dieu -> O(1)
        # khau hao, tong O(N*K) thay vi O(N^2*K).
        dq = deque()
        for i in range(n):
            j = i - 1
            if j >= 0 and prev[j] != NEG:
                while dq and prev[dq[-1]] <= prev[j]:
                    dq.pop()
                dq.append(j)
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


def events_to_scores(store, video_id, encoder, events):
    """(pts_times, frame_idxs, score_matrix) cho mot video.

    Tach rieng khoi dp_alignment de DP thuan tuy la thuat toan, khong dinh gi
    den tang du lieu -- nho vay test duoc bang ma tran dung san.
    """
    df = store.frames_of(video_id)
    if df.empty:
        return [], [], np.zeros((0, len(events)))

    sl = store.video_slice[video_id]
    X = store.X[sl[0]:sl[1]]
    Q = encoder.encode_texts(list(events))        # (K, D)
    return (df["pts_time"].tolist(), df["frame_idx"].tolist(), X @ Q.T)
