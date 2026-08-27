#!/usr/bin/env python3
"""Cham diem theo DUNG cong thuc cua BTC AIC 2026 (xem Debug/mark.md).

    R@k        = max{ R-Score(r_1..r_k) }          k in {1, 5, 20, 50, 100}
    Final      = trung binh cong cua 5 gia tri R@k

Voi KIS, R-Score la NHI PHAN (dung video VA frame nam trong [s,e]). Ket hop hai
dieu do lai:

    R@k = 1 voi moi k >= r, trong do r la thu hang cua dap an DUNG DAU TIEN

nen Final Score chi phu thuoc vao MOT con so: r.

    r = 1       -> 5/5 = 1.00
    r = 2..5    -> 4/5 = 0.80
    r = 6..20   -> 3/5 = 0.60
    r = 21..50  -> 2/5 = 0.40
    r = 51..100 -> 1/5 = 0.20
    r > 100     -> 0

HAI HE QUA quyet dinh moi thiet ke ben duoi:

1. CHI dap an dung DAU TIEN co gia tri. Dap an dung thu hai tro di cong 0.
   Nen dien them mot frame gan het suc giong frame da chon la VUT MOT O.

2. Gia tri bien cua tung o rat lech nhau:
       o 1        1.00      <- gap 5 lan o thu 100
       o 2-5      0.80
       o 6-20     0.60
       o 21-50    0.40
       o 51-100   0.20
   Day la ly do co ham `slot_weight`: moi thuat toan chon 100 dong deu phai
   toi uu theo trong so nay, khong phai theo "phu deu".
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BREAKPOINTS = (1, 5, 20, 50, 100)

# Dap an cua BTC la mot KHOANG [s, e]; bo eval cua ta chi ghi tung frame roi rac
# nen lay cua so +/- TOL quanh moi frame lam xap xi cua khoang do.
TOL_SEC = 2.0


def slot_weight(i, breakpoints=BREAKPOINTS):
    """Gia tri toi da mot o thu i (dem tu 1) co the mang lai cho Final Score."""
    return sum(1 for k in breakpoints if k >= i) / len(breakpoints)


def first_correct_rank(rows, answers, fps, tol=TOL_SEC):
    """Thu hang (dem tu 1) cua dap an dung dau tien, hoac None."""
    want = [(v, f / (fps.get(v) or 25.0)) for v, f, _ in answers]
    for i, r in enumerate(rows, 1):
        v = r["video_id"]
        t = r.get("pts_time")
        if t is None:
            t = int(r["frame_idx"]) / (fps.get(v) or 25.0)
        if any(v == wv and abs(t - wt) <= tol for wv, wt in want):
            return i
    return None


def final_score(rows, answers, fps, tol=TOL_SEC, breakpoints=BREAKPOINTS):
    r = first_correct_rank(rows, answers, fps, tol)
    if r is None:
        return 0.0, None
    return sum(1 for k in breakpoints if k >= r) / len(breakpoints), r


def score_set(per_query):
    """[(id, final, rank)] -> bang tong ket."""
    n = len(per_query)
    if not n:
        return {}
    mean = sum(f for _, f, _ in per_query) / n
    dist = {}
    for _, _, r in per_query:
        band = ("khong co" if r is None else
                "1" if r == 1 else "2-5" if r <= 5 else "6-20" if r <= 20
                else "21-50" if r <= 50 else "51-100")
        dist[band] = dist.get(band, 0) + 1
    return {"final_score": mean, "n": n, "phan_bo_hang": dist}


def print_report(per_query, label=""):
    s = score_set(per_query)
    print(f"\n{'=' * 60}")
    if label:
        print(label)
    print(f"  FINAL SCORE = {s['final_score']:.4f}   ({s['n']} truy van)")
    order = ["1", "2-5", "6-20", "21-50", "51-100", "khong co"]
    val = {"1": 1.0, "2-5": .8, "6-20": .6, "21-50": .4, "51-100": .2,
           "khong co": 0.0}
    for b in order:
        if b in s["phan_bo_hang"]:
            print(f"    hang {b:9} x{s['phan_bo_hang'][b]}   -> {val[b]:.1f} diem/cau")
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Bang gia tri bien cua tung o")
    ap.parse_args()
    print("Gia tri toi da moi o co the mang lai (Final Score):")
    prev = None
    for i in (1, 2, 5, 6, 20, 21, 50, 51, 100, 101):
        w = slot_weight(i) if i <= 100 else 0.0
        mark = "  <-- nguong" if w != prev else ""
        print(f"  o {i:4}   {w:.2f}{mark}")
        prev = w
