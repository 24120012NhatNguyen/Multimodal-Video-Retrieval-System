#!/usr/bin/env python3
"""Bo do tren ground truth THAT cua BTC (eval/*_ground_truth.csv).

Khac han bo 7 cau truoc: o day dap an la mot KHOANG [s, e] tinh bang frame_idx,
dung chuan cham diem cua BTC, khong phai mot diem roi le va cung khong con dung
sai +/-2s do ta tu bia ra.

DO DAC QUAN TRONG NHAT, do truoc khi lam bat cu gi:

    khoang cach hai keyframe lien tiep   trung vi  75 frame (3.0s), p90 135
    do rong khoang dap an                trung vi  75 frame, min 30, max 150

Hai con so nay BANG NHAU. Nghia la bo keyframe cua ta THUA THOT ngang voi do
hep cua dap an -- 11 khoang dap an chi chua tong cong 0-2 keyframe moi khoang,
va co khoang khong chua cai nao (sample_10).

He qua: KHONG the trong cho vao viec dat dung keyframe co san. Nhung bai nop
nhan `frame_id` BAT KY, khong bat buoc phai la keyframe cua ta -- va may local
trich duoc frame bat ky tu mp4. Lay mau day quanh cho da tim ra:

    buoc 100 frame (4.0s)  ->  9/11 khoang dap an duoc phu
    buoc  25 frame (1.0s)  -> 11/11

Do la ly do co `densify` trong retrieval/autofill.py.

HAI THUOC DO, vi he nay CO NGUOI DUNG NGOI TRONG:

  final_score   diem chinh thuc cua BTC -- may tu chay, khong ai can thiep
  trong_tam_tay video dung co nam trong top-K khong. Neu co, nguoi dung mo
                "Ca video" / "Dai frame" ra quet tay la thay -- va do moi la
                cach he nay thuc su duoc dung.
"""

import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BREAKPOINTS = (1, 5, 20, 50, 100)
HERE = os.path.dirname(os.path.abspath(__file__))


def _rows(name):
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8-sig") as f:
        return [r for r in csv.DictReader(f) if r.get("sample_id")]


def load_kis():
    """[{id, query, video_id, s, e, example}] -- gom ca KIS lan Q&A.

    Q&A duoc cham nhu KIS o day: phan `answer` do NGUOI DUNG dien, may chi co
    nhiem vu dua dung khung hinh len. Danh gia rieng phan chu la viec khac.
    """
    out = []
    for name, kind in (("kis_ground_truth.csv", "kis"),
                       ("qa_ground_truth.csv", "qa")):
        for r in _rows(name):
            out.append({
                "id": r["sample_id"], "kind": kind,
                "query": r["query_text"], "video_id": r["video_id"],
                "s": int(r["s"]), "e": int(r["e"]),
                "example": int(r["example_frame_id"]),
                "answer": (r.get("answer") or "").strip(),
            })
    return sorted(out, key=lambda x: x["id"])


def load_trake():
    p = os.path.join(HERE, "trake_ground_truth.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raw = _rows("trake_ground_truth.csv")
    out = []
    for q in raw:
        n = int(q["n_events"])
        out.append({
            "id": q["sample_id"], "query": q["query_text"],
            "video_id": q["video_id"], "n_events": n,
            "events": [(int(q[f"event{k}_s"]), int(q[f"event{k}_e"]))
                       for k in range(1, n + 1)],
        })
    return sorted(out, key=lambda x: x["id"])


# ---------------------------------------------------------------------------
def slot_weight(i, breakpoints=BREAKPOINTS):
    return sum(1 for k in breakpoints if k >= i) / len(breakpoints)


def first_hit(rows, video_id, s, e):
    """Thu hang (tu 1) cua dong dau tien dung video VA frame nam trong [s, e]."""
    for i, r in enumerate(rows, 1):
        if r["video_id"] == video_id and s <= int(r["frame_idx"]) <= e:
            return i
    return None


def final_score(rows, video_id, s, e, breakpoints=BREAKPOINTS):
    r = first_hit(rows, video_id, s, e)
    if r is None:
        return 0.0, None
    return sum(1 for k in breakpoints if k >= r) / len(breakpoints), r


def video_rank(videos, video_id):
    """Video dung dung thu may trong danh sach video da xep hang."""
    for i, v in enumerate(videos, 1):
        if v == video_id:
            return i
    return None


def trake_score(matched, events):
    """R-Score cua TRAKE: ty le su kien co frame roi dung khoang cua no.

    Sai video thi ben goi da tra 0 truoc khi den day.
    """
    if not events:
        return 0.0
    ok = 0
    for k, (a, b) in enumerate(events):
        if k < len(matched) and matched[k] is not None and a <= matched[k] <= b:
            ok += 1
    return ok / len(events)


if __name__ == "__main__":
    k, t = load_kis(), load_trake()
    print(f"KIS+Q&A: {len(k)} mau | TRAKE: {len(t)} mau")
    for q in k:
        print(f"  {q['id']:10} {q['kind']:4} {q['video_id']:10} "
              f"[{q['s']}, {q['e']}] rong {q['e']-q['s']:4d}")
    for q in t:
        print(f"  {q['id']:10} trake {q['video_id']:10} {q['n_events']} su kien "
              f"{q['events']}")
