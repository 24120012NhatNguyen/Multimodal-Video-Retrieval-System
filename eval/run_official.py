#!/usr/bin/env python3
"""Chay TOAN BO duong nop bai roi cham theo cong thuc cua BTC.

    truy van -> tim kiem -> lap day 100 dong -> Final Score

Khac `run_eval.py`: file do do chat luong TANG TIM KIEM (Recall@k, MRR). File
nay do thu ma BTC thuc su cham -- ke ca co che chon 100 dong. Hai thu do co the
di nguoc nhau: xep hang tim kiem tot len ma co che chon 100 dong lam hong thi
diem van tut.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from official import BREAKPOINTS, TOL_SEC, final_score, print_report
from run_eval import DEFAULT_SET, load_set


def rows_from_search(res):
    """Ket qua engine.search -> [{"video_id","frame_idx","pts_time","score"}]."""
    out = []
    for v in res.get("videos", []):
        vi = v["video_info"]
        for j, fi in enumerate(vi["lst_keyframe_idxs"]):
            out.append({
                "video_id": v["video_id"],
                "frame_idx": int(fi),
                "pts_time": vi["lst_pts_times"][j],
                "score": (vi.get("lst_scores") or [None])[j],
            })
    return out


def run(engine, store, qset, use_llm=False, target=100, frame_topk=500,
        video_topn=30, align=False, cfg_over=None, tol=TOL_SEC, verbose=True):
    from retrieval.autofill import autofill
    from retrieval.query import decompose

    cfg = engine.cfg
    saved = {}
    for k, v in (cfg_over or {}).items():
        saved[k] = getattr(cfg, k, None)
        setattr(cfg, k, v)

    per_query, rows_log = [], []
    try:
        for q in qset:
            dq = decompose(query_vi=q["question"], use_llm=use_llm)
            t0 = time.time()
            res = engine.search(query_en=dq.query_en, query_vi=dq.query_vi,
                                video_topn=video_topn, frame_topk=frame_topk,
                                kind=dq.kind)
            if align and len(dq.clauses_en) >= 2:
                aligned = engine.align_videos(
                    dq.clauses_en, [v["video_id"] for v in res["videos"]])
                if aligned:
                    order = {a["video_id"]: i for i, a in enumerate(aligned)}
                    res["videos"].sort(
                        key=lambda v: order.get(v["video_id"], 1 << 30))

            cands = rows_from_search(res)
            rows = autofill(store, manual=[], candidates=cands, config=cfg,
                            target=target)
            el = time.time() - t0

            f, r = final_score(rows, q["answers"], store.fps, tol=tol)
            per_query.append((q["id"], f, r))
            rows_log.append({"id": q["id"], "final": f, "rank": r,
                             "n_row": len(rows), "n_cand": len(cands),
                             "sec": round(el, 2)})
            if verbose:
                rr = f"#{r}" if r else "--"
                print(f"  Q{q['id']} hang {rr:>5}  diem {f:.2f}  "
                      f"({len(rows)} dong tu {len(cands)} ung vien, {el:.1f}s)")
    finally:
        for k, v in saved.items():
            setattr(cfg, k, v)
    return per_query, rows_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=DEFAULT_SET)
    ap.add_argument("--llm", action="store_true", help="bat phan ra bang LLM")
    ap.add_argument("--align", action="store_true")
    ap.add_argument("--target", type=int, default=100)
    ap.add_argument("--frame-topk", type=int, default=500)
    ap.add_argument("--tol", type=float, default=TOL_SEC)
    ap.add_argument("--json")
    a = ap.parse_args()

    from retrieval import service

    s = service.get()
    qset = load_set(a.set)
    print(f"bo eval: {a.set} -- {len(qset)} cau hoi, "
          f"nguong k = {BREAKPOINTS}, dung sai {a.tol}s\n")
    pq, log = run(s["engine"], s["store"], qset, use_llm=a.llm,
                  target=a.target, frame_topk=a.frame_topk, align=a.align,
                  tol=a.tol)
    rep = print_report(pq)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"tong_ket": rep, "tung_cau": log}, f,
                      ensure_ascii=False, indent=2)
        print(f"\nda ghi {a.json}")


if __name__ == "__main__":
    main()
