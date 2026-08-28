#!/usr/bin/env python3
"""Chay toan bo duong nop bai tren ground truth THAT roi cham theo chuan BTC."""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gt import final_score, load_kis, load_trake, trake_score, video_rank


def rows_of(res):
    out = []
    for v in res.get("videos", []):
        vi = v["video_info"]
        for j, fi in enumerate(vi["lst_keyframe_idxs"]):
            out.append({"video_id": v["video_id"], "frame_idx": int(fi),
                        "pts_time": vi["lst_pts_times"][j],
                        "score": (vi.get("lst_scores") or [None])[j]})
    return out


def run_kis(engine, store, qset, use_llm=False, target=100, frame_topk=1500,
            cfg_over=None, verbose=True):
    from retrieval.autofill import autofill
    from retrieval.query import decompose

    cfg = engine.cfg
    saved = {k: getattr(cfg, k, None) for k in (cfg_over or {})}
    for k, v in (cfg_over or {}).items():
        setattr(cfg, k, v)

    rows_out = []
    try:
        for q in qset:
            dq = decompose(query_vi=q["query"], use_llm=use_llm)
            t0 = time.time()
            res = engine.search(query_en=dq.query_en, query_vi=dq.query_vi,
                                video_topn=30, frame_topk=frame_topk, kind=dq.kind)
            cands = rows_of(res)
            rows = autofill(store, manual=[], candidates=cands, config=cfg,
                            target=target)
            el = time.time() - t0

            f, r = final_score(rows, q["video_id"], q["s"], q["e"])
            vr = video_rank([v["video_id"] for v in res["videos"]], q["video_id"])
            rows_out.append({"id": q["id"], "kind": q["kind"], "final": f,
                             "rank": r, "vrank": vr, "sec": el})
            if verbose:
                rr = f"#{r}" if r else "--"
                vv = f"#{vr}" if vr else "--"
                print(f"  {q['id']:10} {q['kind']:4} video {vv:>4}  dap an {rr:>5}"
                      f"  diem {f:.2f}  {el:4.1f}s")
    finally:
        for k, v in saved.items():
            setattr(cfg, k, v)
    return rows_out


def report(rows, label=""):
    n = len(rows)
    if not n:
        return {}
    final = sum(r["final"] for r in rows) / n
    # "Trong tam tay": video dung nam trong top-K -> nguoi dung mo ra quet la thay.
    reach = {k: sum(1 for r in rows if r["vrank"] and r["vrank"] <= k) / n
             for k in (1, 5, 10, 30)}
    print("\n" + "=" * 64)
    if label:
        print(label)
    print(f"  FINAL SCORE (chuan BTC)   {final:.4f}   tren {n} mau")
    print("  Trong tam tay -- video dung nam trong top-K:")
    for k, v in reach.items():
        print(f"     top-{k:<3} {v:6.1%}")
    print(f"  do tre trung binh          {sum(r['sec'] for r in rows)/n:.1f}s")
    return {"final": final, "reach": reach, "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--target", type=int, default=100)
    ap.add_argument("--frame-topk", type=int, default=1500)
    a = ap.parse_args()

    from retrieval import service

    s = service.get()
    qset = load_kis()
    print(f"KIS + Q&A: {len(qset)} mau (dap an la khoang [s,e] that cua BTC)\n")
    rows = run_kis(s["engine"], s["store"], qset, use_llm=a.llm,
                   target=a.target, frame_topk=a.frame_topk)
    report(rows)


if __name__ == "__main__":
    main()
