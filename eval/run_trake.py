#!/usr/bin/env python3
"""Do TRAKE tren ground truth that (eval/trake_ground_truth.json)."""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gt import load_trake, trake_score

# Truy van TRAKE cua BTC danh so su kien ngay trong cau: "(1) ... (2) ... (3) ..."
_NUM = re.compile(r"\(\s*(\d+)\s*\)")


def split_events(text, n_events):
    """Tach cau truy van thanh dung n_events menh de.

    BTC viet san so thu tu "(1) ... (2) ..." nen khong phai doan: cat theo dung
    cac moc do. Khong co moc thi lui ve tach theo dau phay/cham phay.
    """
    parts = _NUM.split(text)
    if len(parts) >= 3:
        ev = [parts[i + 1].strip(" ,.;:") for i in range(1, len(parts) - 1, 2)]
        ev = [e for e in ev if e]
        if len(ev) >= n_events:
            return ev[:n_events]
    chunks = [c.strip(" ,.;:") for c in re.split(r"[,;.]", text) if c.strip()]
    return chunks[-n_events:] if len(chunks) >= n_events else chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="dich menh de bang LLM")
    ap.add_argument("--video-topn", type=int, default=30)
    a = ap.parse_args()

    from retrieval import service
    from retrieval.query import _translate

    s = service.get()
    eng, store = s["engine"], s["store"]
    qset = load_trake()
    print(f"TRAKE: {len(qset)} mau\n")

    tot_r, tot_final = 0.0, 0.0
    for q in qset:
        ev_vi = split_events(q["query"], q["n_events"])
        ev_en = [(_translate(e) or e) for e in ev_vi]

        # Buoc 1: tim video ung vien bang tang hop nhat
        fused, _, _ = eng.rank_videos("\n".join(ev_en), q["query"],
                                      kind="generic_chain")
        cands = [v for v, _ in fused if v in store.video_slice][:a.video_topn]
        vrank = cands.index(q["video_id"]) + 1 if q["video_id"] in cands else None

        # Buoc 2: dong hang tren tung video
        aligned = eng.align_videos(ev_en, cands)
        best = aligned[0] if aligned else None
        rank_dp = next((i + 1 for i, x in enumerate(aligned)
                        if x["video_id"] == q["video_id"]), None)

        # R-Score chi tinh khi DUNG video (luat cua BTC)
        r = 0.0
        mine = next((x for x in aligned if x["video_id"] == q["video_id"]), None)
        if mine:
            r = trake_score(mine["matched_frames"] if "matched_frames" in mine
                            else [m["frame_idx"] for m in mine["matched"]],
                            q["events"])
        tot_r += r
        # Final Score: hang cua video trong danh sach nop
        if rank_dp:
            fs = sum(1 for k in (1, 5, 20, 50, 100) if k >= rank_dp) / 5 * r
        else:
            fs = 0.0
        tot_final += fs

        print(f"  {q['id']}  {q['video_id']}  {q['n_events']} su kien")
        print(f"    menh de tach ra: {[e[:36] for e in ev_vi]}")
        print(f"    video: hang tim kiem {vrank}, sau dong hang {rank_dp}"
              f"{' (dung dau)' if rank_dp == 1 else ''}")
        if mine:
            got = [m["frame_idx"] for m in mine["matched"]]
            print(f"    frame khop: {got}")
            print(f"    khoang dung: {q['events']}")
            print(f"    -> R-Score {r:.2f}  ({int(r*q['n_events'])}/{q['n_events']} su kien dung khoang)")
        else:
            print("    -> video dung KHONG duoc dong hang")
        print(f"    Final {fs:.3f}\n")

    n = len(qset)
    print("=" * 60)
    print(f"  R-Score trung binh  {tot_r/n:.4f}")
    print(f"  FINAL SCORE         {tot_final/n:.4f}   tren {n} mau")


if __name__ == "__main__":
    main()
