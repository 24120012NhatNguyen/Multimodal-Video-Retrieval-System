#!/usr/bin/env python3
"""Do chat luong tim kiem tren bo cau hoi co dap an (muc C1 cua checklist).

Khong co bo nay thi "tot hon / te hon" chi la cam giac. Moi hang so trong
config/fusion.json deu phai duoc chinh dua tren so o day.

    python eval/run_eval.py                       # chay mac dinh
    python eval/run_eval.py --no-llm              # bo phan ra bang LLM
    python eval/run_eval.py --weights siglip=6    # thu mot bo trong so khac
    python eval/run_eval.py --compare             # doi chieu truoc/sau khi cong
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_SET = "Debug/7_questions.json"
# Dap an ghi frame_idx cua BTC; keyframe cua ta lay mau khac nen gan nhu khong
# bao gio trung dung con so do. So sanh theo THOI GIAN moi co nghia.
FRAME_TOL_SEC = 2.0


def parse_answer(a):
    """'L30_V046,4865' hoac 'L22_V008,5638,deo ta pua' -> (video_id, frame_idx, text)"""
    parts = [p.strip() for p in str(a).split(",")]
    if len(parts) < 2:
        return None
    try:
        return parts[0], int(parts[1]), (",".join(parts[2:]) if len(parts) > 2 else None)
    except ValueError:
        return None


def load_set(path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out = []
    for i, q in enumerate(raw, 1):
        answers = [x for x in (parse_answer(a) for a in q.get("expected_answers", [])) if x]
        out.append({"id": i, "question": q["question"], "answers": answers,
                    "videos": {v for v, _, _ in answers}})
    return out


def evaluate(engine, store, qset, k_videos=(1, 5, 10, 30), k_frames=(10, 50, 100),
             use_llm=True, weights=None, verbose=True):
    from retrieval.query import decompose

    fps = store.fps
    rows = []
    for q in qset:
        dq = decompose(query_vi=q["question"], use_llm=use_llm)
        t0 = time.time()
        res = engine.search(query_en=dq.query_en, query_vi=dq.query_vi,
                            video_topn=max(k_videos), frame_topk=max(k_frames),
                            weights=weights, kind=dq.kind)
        el = time.time() - t0

        ranked = [v["video_id"] for v in res["videos"]]
        vhit = {k: any(v in q["videos"] for v in ranked[:k]) for k in k_videos}
        vrank = next((i + 1 for i, v in enumerate(ranked) if v in q["videos"]), None)

        # frame: dung khi cung video VA lech thoi gian <= FRAME_TOL_SEC
        flat = []
        for v in res["videos"]:
            vi = v["video_info"]
            for j in range(len(vi["lst_keyframe_idxs"])):
                flat.append((v["video_id"], vi["lst_keyframe_idxs"][j],
                             vi.get("lst_pts_times", [None] * 99)[j]))
        want = [(v, f / (fps.get(v) or 25.0)) for v, f, _ in q["answers"]]
        fhit, frank = {k: False for k in k_frames}, None
        for i, (v, fi, pts) in enumerate(flat):
            t = pts if pts is not None else fi / (fps.get(v) or 25.0)
            if any(v == wv and abs(t - wt) <= FRAME_TOL_SEC for wv, wt in want):
                frank = i + 1
                for k in k_frames:
                    if i < k:
                        fhit[k] = True
                break

        rows.append({"id": q["id"], "kind": dq.kind, "src": dq.source,
                     "vhit": vhit, "vrank": vrank, "fhit": fhit, "frank": frank,
                     "sec": el, "n_rank": res.get("n_videos_ranked", 0),
                     "aligned": res.get("aligned"),
                     "question": q["question"][:58]})

        if verbose:
            vr = f"#{vrank}" if vrank else "--"
            fr = f"#{frank}" if frank else "--"
            print(f"  Q{q['id']} {dq.kind:14} video {vr:>5}  frame {fr:>5}  "
                  f"{el:5.1f}s   {q['question'][:46]}")
    return rows


def summarize(rows, k_videos, k_frames):
    n = len(rows)
    print("\n" + "=" * 66)
    print(f"{'chi so':<22} {'gia tri':>10}")
    print("-" * 66)
    for k in k_videos:
        c = sum(r["vhit"][k] for r in rows)
        print(f"  Recall@{k:<3} (video)      {c}/{n} = {100*c/n:5.1f}%")
    for k in k_frames:
        c = sum(r["fhit"][k] for r in rows)
        print(f"  Recall@{k:<3} (frame)      {c}/{n} = {100*c/n:5.1f}%")
    mrr = sum(1.0 / r["vrank"] for r in rows if r["vrank"]) / n
    print(f"  MRR (video)            {mrr:10.3f}")
    print(f"  do tre trung binh      {sum(r['sec'] for r in rows)/n:9.1f}s")
    miss = [r["id"] for r in rows if not r["vhit"][max(k_videos)]]
    if miss:
        print(f"  truot han (Q):         {miss}")
    return {"mrr": mrr,
            "recall_video": {k: sum(r["vhit"][k] for r in rows) / n for k in k_videos},
            "recall_frame": {k: sum(r["fhit"][k] for r in rows) / n for k in k_frames}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=DEFAULT_SET)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--weights", nargs="*", default=[],
                    help="vd: siglip=6 meta=0.2")
    ap.add_argument("--json", help="ghi ket qua ra file")
    a = ap.parse_args()

    weights = {}
    for w in a.weights:
        k, _, v = w.partition("=")
        weights[k] = float(v)

    from retrieval import service
    s = service.get()
    qset = load_set(a.set)
    print(f"bo eval: {a.set} -- {len(qset)} cau hoi, "
          f"{sum(len(q['answers']) for q in qset)} dap an")
    print(f"LLM: {'TAT' if a.no_llm else 'BAT'} | weights: {weights or 'mac dinh'}\n")

    kv, kf = (1, 5, 10, 30), (10, 50, 100)
    rows = evaluate(s["engine"], s["store"], qset, kv, kf,
                    use_llm=not a.no_llm, weights=weights or None)
    out = summarize(rows, kv, kf)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"summary": out, "rows": rows}, f, ensure_ascii=False, indent=2)
        print(f"\nda ghi {a.json}")


if __name__ == "__main__":
    main()
