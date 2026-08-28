#!/usr/bin/env python3
"""Trich truoc anh keyframe vao data/keyframe_cache.

KHONG lien quan gi den frontend/public/static/images/Keyframes -- thu muc do la
cua he cu va KHONG con ai doc. Duong dan anh ma backend tra ve bay gio deu la
/keyframe/{video}/{frame}.jpg, do socket_app.py phuc vu bang cach trich thang tu
data/videos/*.mp4.

Vi sao nen lam am truoc:

    trich le tung anh (ffmpeg seek)   ~130 ms/anh
    warm_video (mot lan doc file)      ~35 ms/anh

Luoi 500 anh mo lan dau ma chua co cache thi la 500 lan ffmpeg seek chay song
song -- may dung hinh. Do la loi "Ca video bi treo" da gap.

    python scripts/prewarm_keyframes.py                 # ca corpus
    python scripts/prewarm_keyframes.py --packs L24 L25 L30
    python scripts/prewarm_keyframes.py --videos L25_V013 L24_V026
    python scripts/prewarm_keyframes.py --workers 8 --dry-run
"""

import argparse
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.config import KEYFRAME_CACHE
from retrieval.frames import KeyframeImages, cache_path
from retrieval.store import ArtifactStore

KB_PER_FRAME = 125          # do tren du lieu that
MARGIN_GB = 3.0             # chua trong o dia sau khi lam am xong


def missing_of(store, video_id):
    df = store.frames_of(video_id)
    if df.empty:
        return 0
    return sum(1 for r in df.itertuples()
               if not os.path.exists(cache_path(video_id, r.frame_idx)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs", nargs="*", help="vd: L24 L25 L30")
    ap.add_argument("--videos", nargs="*")
    ap.add_argument("--workers", type=int, default=6,
                    help="so video lam am song song (mac dinh 6)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="bo qua kiem tra dung luong o dia")
    a = ap.parse_args()

    store = ArtifactStore()
    imgs = KeyframeImages(store)

    vids = list(store.video_ids)
    if a.videos:
        want = set(a.videos)
        vids = [v for v in vids if v in want]
    elif a.packs:
        want = set(a.packs)
        vids = [v for v in vids if v.split("_")[0] in want]
    if not vids:
        print("khong co video nao khop dieu kien")
        return 1

    print(f"quet {len(vids)} video ...", flush=True)
    todo = [(v, n) for v in vids if (n := missing_of(store, v)) > 0]
    n_missing = sum(n for _, n in todo)
    n_have = sum(len(store.frames_of(v)) for v in vids) - n_missing

    print(f"  da co san : {n_have:,} anh")
    print(f"  con thieu : {n_missing:,} anh tren {len(todo)} video")
    if n_missing == 0:
        print("  -> khong co gi de lam")
        return 0

    need_gb = n_missing * KB_PER_FRAME / 1e6
    free_gb = shutil.disk_usage(os.path.dirname(KEYFRAME_CACHE) or ".").free / 1e9
    print(f"  can khoang {need_gb:.1f} GB, o dia con {free_gb:.1f} GB")

    if not a.force and free_gb - need_gb < MARGIN_GB:
        print(f"\n  DUNG LAI: sau khi lam am chi con {free_gb - need_gb:.1f} GB.")
        old = "frontend/public/static/images/Keyframes"
        if os.path.isdir(old):
            sz = sum(os.path.getsize(os.path.join(r, f))
                     for r, _, fs in os.walk(old) for f in fs) / 1e9
            print(f"  Goi y: {old} dang chiem {sz:.1f} GB va KHONG con ai doc")
            print(f"         (moi duong dan anh bay gio deu la /keyframe/...).")
            print(f"         Xoa no la du cho. Dung --force de bo qua kiem tra nay.")
        return 1

    if a.dry_run:
        print("\n  --dry-run: dung o day")
        return 0

    print(f"\nlam am bang {a.workers} luong song song ...\n", flush=True)
    t0 = time.time()
    done_frames = 0
    done_videos = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(imgs.warm_video, v): (v, n) for v, n in todo}
        for fut in as_completed(futs):
            v, n = futs[fut]
            try:
                got = fut.result()
            except Exception as e:
                print(f"  {v}: LOI {type(e).__name__}: {e}", flush=True)
                got = 0
            done_frames += got
            done_videos += 1
            el = time.time() - t0
            rate = done_frames / el if el else 0
            left = (n_missing - done_frames) / rate if rate else 0
            print(f"  [{done_videos}/{len(todo)}] {v}: {got} anh "
                  f"| {rate:.0f} anh/s | con ~{left/60:.0f} phut", flush=True)

    el = time.time() - t0
    print(f"\nxong: {done_frames:,} anh trong {el/60:.1f} phut "
          f"({done_frames/el:.0f} anh/s)")
    print(f"cache: {imgs.stats()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
