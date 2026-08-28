#!/usr/bin/env python3
"""Chuyen anh keyframe da trich sang dung cho ma backend doc.

data/extract_keyframes.py ghi ra:

    frontend/public/static/images/Keyframes/{pack}/{Vxxx}/{frame:06d}.jpg

Backend doc o:

    data/keyframe_cache/{pack}_{Vxxx}/{frame:06d}.jpg

Chi lech MOT CAP thu muc -- ten file va so hieu frame da khop hoan toan (kiem
tren 14 video: 3.987 JPG, khong thieu khong thua cai nao). Nen day chi la doi
ten thu muc, KHONG copy du lieu, va vi hai cho nam cung o dia nen tuc thoi.

    python scripts/migrate_keyframes.py --dry-run
    python scripts/migrate_keyframes.py
    python scripts/migrate_keyframes.py --copy     # giu lai ban cu
"""

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.config import KEYFRAME_CACHE

SRC = os.path.join("frontend", "public", "static", "images", "Keyframes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dst", default=KEYFRAME_CACHE)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--copy", action="store_true",
                    help="copy thay vi doi ten (cham va ton gap doi cho)")
    ap.add_argument("--verify", action="store_true",
                    help="doi chieu ten file voi frame_idx trong CSV truoc khi chuyen")
    a = ap.parse_args()

    if not os.path.isdir(a.src):
        print(f"khong thay {a.src}")
        return 1

    store = None
    if a.verify:
        from retrieval.store import ArtifactStore
        store = ArtifactStore()

    plan, n_files, n_skip, warn = [], 0, 0, []
    for pack in sorted(os.listdir(a.src)):
        pdir = os.path.join(a.src, pack)
        if not os.path.isdir(pdir):
            continue
        for v in sorted(os.listdir(pdir)):
            vdir = os.path.join(pdir, v)
            if not os.path.isdir(vdir):
                continue
            jpgs = [f for f in os.listdir(vdir) if f.endswith(".jpg")]
            if not jpgs:
                continue
            full = f"{pack}_{v}"
            out = os.path.join(a.dst, full)

            if store is not None:
                df = store.frames_of(full)
                if df.empty:
                    warn.append(f"{full}: khong co trong index, van chuyen")
                else:
                    want = {int(x) for x in df["frame_idx"]}
                    have = {int(f[:-4]) for f in jpgs}
                    if want - have:
                        warn.append(f"{full}: thieu {len(want - have)} frame so voi CSV")
                    if have - want:
                        warn.append(f"{full}: thua {len(have - want)} frame khong co trong CSV")

            if os.path.isdir(out):
                n_skip += 1
                continue
            plan.append((vdir, out, len(jpgs)))
            n_files += len(jpgs)

    print(f"  {len(plan)} thu muc video se chuyen, {n_files:,} anh")
    if n_skip:
        print(f"  {n_skip} thu muc da co san o dich -> bo qua")
    for w in warn[:10]:
        print(f"  [chu y] {w}")
    if len(warn) > 10:
        print(f"  ... va {len(warn) - 10} canh bao nua")
    if not plan:
        print("  khong co gi de chuyen")
        return 0

    for src, dst, n in plan[:3]:
        print(f"    {src}  ->  {dst}  ({n} anh)")
    if len(plan) > 3:
        print(f"    ... {len(plan) - 3} thu muc nua")

    if a.dry_run:
        print("\n  --dry-run: khong dong gi ca")
        return 0

    os.makedirs(a.dst, exist_ok=True)
    done = 0
    for src, dst, n in plan:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            if a.copy:
                shutil.copytree(src, dst)
            else:
                os.rename(src, dst)      # cung o dia -> tuc thoi
            done += 1
        except OSError as e:
            print(f"  loi {src}: {e}")
    print(f"\n  da chuyen {done}/{len(plan)} thu muc")
    if not a.copy:
        print(f"  (doi ten, khong copy -- {a.src} gio con lai thu muc rong)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
