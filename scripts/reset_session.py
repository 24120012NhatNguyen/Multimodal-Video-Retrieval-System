#!/usr/bin/env python3
"""Xoa sach du lieu cua PHIEN ANSWER truoc do.

Trang thai mot phien nam o `back_up/*.json` (socket_app.py doc luc khoi dong,
ghi lai sau moi thao tac):

    answer.json         dap an tung cau, list co thu tu theo schema Viec 2
    answer_ignore.json  cac frame da bam bo qua
    user.json           ai dang giu cau nao
    reorder_status.json trang thai sap xep lai

Ngoai ra co hai loai cache SINH RA DUOC, xoa di chi mat thoi gian:

    data/clip_cache/    clip mp4 da cat quanh keyframe -- cat lai tu data/videos
    __pycache__/        bytecode Python

KHONG BAO GIO dung toi (chuong trinh se tu chan, xem PROTECTED):

    data/keyframe_cache 32 GB, 154.640 anh -- tai lai mat hang gio
    data/artifacts      2,6 GB features/asr/ocr cua BTC
    data/videos         nguon goc, BTC khong chia se
    data/ocr_backup     ban goc OCR truoc khi chuan hoa
    config/fusion.json  tham so da hieu chinh bang bo eval

Hai bo dem LLM (`data/decompose_cache.json`, `data/translation_cache.json`)
KHONG bi xoa mac dinh: chung khong thuoc ve mot phien, chung la thu giu cho LLM
tat dinh giua hai lan chay -- xoa di thi cung mot truy van go lai giua buoi thi
se ra ket qua khac, va phai goi lai API (mat tien). Muon xoa thi noi ro:

    python scripts/reset_session.py                  # trang thai + clip + pycache
    python scripts/reset_session.py --dry-run        # chi xem, khong xoa
    python scripts/reset_session.py --llm            # xoa them bo dem LLM
    python scripts/reset_session.py --all -y         # xoa tat, khong hoi
    python scripts/reset_session.py --only state     # chi trang thai phien
    python scripts/reset_session.py --keep-user      # giu user.json (ai giu cau nao)

Mac dinh van CHEP MOT BAN vao back_up/_archive/<thoi-diem>/ truoc khi xoa, de
lo tay con lay lai duoc. `--no-archive` de tat.
"""

import argparse
import datetime
import glob
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BACK_UP = os.environ.get("BACK_UP_DIR", os.path.join(ROOT, "back_up"))
ARCHIVE = os.path.join(BACK_UP, "_archive")

STATE_FILES = ["answer.json", "answer_ignore.json", "user.json",
               "reorder_status.json"]


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


CLIP_CACHE = _abs(os.environ.get("CLIP_CACHE", "data/clip_cache"))
DECOMPOSE_CACHE = _abs(os.environ.get("DECOMPOSE_CACHE",
                                      "data/decompose_cache.json"))
TRANSLATION_CACHE = _abs(os.environ.get("TRANSLATION_CACHE",
                                        "data/translation_cache.json"))

# Duong dan tuyet doi KHONG duoc phep xoa. Kiem tra bang cach so tien to duong
# dan da chuan hoa, nen mot ky tu dai dien lo tay cung khong cham toi duoc.
PROTECTED = [
    _abs(os.environ.get("KEYFRAME_CACHE", "data/keyframe_cache")),
    _abs(os.environ.get("ARTIFACT_ROOT", "data/artifacts")),
    _abs(os.environ.get("VIDEO_DIR", "data/videos")),
    _abs("data/ocr_backup"),
    _abs(os.environ.get("FUSION_CONFIG", "config/fusion.json")),
    ARCHIVE,
]

GROUPS = ("state", "clips", "pycache", "llm")


def is_protected(path):
    p = os.path.realpath(path)
    for guard in PROTECTED:
        g = os.path.realpath(guard)
        if p == g or p.startswith(g + os.sep):
            return guard
    return None


def size_of(path):
    if os.path.isfile(path):
        try:
            return os.path.getsize(path)
        except OSError:
            return 0
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for fn in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, fn))
            except OSError:
                pass
    return total


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0


def collect(groups, keep_user):
    """Tra ve list (nhom, duong_dan) that su ton tai."""
    targets = []

    if "state" in groups:
        for name in STATE_FILES:
            if keep_user and name == "user.json":
                continue
            p = os.path.join(BACK_UP, name)
            if os.path.exists(p):
                targets.append(("state", p))
            # socket_app.py ghi qua file .tmp roi os.replace; mot lan tat may
            # dung luc do co the de lai rac
            tmp = p + ".tmp"
            if os.path.exists(tmp):
                targets.append(("state", tmp))

    if "clips" in groups and os.path.isdir(CLIP_CACHE):
        # Xoa NOI DUNG chu khong xoa chinh thu muc: retrieval/clips.py coi su
        # ton tai cua thu muc goc la chuyen duong nhien.
        for entry in sorted(os.listdir(CLIP_CACHE)):
            targets.append(("clips", os.path.join(CLIP_CACHE, entry)))

    if "pycache" in groups:
        for p in sorted(glob.glob(os.path.join(ROOT, "**", "__pycache__"),
                                  recursive=True)):
            if "node_modules" in p or os.sep + ".git" + os.sep in p:
                continue
            targets.append(("pycache", p))

    if "llm" in groups:
        for p in (DECOMPOSE_CACHE, TRANSLATION_CACHE):
            if os.path.exists(p):
                targets.append(("llm", p))

    return targets


def server_dang_chay():
    """socket_app.py giu trang thai TRONG BO NHO va ghi de len dia sau moi thao
    tac. Xoa file trong luc no dang chay thi thao tac ke tiep se dung ngay ban
    trong bo nho ghi de lai -- coi nhu chua xoa gi."""
    try:
        out = subprocess.run(["pgrep", "-af", "socket_app"],
                             capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return []
    return [ln for ln in out.stdout.splitlines()
            if ln.strip() and "reset_session" not in ln]


def archive(targets):
    """Chep ban trang thai hien tai truoc khi xoa. Chi chep nhom `state`: cac
    nhom con lai deu sinh lai duoc, chep chi ton o dia."""
    files = [p for g, p in targets if g == "state" and os.path.isfile(p)]
    if not files:
        return None
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dst = os.path.join(ARCHIVE, stamp)
    os.makedirs(dst, exist_ok=True)
    for p in files:
        shutil.copy2(p, os.path.join(dst, os.path.basename(p)))
    return dst


def remove(path):
    if os.path.islink(path) or os.path.isfile(path):
        os.remove(path)
    else:
        shutil.rmtree(path)


def main():
    ap = argparse.ArgumentParser(
        description="Xoa du lieu phien answer truoc do.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Hai bo dem LLM")[0])
    ap.add_argument("--only", action="append", choices=GROUPS, metavar="NHOM",
                    help="chi xoa nhom nay (lap lai duoc): "
                         + ", ".join(GROUPS))
    ap.add_argument("--llm", action="store_true",
                    help="xoa them bo dem dich + phan ra cua LLM")
    ap.add_argument("--all", action="store_true",
                    help="xoa tat ca cac nhom, ke ca bo dem LLM")
    ap.add_argument("--keep-user", action="store_true",
                    help="giu user.json (ai dang giu cau nao)")
    ap.add_argument("--no-archive", action="store_true",
                    help="khong chep ban luu truoc khi xoa")
    ap.add_argument("--dry-run", action="store_true",
                    help="chi liet ke, khong xoa gi")
    ap.add_argument("-y", "--yes", action="store_true",
                    help="khong hoi lai")
    a = ap.parse_args()

    if a.all:
        groups = set(GROUPS)
    elif a.only:
        groups = set(a.only)
    else:
        groups = {"state", "clips", "pycache"}
        if a.llm:
            groups.add("llm")
    if a.llm:
        groups.add("llm")

    targets = collect(groups, a.keep_user)

    # Chan truoc khi in bat cu thu gi: mot duong dan bi cau hinh tro nham vao
    # vung 32 GB thi phai chet o day, khong phai sau khi nguoi dung bam "co".
    for _g, p in targets:
        guard = is_protected(p)
        if guard:
            print(f"DUNG: {p} nam trong vung duoc bao ve ({guard}). "
                  f"Kiem tra lai bien moi truong.", file=sys.stderr)
            return 2

    if not targets:
        print("Khong co gi de xoa -- phien da sach.")
        return 0

    print(f"Goc: {ROOT}")
    print(f"Nhom: {', '.join(sorted(groups))}\n")
    tong = 0
    for g in sorted(groups):
        items = [p for gr, p in targets if gr == g]
        if not items:
            continue
        cum = sum(size_of(p) for p in items)
        tong += cum
        print(f"  [{g}] {len(items)} muc, {human(cum)}")
        for p in items[:8]:
            print(f"      {os.path.relpath(p, ROOT)}")
        if len(items) > 8:
            print(f"      ... va {len(items) - 8} muc nua")
    print(f"\nTong: {len(targets)} muc, {human(tong)}")

    if a.dry_run:
        print("\n--dry-run: khong xoa gi.")
        return 0

    dang_chay = server_dang_chay()
    if dang_chay:
        print("\nCANH BAO: socket_app.py dang chay:")
        for ln in dang_chay:
            print(f"      {ln}")
        print("      No giu dap an TRONG BO NHO va se ghi de len dia sau thao"
              "\n      tac ke tiep. Tat server truoc, xoa xong roi bat lai.")

    if not a.yes:
        try:
            tra_loi = input("\nXoa? [y/N] ").strip().lower()
        except EOFError:
            tra_loi = ""
        if tra_loi not in ("y", "yes"):
            print("Bo qua.")
            return 1

    luu = None
    if not a.no_archive:
        luu = archive(targets)

    da_xoa, loi = 0, 0
    for _g, p in targets:
        try:
            remove(p)
            da_xoa += 1
        except OSError as e:
            print(f"  khong xoa duoc {p}: {e}", file=sys.stderr)
            loi += 1

    if luu:
        print(f"\nBan luu: {os.path.relpath(luu, ROOT)}")
    print(f"Da xoa {da_xoa}/{len(targets)} muc, giai phong ~{human(tong)}.")
    if loi:
        print(f"{loi} muc that bai.", file=sys.stderr)
        return 1
    if dang_chay:
        print("Nho khoi dong lai socket_app.py de no nap lai trang thai rong.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
