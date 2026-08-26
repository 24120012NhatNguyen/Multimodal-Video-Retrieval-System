#!/usr/bin/env python3
"""Chuan hoa toan bo file OCR trong data/artifacts/*/ocr/*.json.

BOI CANH
--------
Bo nhan dang OCR chay sai model ngon ngu: moi ky tu tieng Viet co dau deu hong
trong khi conf van 0.9+ (nhom hong co conf trung binh 0.897, CAO HON muc chung
0.874 -- nen khong the loc bang nguong conf).

KHONG THE PHUC HOI LAI DAU. Anh xa la nhieu-thanh-mot:

    'Bäc'  = 'Bắc'   ä -> ắ        'Chäy' = 'Cháy'  ä -> á
    'ngäy' = 'ngày'  ä -> à        'dät'  = 'đất'   ä -> ấ

Cung mot ky tu 'ä' dung cho 4 chu khac nhau. Day khong phai loi giai ma sai bang
ma (neu vay da phuc hoi duoc) ma la model chon glyph gan giong nhat. Vi vay thao
tac duy nhat dung dan la BO DAU ca hai phia -- tai lieu va truy van -- de chung
gap nhau o dang khong dau.

HAI NHOM LOI, XU LY KHAC NHAU
-----------------------------
1. Mojibake Latin-1 (o a e u ...): NFD roi bo ky tu to dau la xu ly duoc.

2. Ky tu KY HIEU thay cho chu co dau. Nhom nay bi tach tu lam MAT HAN tu:

       'Chäy l°n tai xu&ng'  -> ['chay','tai','xu','ng']   mat: lon, xuong
       'vi€c'                -> ['vi']                      mat: viec

   Bang anh xa duoi day dung tren thong ke ngu canh cua toan bo 502.558 item,
   khong phai suy doan tu vai vi du.
"""

import argparse
import json
import os
import re
import shutil
import sys
import unicodedata
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OCR_GLOB = "data/artifacts/*/ocr/*.json"

# --- Bang anh xa ky hieu -> chu cai ---------------------------------------
# Con so trong ngoac la tan suat do duoc tren toan bo corpus.
SYMBOL_MAP = {
    "@": "e",   # (1.623) s@=se, li@u=lieu, l@=le, nghi@p=nghiep -- rat nhat quan
    "€": "e",   # (202)   hi€u=hieu, nhi€u=nhieu, th€=the, tu€=tue
    "δ": "o",   # (233)   vδi=voi, Nδi=Noi, Khδi=Khoi, sδ=so, truδ=truong
    "&": "o",   # (7.412) da so tuong doi: &=o, l&=lo, kh&i=khoi, tr&=tro,
                #         tru&ng=truong, nu&c=nuoc. Phan con lai (x&=xu, s&=se)
                #         sinh token sai nhung vo hai -- no chi khong khop duoc,
                #         chu khong tao khop gia voi truy van that.
    "°": "o",   # (308)   CHI khi canh chu cai: l°n=lon, S°m=som, B°=Bo.
                #         74% con lai la '206:40:53°' -- ky hieu do that, giu.
}

# Ky hieu bi OCR chen vao giua hai tu -> tra ve khoang trang, khong phai chu.
SYMBOL_TO_SPACE = {"`"}

# Ky hieu chi anh xa khi co CHU CAI ke ben. Dung NGAY SAU chu so thi giu nguyen
# vi do la ky hieu that: '206:40:53°', '35°C'.
LETTER_CONTEXT_ONLY = {"°"}

# --- Chu so bi dung thay cho chu cai --------------------------------------
# CHI '6'. Do duoc 2.652 lan '6' nam giua hai chu cai, va cac tu deu nhat quan:
#   m6i=moi(249) gi6i=gioi(230) nu6c=nuoc(196) v6i=voi(155) l6n=lon(133)
#   g6i=goi(88) h6a=hoa(72) m6n=mon(58) d6ng=dong(51) n6i=noi(47)
# Cac chu so khac KHONG phai thay the: '7'/'9' la logo kenh (htv7hd, h7tv),
# '1'/'2' la don vi do (bam1m, nang1m). Dung vao chung se pha du lieu that.
DIGIT_MAP = {"6": "o"}


_TOKEN_SPLIT = re.compile(r"(\s+)")


def _token_has_other_digit(token, idx_in_token, ch):
    """Token con chu so nao KHAC vi tri dang xet khong?

    Chot chan de khong dung vao '2024', '206:40:53', 'HTV7' -- nhung chuoi ma
    chu so la chu so that. Tu tieng Viet bi hong ('h6a', 'l6n') chi co dung mot
    chu so va do chinh la ky tu bi thay.
    """
    for i, c in enumerate(token):
        if i != idx_in_token and c.isdigit():
            return True
    return False

# Dau nhay o moi bien the. OCR bieu dien dau moc (u', o') bang dau nhay, nen
# phai xoa TRUOC khi tach tu -- neu tach truoc, "LU'C" thanh ["lu","c"] va khong
# bao gio khop duoc "luc".
APOSTROPHES = "'’‘ʼ`´"


def _is_letter(c):
    return bool(c) and c.isalpha()


def map_symbols(text, counter=None):
    """Anh xa ky hieu/chu so theo ngu canh. Tra ve (text_moi, so_lan_doi)."""
    n = 0
    pieces = []

    # Tach giu nguyen khoang trang de con xac dinh duoc ranh gioi token.
    for token in _TOKEN_SPLIT.split(text):
        if not token or token.isspace():
            pieces.append(token)
            continue

        out = []
        for i, c in enumerate(token):
            before = token[i - 1] if i else ""
            after = token[i + 1] if i + 1 < len(token) else ""

            # --- ky hieu chen giua hai tu -> khoang trang ---
            if c in SYMBOL_TO_SPACE:
                if _is_letter(before) and _is_letter(after):
                    out.append(" ")
                    n += 1
                    if counter is not None:
                        counter[f"{c} -> khoang trang"] += 1
                else:
                    out.append(c)
                continue

            # --- chu so thay cho chu cai (chi '6') ---
            drepl = DIGIT_MAP.get(c)
            if drepl is not None:
                if (_is_letter(before)
                        and not _token_has_other_digit(token, i, c)):
                    out.append(drepl)
                    n += 1
                    if counter is not None:
                        counter[f"{c} -> {drepl}"] += 1
                else:
                    out.append(c)
                continue

            # --- ky hieu thay cho chu co dau ---
            repl = SYMBOL_MAP.get(c)
            if repl is None:
                out.append(c)
                continue

            if c in LETTER_CONTEXT_ONLY:
                # '206:40:53°' va '35°C': chu so ngay truoc -> ky hieu do that.
                if before.isdigit() or (before == "" and after.isdigit()):
                    out.append(c)
                    continue
                if not (_is_letter(before) or _is_letter(after) or before == ""):
                    out.append(c)
                    continue

            out.append(repl)
            n += 1
            if counter is not None:
                counter[f"{c} -> {repl}"] += 1

        pieces.append("".join(out))

    return "".join(pieces), n


def strip_apostrophes(text):
    """Xoa dau nhay. PHAI chay TRUOC map_symbols.

    OCR bieu dien dau moc bang dau nhay, va no chen vao giua chu cai voi ky tu
    can anh xa: "NU'6C" (= NUOC). Neu map_symbols chay truoc, phep kiem tra
    "co chu cai lien truoc" nhin thay dau nhay chu khong phai chu 'U', nen bo
    qua '6' va lam mat tu.
    """
    for ch in APOSTROPHES:
        text = text.replace(ch, "")
    return text


def strip_diacritics(text):
    """Bo dau thanh, GIU NGUYEN chu hoa/thuong.

    Giu nguyen kieu chu de panel UI con doc duoc ("NO LUC GIAM NOI LO HOC PHI"
    de nhin hon "no luc giam noi lo hoc phi"). Phia lap chi muc tu ha chu thuong
    bang tokenize_fold.
    """
    text = strip_apostrophes(text)
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.replace("đ", "d").replace("Đ", "D")
    text = text.replace("ð", "d").replace("Ð", "D")
    return unicodedata.normalize("NFC", text)


def normalize_text(text, counter=None, _max_pass=3):
    """Xoa dau nhay -> anh xa ky hieu -> bo dau thanh, lap den khi on dinh.

    Vong lap la de chac chan idempotent: bo dau co the tao ra ngu canh moi cho
    map_symbols (vd 'ö' thanh 'o' lam mot '6' ben canh du dieu kien anh xa).
    Chay lai script tren du lieu da chuan hoa phai khong doi gi.
    """
    cur = strip_apostrophes(str(text))
    for _ in range(_max_pass):
        mapped, _n = map_symbols(cur, counter)
        out = strip_diacritics(mapped)
        if out == cur:
            return out
        cur = out
        counter = None  # chi dem o luot dau, tranh dem trung
    return cur


# --------------------------------------------------------------------------
def process_file(path, stats, dedupe=True):
    """-> (data_moi, co_thay_doi). Khong ghi dia."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    changed = False
    for frame in data.get("frames") or []:
        items = frame.get("items") or []
        kept = []
        seen = set()
        for it in items:
            raw = it.get("text")
            if raw is None:
                kept.append(it)
                continue

            new = normalize_text(raw, stats["symbols"])
            stats["items"] += 1
            if new != raw:
                stats["items_changed"] += 1
                changed = True

            if dedupe:
                key = new.strip().lower()
                if key and key in seen:
                    stats["dedup"] += 1
                    changed = True
                    continue
                if key:
                    seen.add(key)

            if new != raw:
                it["text"] = new
            kept.append(it)

        if dedupe and len(kept) != len(items):
            frame["items"] = kept

    # Danh dau da chuan hoa de chay lai khong bi nhap nhang
    prov = data.setdefault("normalization", {})
    prov["diacritics"] = "stripped"
    prov["symbol_map"] = {k: v for k, v in SYMBOL_MAP.items()}
    prov["note"] = ("Dau tieng Viet da bi bo. Khong phuc hoi duoc vi OCR goc "
                    "anh xa nhieu-thanh-mot. Xem scripts/normalize_ocr.py")
    return data, changed


def main():
    ap = argparse.ArgumentParser(
        description="Chuan hoa OCR: bo dau + anh xa ky hieu + khu trung lap.")
    ap.add_argument("--glob", default=OCR_GLOB)
    ap.add_argument("--apply", action="store_true",
                    help="THUC SU ghi de file. Khong co co nay thi chi thu chay.")
    ap.add_argument("--backup", default="data/ocr_backup",
                    help="thu muc luu ban goc truoc khi ghi de")
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--no-dedupe", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    import glob as _glob

    files = sorted(_glob.glob(a.glob))
    if a.limit:
        files = files[:a.limit]
    if not files:
        print(f"Khong tim thay file nao khop {a.glob!r}")
        return 1

    stats = {"items": 0, "items_changed": 0, "dedup": 0,
             "symbols": Counter(), "files_changed": 0}
    samples = []

    for path in files:
        try:
            data, changed = process_file(path, stats, dedupe=not a.no_dedupe)
        except Exception as e:
            print(f"  LOI {path}: {type(e).__name__}: {e}")
            continue

        if not changed:
            continue
        stats["files_changed"] += 1

        if len(samples) < 6:
            for fr in (data.get("frames") or [])[:4]:
                for it in (fr.get("items") or [])[:3]:
                    t = it.get("text")
                    if t and len(samples) < 6:
                        samples.append((os.path.basename(path), t))

        if not a.apply:
            continue

        if not a.no_backup:
            dst = os.path.join(a.backup, os.path.relpath(path, "data/artifacts"))
            if not os.path.exists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(path, dst)

        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp, path)

    mode = "DA GHI DE" if a.apply else "THU CHAY (chua ghi gi)"
    print(f"\n=== {mode} ===")
    print(f"  file quet     : {len(files)}")
    print(f"  file thay doi : {stats['files_changed']}")
    print(f"  item          : {stats['items']}")
    print(f"  item doi      : {stats['items_changed']} "
          f"({100 * stats['items_changed'] / max(1, stats['items']):.1f}%)")
    print(f"  item trung bo : {stats['dedup']}")
    if stats["symbols"]:
        print("  anh xa ky hieu:")
        for k, n in stats["symbols"].most_common():
            print(f"     {k:22} {n}")
    if samples:
        print("  mau sau chuan hoa:")
        for name, t in samples:
            print(f"     [{name}] {t[:64]!r}")
    if not a.apply:
        print("\n  Them --apply de ghi de that su. "
              f"Ban goc se duoc chep sang {a.backup}/ truoc khi ghi.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
