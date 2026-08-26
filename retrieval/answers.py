"""Viec 2 -- schema dap an.

Tu dict phang doi sang LIST CO THU TU:

    [{"video_id": "L24_V007", "frame_idx": 12450, "source": "manual"}, ...]

`source` nhan "manual" (nguoi chon) hoac "autofill" (may lap). Viec 3 can dung
truong nay de khong day phan doan cua nguoi dung xuong duoi.

Dinh dang file nop bai cuoi cung giu nguyen, nen entry con mang hai truong tuy
chon: `answer` (dinh dang QA) va `frames` (dinh dang TRAKE, mot dong nhieu
frame). `frame_idx` luon bang frames[0] khi co frames.
"""

MANUAL = "manual"
AUTOFILL = "autofill"
_SOURCES = (MANUAL, AUTOFILL)


def make(video_id, frame_idx, source=MANUAL, answer=None, frames=None, **extra):
    if source not in _SOURCES:
        source = MANUAL
    if frames:
        frames = [int(f) for f in frames]
        frame_idx = frames[0]
    entry = {
        "video_id": str(video_id),
        "frame_idx": int(frame_idx),
        "source": source,
    }
    if answer is not None:
        entry["answer"] = answer
    if frames and len(frames) > 1:
        entry["frames"] = frames
    for k, v in extra.items():
        if v is not None:
            entry[k] = v
    return entry


def key_of(entry):
    return (entry["video_id"], int(entry["frame_idx"]))


def normalize(lst, resolver=None):
    """Doc duoc moi dinh dang tung ton tai, tra ve list schema moi.

    resolver(idx) -> (video_id, frame_idx) dung cho cac ban ghi cu chi luu chi
    so keyframe toan cuc. Thieu resolver thi ban ghi kieu do bi bo qua thay vi
    doan bua.
    """
    out = []
    seen = set()
    for ans in (lst or []):
        entry = None

        if isinstance(ans, dict):
            if "video_id" in ans and "frame_idx" in ans:
                entry = make(
                    ans["video_id"], ans["frame_idx"],
                    ans.get("source", MANUAL),
                    ans.get("answer"), ans.get("frames"),
                    pts_time=ans.get("pts_time"), reason=ans.get("reason"),
                )
            elif "video" in ans and ans.get("frames"):
                # dinh dang cu: {"video", "frames", "answer", "idxs"}
                entry = make(ans["video"], ans["frames"][0], MANUAL,
                             ans.get("answer"), ans.get("frames"))

        elif isinstance(ans, int) and resolver is not None:
            hit = resolver(ans)
            if hit:
                entry = make(hit[0], hit[1], MANUAL)

        if entry is None:
            continue
        k = key_of(entry)
        if k in seen:
            continue
        seen.add(k)
        out.append(entry)
    return out


def reorder(lst, order):
    """Sap xep lai theo danh sach khoa [(video_id, frame_idx)] hoac list entry."""
    by_key = {key_of(e): e for e in lst}
    out, used = [], set()
    for it in (order or []):
        if isinstance(it, dict):
            k = (it.get("video_id"), int(it.get("frame_idx", -1)))
        else:
            k = (it[0], int(it[1]))
        e = by_key.get(k)
        if e is not None and k not in used:
            used.add(k)
            out.append(e)
    # bat ky entry nao khong duoc nhac den van phai giu lai, xep sau
    for e in lst:
        if key_of(e) not in used:
            out.append(e)
    return out


def split_by_source(lst):
    return ([e for e in lst if e.get("source") == MANUAL],
            [e for e in lst if e.get("source") != MANUAL])


# --------------------------------------------------------------------------
# Dinh dang file nop bai -- KHONG doi.
def kis_csv(lst):
    return "".join(f"{e['video_id']},{int(e['frame_idx'])}\n" for e in lst)


def qa_csv(lst):
    return "".join(
        f"{e['video_id']},{int(e['frame_idx'])},{e.get('answer') or ''}\n"
        for e in lst
    )


def trake_csv(lst):
    lines = []
    for e in lst:
        frames = e.get("frames") or [e["frame_idx"]]
        lines.append(e["video_id"] + "," + ",".join(str(int(f)) for f in frames))
    return "\n".join(lines) + "\n" if lines else ""
