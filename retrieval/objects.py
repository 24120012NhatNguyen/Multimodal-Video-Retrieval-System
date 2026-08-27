"""Index object cua BTC -- tra cuu theo LOP + VI TRI trong khong gian.

Vi sao phai dung index rieng: objects-aic25-b1 co 177.321 file JSON (2,1 GB).
Doc chung luc truy van la khong the -- do mot lan mat vai phut. Nen gom thanh
mot .npz duy nhat (~50 MB) roi nap mmap.

Cach xep (CSR, giong ma tran thua):

    kf_*      moi dong la MOT keyframe cua BTC (video, thu tu n, pts_time)
    kf_ptr    kf_ptr[i]:kf_ptr[i+1] la lat cat detection cua keyframe i
    det_*     detection phang: lop, diem, hop bao

Toa do hop: OpenImages ghi [ymin, xmin, ymax, xmax] da chuan hoa ve [0,1] --
DA KIEM CHUNG tren du lieu that (cot 0 max 0.727 < cot 2 min 0.500). Panel cua
UI lai goi truc doc la "x" (xTop = offsetTop/PANEL) va truc ngang la "y", nen
anh xa dung la xTop->ymin, yTop->xmin. Nham cap nay thi hop bi lat cheo va ket
qua tim theo vi tri sai hoan toan ma khong he bao loi.
"""

import csv
import glob
import json
import os
import time

import numpy as np

from retrieval.config import ARTIFACT_ROOT, OBJECT_DIR

# Cho index vao dau -- va tim o dau khi nap.
#
# Tren Kaggle, ARTIFACT_ROOT tro sang /kaggle/input/... la mount CHI DOC: dung
# mot duong dan duy nhat o do thi vua khong ghi duoc luc dung, vua khong co file
# luc nap, va kenh object chet am tham. Nen tim theo nhieu cho, va ghi vao cho
# dau tien GHI DUOC.
_INDEX_NAME = "object_index.npz"
INDEX_SEARCH_PATH = [
    p for p in (
        os.environ.get("OBJECT_INDEX"),
        os.path.join(ARTIFACT_ROOT, _INDEX_NAME),
        os.path.join("data", _INDEX_NAME),
        _INDEX_NAME,
    ) if p
]


def find_index():
    """Duong dan index dang co, hoac None."""
    for p in INDEX_SEARCH_PATH:
        if os.path.exists(p):
            return p
    return None


def default_out():
    """Cho ghi index: cho dau tien trong danh sach ma thu muc cha GHI DUOC."""
    for p in INDEX_SEARCH_PATH:
        d = os.path.dirname(p) or "."
        if os.path.isdir(d) and os.access(d, os.W_OK):
            return p
        if not os.path.exists(d):
            try:
                os.makedirs(d, exist_ok=True)
                return p
            except OSError:
                continue
    return os.path.join("data", _INDEX_NAME)


INDEX_PATH = find_index() or default_out()

# Detection duoi nguong nay bi loai ngay tu luc dung index: 100 detection/file
# thi ~80 la nhieu duoi 0.15, giu lai chi lam index phinh 5 lan.
BUILD_MIN_SCORE = 0.15

# --- Ten COCO tren UI -> ten OpenImages trong du lieu ----------------------
# Bang icon cua UI la 80 lop COCO; du lieu object cua BTC la OpenImages (545
# lop). 69/80 ten trung nhau khi bo qua hoa thuong, 11 ten con lai khac han --
# nguoi dung bam icon "tv" se khong ra gi ma UI khong bao gi ca.
# ("parking meter" va "hair drier" khong co doi ung trong OpenImages, de nguyen
# de bao "khong co trong du lieu" -- do la su that.)
COCO_ALIAS = {
    "frisbee": "Flying disc",
    "skis": "Ski",
    "sports ball": "Ball",
    "donut": "Doughnut",
    "potted plant": "Houseplant",
    "dining table": "Table",
    "tv": "Television",
    "keyboard": "Computer keyboard",
    "cell phone": "Mobile phone",
}

MAP_KEYFRAME_DIRS = (
    os.path.join(ARTIFACT_ROOT, "map-keyframes-aic25-b1", "map-keyframes"),
    os.path.join(ARTIFACT_ROOT, "map-keyframes"),
)


def _map_dir():
    return next((d for d in MAP_KEYFRAME_DIRS if os.path.isdir(d)), None)


def _read_map(path):
    """{n: pts_time} tu map-keyframes cua BTC."""
    out = {}
    try:
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                out[int(r["n"])] = float(r["pts_time"])
    except Exception:
        return {}
    return out


# ---------------------------------------------------------------------------
def build_index(object_dir=OBJECT_DIR, out=None, min_score=BUILD_MIN_SCORE,
                progress=None):
    """Quet toan bo objects/*/*.json -> mot file .npz. Chay mot lan."""
    out = out or default_out()
    md = _map_dir()
    if md is None:
        raise RuntimeError(
            f"khong tim thay map-keyframes trong {MAP_KEYFRAME_DIRS} -- "
            f"object khong co moc thoi gian de bac cau")
    if not os.path.isdir(object_dir):
        raise RuntimeError(f"khong tim thay thu muc object: {object_dir}")

    videos, ent_code = [], {}
    kf_video, kf_n, kf_pts, kf_len = [], [], [], []
    det_ent, det_score, det_box = [], [], []

    t0 = time.time()
    vids = sorted(os.listdir(object_dir))
    for vi, vid in enumerate(vids):
        d = os.path.join(object_dir, vid)
        if not os.path.isdir(d):
            continue
        times = _read_map(os.path.join(md, f"{vid}.csv"))
        if not times:
            continue
        videos.append(vid)
        v_code = len(videos) - 1

        names = sorted(f for f in os.listdir(d) if f.endswith(".json"))
        for name in names:
            try:
                n = int(name[:-5])
            except ValueError:
                continue
            t = times.get(n)
            if t is None:
                continue
            try:
                with open(os.path.join(d, name), encoding="utf-8") as f:
                    j = json.load(f)
            except Exception:
                continue

            ents = j.get("detection_class_entities") or []
            scores = j.get("detection_scores") or []
            boxes = j.get("detection_boxes") or []
            cnt = 0
            for k, e in enumerate(ents):
                try:
                    s = float(scores[k])
                except (IndexError, TypeError, ValueError):
                    continue
                if s < min_score:
                    # danh sach da sap giam dan theo diem -> gap cai dau tien
                    # duoi nguong la dung duoc
                    break
                c = ent_code.get(e)
                if c is None:
                    c = ent_code[e] = len(ent_code)
                try:
                    b = [float(x) for x in boxes[k]]
                except (IndexError, TypeError, ValueError):
                    b = [0.0, 0.0, 1.0, 1.0]
                det_ent.append(c)
                det_score.append(s)
                det_box.append(b)
                cnt += 1

            kf_video.append(v_code)
            kf_n.append(n)
            kf_pts.append(t)
            kf_len.append(cnt)

        if progress and vi % 50 == 0:
            progress(vi, len(vids), time.time() - t0)

    kf_ptr = np.zeros(len(kf_len) + 1, dtype=np.int64)
    np.cumsum(np.asarray(kf_len, dtype=np.int64), out=kf_ptr[1:])

    entities = [None] * len(ent_code)
    for e, c in ent_code.items():
        entities[c] = e

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez(
        out,
        videos=np.array(videos),
        entities=np.array(entities),
        kf_video=np.asarray(kf_video, dtype=np.int32),
        kf_n=np.asarray(kf_n, dtype=np.int32),
        kf_pts=np.asarray(kf_pts, dtype=np.float32),
        kf_ptr=kf_ptr,
        det_ent=np.asarray(det_ent, dtype=np.int16),
        det_score=np.asarray(det_score, dtype=np.float16),
        det_box=np.asarray(det_box, dtype=np.float16),
        min_score=np.float32(min_score),
    )
    return {"n_video": len(videos), "n_keyframe": len(kf_len),
            "n_detection": len(det_ent), "n_entity": len(entities),
            "path": out, "giay": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------------
def iou(box, q):
    """IoU giua mot mang hop (N,4) va mot hop truy van, ca hai [ymin,xmin,ymax,xmax]."""
    y0 = np.maximum(box[:, 0], q[0])
    x0 = np.maximum(box[:, 1], q[1])
    y1 = np.minimum(box[:, 2], q[2])
    x1 = np.minimum(box[:, 3], q[3])
    inter = np.clip(y1 - y0, 0, None) * np.clip(x1 - x0, 0, None)
    a = np.clip(box[:, 2] - box[:, 0], 0, None) * np.clip(box[:, 3] - box[:, 1], 0, None)
    b = max(0.0, q[2] - q[0]) * max(0.0, q[3] - q[1])
    denom = a + b - inter
    return np.where(denom > 0, inter / np.maximum(denom, 1e-9), 0.0)


class ObjectIndex:
    """Tra cuu keyframe BTC theo lop object va vi tri.

    khong co file index -> `self.ok` False, moi truy van tra ve rong kem ly do.
    Khong tu dung index luc khoi dong: mat vai phut, khong duoc phep xay ra
    giua buoi thi.
    """

    def __init__(self, path=None):
        self.path = path or find_index() or INDEX_PATH
        path = self.path
        self.ok = False
        self.error = None
        if not os.path.exists(path):
            self.error = (
                f"khong tim thay {_INDEX_NAME} o bat ky cho nao trong "
                f"{INDEX_SEARCH_PATH} -- chay: python -m retrieval.objects build "
                f"(hoac dat OBJECT_INDEX tro toi file da dung san). Kenh object "
                f"se TAT, /panel tra ve rong.")
            return
        try:
            z = np.load(path, allow_pickle=False)
            self.videos = [str(v) for v in z["videos"]]
            self.entities = [str(e) for e in z["entities"]]
            self.kf_video = z["kf_video"]
            self.kf_n = z["kf_n"]
            self.kf_pts = z["kf_pts"]
            self.kf_ptr = z["kf_ptr"]
            self.det_ent = z["det_ent"]
            self.det_score = z["det_score"].astype(np.float32)
            self.det_box = z["det_box"].astype(np.float32)
            self.min_score = float(z["min_score"])
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            return

        self.vcode = {v: i for i, v in enumerate(self.videos)}
        # ten lop -> ma, khong phan biet hoa thuong (UI go "person", du lieu ghi "Person")
        self.ecode = {}
        for i, e in enumerate(self.entities):
            self.ecode.setdefault(e.lower(), i)

        n_kf = len(self.kf_video)
        # detection -> keyframe chua no
        self.det_kf = np.repeat(np.arange(n_kf, dtype=np.int32),
                                np.diff(self.kf_ptr))
        # Nghich dao: voi moi lop, danh sach detection cua no, sap theo keyframe.
        order = np.lexsort((self.det_kf, self.det_ent))
        self._order = order
        ents_sorted = self.det_ent[order]
        self._ent_start = np.searchsorted(
            ents_sorted, np.arange(len(self.entities), dtype=np.int16), "left")
        self._ent_end = np.searchsorted(
            ents_sorted, np.arange(len(self.entities), dtype=np.int16), "right")
        self.ok = True

    # ------------------------------------------------------------------
    def resolve(self, name):
        """Ten lop nguoi dung go -> ma. Chap nhan sai hoa thuong va gach duoi."""
        if not name:
            return None
        s = str(name).strip().lower().replace("_", " ")
        s = COCO_ALIAS.get(s, s).lower()
        c = self.ecode.get(s)
        if c is not None:
            return c
        # tim theo tien to, uu tien ten ngan nhat (tranh "Person" -> "Personal care")
        cands = [i for lo, i in self.ecode.items() if lo.startswith(s)]
        if cands:
            return min(cands, key=lambda i: len(self.entities[i]))
        return None

    def _dets_of(self, code):
        """(chi so detection, keyframe, diem, hop) cua mot lop."""
        idx = self._order[self._ent_start[code]:self._ent_end[code]]
        return idx, self.det_kf[idx], self.det_score[idx], self.det_box[idx]

    def vocabulary(self, top=None):
        """[(ten_lop, so_detection)] giam dan -- de UI goi y tag."""
        cnt = np.bincount(self.det_ent.astype(np.int64),
                          minlength=len(self.entities))
        order = np.argsort(-cnt)
        out = [(self.entities[i], int(cnt[i])) for i in order if cnt[i] > 0]
        return out[:top] if top else out

    # ------------------------------------------------------------------
    def search(self, tags=(), boxes=(), counts=None, restrict_videos=None,
               topk=500, min_score=None):
        """Keyframe BTC thoa MOI rang buoc, xep theo do manh bang chung.

        tags       ten lop bat buoc co mat, vd ["Person", "Dog"]
        boxes      [{"type": ten_lop, "box": [ymin,xmin,ymax,xmax]}] -- rang
                   buoc VI TRI, cham diem bang IoU
        counts     {ten_lop: so_luong_toi_da}; None = khong rang buoc
        -> ([{video_id, n, pts_time, score, matched}], ghi_chu)
        """
        if not self.ok:
            return [], self.error
        min_score = self.min_score if min_score is None else max(min_score, self.min_score)

        n_kf = len(self.kf_video)
        keep = np.ones(n_kf, dtype=bool)
        if restrict_videos:
            codes = [self.vcode[v] for v in restrict_videos if v in self.vcode]
            if not codes:
                return [], "khong video nao trong danh sach gioi han co du lieu object"
            keep &= np.isin(self.kf_video, np.asarray(codes, dtype=np.int32))

        score = np.zeros(n_kf, dtype=np.float32)
        unknown = []
        matched_names = []
        n_asked = len(tags or ()) + len(boxes or ())
        n_applied = 0

        # --- rang buoc LOP (chi can co mat) --------------------------------
        for name in tags or ():
            c = self.resolve(name)
            if c is None:
                unknown.append(name)
                continue
            _, kf, sc, _ = self._dets_of(c)
            m = sc >= min_score
            best = np.zeros(n_kf, dtype=np.float32)
            np.maximum.at(best, kf[m], sc[m])
            keep &= best > 0
            score += best
            matched_names.append(self.entities[c])
            n_applied += 1

        # --- rang buoc VI TRI (lop + o dung cho) ---------------------------
        for b in boxes or ():
            name = (b or {}).get("type")
            q = (b or {}).get("box")
            c = self.resolve(name)
            if c is None or not q or len(q) != 4:
                if name:
                    unknown.append(name)
                continue
            _, kf, sc, bx = self._dets_of(c)
            m = sc >= min_score
            if not m.any():
                keep[:] = False
                break
            # Diem = do tin cay * IoU. Hop ve lech han -> IoU 0 -> keyframe bi
            # loai, dung y nghia "vat the o CHO NAY".
            v = sc[m] * iou(bx[m], [float(x) for x in q]).astype(np.float32)
            best = np.zeros(n_kf, dtype=np.float32)
            np.maximum.at(best, kf[m], v)
            keep &= best > 0
            score += 2.0 * best      # rang buoc vi tri kho hon -> tinh nang hon
            matched_names.append(f"{self.entities[c]}@box")
            n_applied += 1

        # --- rang buoc SO LUONG --------------------------------------------
        for name, want in (counts or {}).items():
            c = self.resolve(name)
            if c is None:
                unknown.append(name)
                continue
            _, kf, sc, _ = self._dets_of(c)
            m = sc >= min_score
            cnt = np.bincount(kf[m].astype(np.int64), minlength=n_kf)
            keep &= cnt <= int(want)

        if n_asked == 0:
            # Khong co rang buoc object nao -> khong loc gi, de tang tren quyet dinh.
            keep[:] = True
        elif n_applied == 0:
            # Da doi hoi lop nao do nhung KHONG lop nao co trong du lieu. Tra ve
            # ca corpus voi diem 0 la sai nghiem trong: nguoi dung tuong minh
            # dang loc, thuc te khong loc gi.
            return [], ("khong lop nao trong du lieu object khop yeu cau: "
                        + ", ".join(sorted(set(unknown))) +
                        ". Xem danh sach lop tai GET /data.")

        rows = np.flatnonzero(keep)
        if rows.size == 0:
            note = "khong keyframe nao thoa moi rang buoc"
            if unknown:
                note += f"; lop khong co trong du lieu: {sorted(set(unknown))}"
            return [], note

        if score.any():
            rows = rows[np.argsort(-score[rows], kind="stable")]
        rows = rows[:topk]

        out = [{
            "video_id": self.videos[int(self.kf_video[i])],
            "n": int(self.kf_n[i]),
            "pts_time": float(self.kf_pts[i]),
            "score": float(score[i]),
        } for i in rows]
        note = None
        if unknown:
            note = ("lop khong co trong du lieu object (da bo qua): "
                    + ", ".join(sorted(set(unknown))))
        return out, note

    def status(self):
        if not self.ok:
            return {"kha_dung": False, "error": self.error, "path": self.path}
        return {
            "kha_dung": True,
            "path": self.path,
            "n_video": len(self.videos),
            "n_keyframe": len(self.kf_video),
            "n_detection": int(len(self.det_ent)),
            "n_lop": len(self.entities),
            "nguong_diem": self.min_score,
        }


# ---------------------------------------------------------------------------
def parse_counts(text):
    """Chuoi tren o "maximum number of objects" -> {lop: so luong toi da}.

    Nhan "person 2, car 1" hoac "person:2" hoac moi dong mot cap. So tran trui
    khong kem ten lop thi bo qua -- khong doan la "toi da bao nhieu vat the",
    doan sai o day se loc mat ket qua dung ma nguoi dung khong hieu tai sao.
    """
    out = {}
    if not text:
        return out
    for chunk in str(text).replace("\n", ",").replace(";", ",").split(","):
        s = chunk.strip().replace(":", " ")
        if not s:
            continue
        parts = s.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        name, num = parts[0].strip(), parts[1].strip()
        if not name or not num.isdigit():
            continue
        out[name] = int(num)
    return out


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "build":
        def show(i, n, el):
            print(f"  {i}/{n} video, {el:.0f}s", flush=True)
        out = sys.argv[2] if len(sys.argv) > 2 else default_out()
        print(f"Dung index object tu {OBJECT_DIR} -> {out} ...", flush=True)
        print(build_index(out=out, progress=show))
    else:
        ix = ObjectIndex()
        print(f"tim trong: {INDEX_SEARCH_PATH}")
        print(ix.status())
        if ix.ok:
            print("\n20 lop hay gap nhat:")
            for e, c in ix.vocabulary(20):
                print(f"  {e:24s}{c}")
