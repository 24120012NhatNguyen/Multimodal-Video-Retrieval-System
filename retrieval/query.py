"""Phan ra truy van kho -> hai chuoi cho hai loai kenh.

Tang hop nhat can DONG THOI:
  query_en  cac menh de TIENG ANH cho SigLIP (model huan luyen chu yeu tieng Anh,
            khong hieu danh tu rieng tieng Viet)
  query_vi  chuoi TIENG VIET cho BM25 (gia tri lon nhat nam o khop chinh xac
            danh tu rieng: "Cho Lon", "Nhan Nghia Duong")

Bac thang xuong cap (fallback bat buoc cua muc A1):

  1. LLM tier "pro"  -- phan ra thanh nhieu menh de tieng Anh + giu ban tieng Viet
  2. deep-translator -- chi dich, khong phan ra
  3. khong-LLM       -- query_vi giu nguyen, query_en = None

Buoc 3 CO Y bo trong query_en thay vi day chuoi tieng Viet sang SigLIP: lam the
cosine van ra so dep nhung ket qua vo nghia, dung mot trong nhung loi im lang ma
he thong nay phai tranh. Bo kenh siglip va chay bang BM25 la trung thuc hon.
"""

import json
import re
import threading
import os
from dataclasses import dataclass, field
from typing import List, Optional

from retrieval.llm_client import get_client

DECOMPOSE_TIER = "pro"

_PROMPT = """Bạn là bộ phân rã truy vấn cho hệ thống tìm kiếm video tin tức Việt Nam.

Truy vấn của người dùng: {q}

Trả về DUY NHẤT một object JSON, không giải thích, không markdown:
{{
  "reasoning": "một câu ngắn",
  "anchors": [{{"text": "...", "channel": "ocr" | "asr" | "any"}}],
  "kind": "anchored" | "generic_chain",
  "clauses_en": ["..."],
  "query_vi": "...",
  "question": null
}}

Quy tắc:

- "reasoning": trước khi quyết định, tự hỏi hai điều — truy vấn này mô tả MỘT
  cảnh hay một CHUỖI cảnh nối tiếp, và có chi tiết nào hiếm tới mức một mình nó
  chốt được video không.

- "anchors": chi tiết hiếm, gần như chắc chắn chỉ xuất hiện ở đúng một video —
  tên riêng, tên tổ chức, chữ trên biển hiệu hoặc banner, số hiệu, địa danh cụ
  thể. KHÔNG kể hành động hay vật thể phổ biến.
  "channel": "ocr" nếu là chữ hiện trên màn hình; "asr" nếu là thứ được nói ra;
  "any" nếu không chắc.

- "kind":
    "anchored"      = có ít nhất một anchor đủ hiếm để một mình nó chốt được video.
    "generic_chain" = mọi mệnh đề đều là cảnh phổ biến; chỉ THỨ TỰ và khoảng cách
                      thời gian giữa chúng mới phân biệt được.
  Khi phân vân, chọn "generic_chain": dóng hàng theo thời gian chỉ tốn thêm chút
  tính toán, còn bỏ sót anchor thì mất hút kết quả.

- "clauses_en": 1-5 mệnh đề THỊ GIÁC độc lập, theo đúng thứ tự thời gian.
  CHỈ tách khi truy vấn thật sự mô tả các cảnh NỐI TIẾP nhau. Một cảnh duy nhất
  thì đúng MỘT mệnh đề. Thà ít còn hơn tách thừa — tách thừa buộc hệ thống đi tìm
  một chuỗi không tồn tại.
  Viết như alt-text của ảnh: cụm mô tả, thì hiện tại, KHÔNG dùng "the video shows"
  hay "we see".
  Với khái niệm Việt Nam: dùng THUẬT NGỮ TIẾNG ANH đã có sẵn nếu có
  (múa lân -> "lion dance", nón lá -> "conical straw hat", xích lô -> "cyclo").
  Chỉ giữ nguyên từ Việt khi chính nó là tên thông dụng trong tiếng Anh, và khi
  đó viết KHÔNG DẤU như người Anh viết: "ao dai" chứ không phải "áo dài",
  "banh mi", "pho", "xe om".
  Toàn bộ clauses_en phải là tiếng Anh thuần, KHÔNG được lẫn chữ có dấu.

- "query_vi": giữ nguyên tiếng Việt có dấu, giữ TẤT CẢ danh từ riêng và con số
  trong truy vấn gốc.

- "question": nếu truy vấn có câu hỏi cần ĐỌC chi tiết từ khung hình (con số, chữ,
  màu sắc, đếm số lượng), tách riêng vào đây bằng tiếng Việt và KHÔNG đưa nó vào
  clauses_en. Không có thì để null.

- Không bịa thêm chi tiết không có trong truy vấn gốc.

Ví dụ 1
Truy vấn: "Phóng sự về đoàn Nhân Nghĩa Đường biểu diễn múa lân"
{{
  "reasoning": "Một cảnh duy nhất, nhưng tên đoàn là chi tiết rất hiếm.",
  "anchors": [{{"text": "Nhân Nghĩa Đường", "channel": "ocr"}}],
  "kind": "anchored",
  "clauses_en": ["a lion dance performance on a street"],
  "query_vi": "Phóng sự về đoàn Nhân Nghĩa Đường biểu diễn múa lân",
  "question": null
}}

Ví dụ 2
Truy vấn: "Hình ảnh một con cá được đặt lên cân, sau đó có cảnh một con cá khác cùng loại bị một người cầm đuôi. Con số hiển thị cuối cùng trên cân là bao nhiêu?"
{{
  "reasoning": "Hai cảnh nối tiếp, cả hai đều phổ biến; không có chi tiết hiếm nào.",
  "anchors": [],
  "kind": "generic_chain",
  "clauses_en": [
    "a fish placed on a weighing scale",
    "a person holding a fish by its tail"
  ],
  "query_vi": "Hình ảnh một con cá được đặt lên cân, sau đó có cảnh một con cá khác cùng loại bị một người cầm đuôi",
  "question": "Con số hiển thị cuối cùng trên cân là bao nhiêu?"
}}
"""


@dataclass
class DecomposedQuery:
    query_vi: str
    query_en: Optional[str] = None
    clauses_en: List[str] = field(default_factory=list)
    degraded: bool = False
    source: str = "llm"
    reason: Optional[str] = None
    model: Optional[str] = None
    # Chi tiet hiem du de mot minh no chot duoc video (ten rieng, chu tren bien
    # hieu, so hieu). Rong = khong co gi bam vao.
    anchors: List[str] = field(default_factory=list)
    # {anchor: "ocr"|"asr"|"any"} -- GOI Y cho nguoi dung nen bat kenh nao.
    # KHONG tu dong doi trong so: da do va viec do lam ket qua te di (xem
    # retrieval/engine.py, muc bo ho so "anchored").
    anchor_channels: dict = field(default_factory=dict)
    # Phan cau hoi can DOC chi tiet tu khung hinh (Q&A). Khong phai menh de thi
    # giac nen KHONG duoc dua vao clauses_en.
    question: Optional[str] = None
    # "anchored"      -> tim phang la du, uu tien kenh van ban
    # "generic_chain" -> tung menh de deu pho bien, chi thu tu thoi gian moi
    #                    phan biet duoc -> phai dong hang bang DP
    kind: str = "generic_chain"
    kind_why: Optional[str] = None
    # Ai quyet dinh `kind`: "nguoi_dung" > "llm" > "heuristic".
    # Nguoi dung da noi ro thi khong ai duoc ghi de.
    kind_source: str = "heuristic"

    @property
    def needs_dp(self):
        """Co can dong hang theo thoi gian khong.

        Chuoi hanh dong chung chung: tung manh deu pho bien, dong xuat hien theo
        DUNG THU TU moi hiem. Do la luc DP kiem com. Truy van co anchor thi mot
        menh de da chot duoc video, DP chi ton them thoi gian.
        """
        return self.kind == "generic_chain" and len(self.clauses_en) >= 2

    def as_dict(self):
        return {
            "query_vi": self.query_vi,
            "query_en": self.query_en,
            "clauses_en": self.clauses_en,
            "anchors": self.anchors,
            "anchor_channels": self.anchor_channels,
            "question": self.question,
            "kind": self.kind,
            "kind_why": self.kind_why,
            "kind_source": self.kind_source,
            "needs_dp": self.needs_dp,
            "degraded": self.degraded,
            "source": self.source,
            "reason": self.reason,
            "model": self.model,
        }


def _parse_json(text):
    """Model doi khi boc JSON trong ```json ... ``` hoac kem loi dan."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# deep-translator cao noi dung tu web nen khi dich vu loi no tra ve NGUYEN TRANG
# LOI duoi dang "ban dich" ma khong nem exception. Chuoi rac do di thang vao
# SigLIP thi cosine van ra so dep va ket qua vo nghia -- dung loi im lang ma he
# thong nay phai tranh, nen moi ban dich deu phai qua kiem chung.
_TRANSLATE_JUNK = (
    "that's an error", "that’s an error", "server error", "please try again later",
    "that's all we know", "that’s all we know", "<html", "<!doctype",
    "error 500", "error 502", "error 503", "429 too many",
)
_VI_DIACRITIC = re.compile(r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩị"
                           r"òóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]", re.I)


def _looks_like_translation(src, out):
    """Ban dich co ve that khong? Nghi ngo thi tra False de xuong bac duoi."""
    if not out or len(out) < 2:
        return False
    low = out.lower()
    if any(j in low for j in _TRANSLATE_JUNK):
        return False
    # Trang loi thuong dai gap nhieu lan truy van goc
    if len(out) > max(120, len(src) * 4):
        return False
    # Con nguyen dau tieng Viet nghia la chua dich duoc gi
    if _VI_DIACRITIC.search(out):
        return False
    return True


# --- Bo nho dem ban dich ---------------------------------------------------
# Google Translate KHONG tra ve cung mot cau cho cung mot dau vao. Do duoc:
# chay eval/run_eval.py --no-llm hai lan lien tiep tren cung mot cau hinh cho ra
# MRR 0.771 va 0.514 -- toan bo chenh lech den tu ban dich doi, khong tu he
# thong tim kiem. Khong co bo nho dem thi moi phep so sanh trong bo eval deu vo
# nghia, va giua buoi thi cung mot truy van go lai se ra ket qua khac.
#
# Dem tren dia, khoa la nguyen van cau tieng Viet.
TRANSLATION_CACHE = os.environ.get(
    "TRANSLATION_CACHE", "data/translation_cache.json")

_tcache = None
_tcache_lock = threading.Lock()


def _load_tcache():
    global _tcache
    if _tcache is None:
        try:
            with open(TRANSLATION_CACHE, encoding="utf-8") as f:
                _tcache = json.load(f)
        except Exception:
            _tcache = {}
    return _tcache


def _save_tcache():
    try:
        os.makedirs(os.path.dirname(TRANSLATION_CACHE) or ".", exist_ok=True)
        tmp = f"{TRANSLATION_CACHE}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_tcache, f, ensure_ascii=False, indent=1)
        os.replace(tmp, TRANSLATION_CACHE)
    except Exception:
        pass


# --- Bo nho dem phan ra bang LLM ------------------------------------------
# Cung ly do voi bo nho dem ban dich: LLM khong tat dinh. Do duoc, hai lan chay
# cung mot cau hinh tren cung bo eval cho Final Score 0.40 va 0.66 -- toan bo
# chenh lech den tu ban phan ra doi, khong tu he thong tim kiem. Khong dem thi
# khong hieu chinh duoc tham so nao, va giua buoi thi cung mot truy van go lai
# se ra ket qua khac.
DECOMPOSE_CACHE = os.environ.get("DECOMPOSE_CACHE", "data/decompose_cache.json")

_dcache = None
_dcache_lock = threading.Lock()


class _CachedResult:
    """Gia dang ket qua LLM de duong di ben duoi khong phai biet co dem."""

    ok = True
    reason = "cache"
    error = None
    model = "cache"
    latency_ms = 0

    def __init__(self, text):
        self.text = text


def _dcache_get(q):
    global _dcache
    if _dcache is None:
        try:
            with open(DECOMPOSE_CACHE, encoding="utf-8") as f:
                _dcache = json.load(f)
        except Exception:
            _dcache = {}
    return _dcache.get((q or "").strip())


def _dcache_put(q, text):
    key = (q or "").strip()
    if not key or not text:
        return
    with _dcache_lock:
        if _dcache is None:
            _dcache_get(key)
        _dcache[key] = text
        try:
            os.makedirs(os.path.dirname(DECOMPOSE_CACHE) or ".", exist_ok=True)
            tmp = f"{DECOMPOSE_CACHE}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(_dcache, f, ensure_ascii=False, indent=1)
            os.replace(tmp, DECOMPOSE_CACHE)
        except Exception:
            pass


def _translate(text, use_cache=True):
    """Bac 2: chi dich, khong phan ra. Tra None neu ket qua khong dang tin.

    Ket qua duoc dem tren dia: cung mot cau tieng Viet luon cho cung mot ban
    dich, ke ca sau khi khoi dong lai. Xem ghi chu o TRANSLATION_CACHE.
    """
    key = (text or "").strip()
    if not key:
        return None
    if use_cache:
        c = _load_tcache()
        if key in c:
            return c[key] or None

    try:
        from deep_translator import GoogleTranslator

        out = GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        return None
    out = (out or "").strip()
    out = out if _looks_like_translation(text, out) else None

    if use_cache and out:
        with _tcache_lock:
            _load_tcache()[key] = out
            _save_tcache()
    return out


# Tu viet hoa nhung KHONG phai danh tu rieng: hay dung o dau menh de.
_VI_STOP_CAP = {
    "một", "hai", "ba", "bốn", "năm", "sau", "tiếp", "trong", "khi", "có",
    "cảnh", "người", "các", "những", "và", "rồi", "đây", "đó", "thế", "phía",
    "tại", "trên", "dưới", "với", "cho", "của", "này", "kế", "đầu", "cuối",
    "ngoài", "bên", "giữa", "từ", "đến", "sang", "về", "ở", "là", "được",
}
_SENT_SPLIT = re.compile(r"[.;!?\n]+")
_QUOTED = re.compile("[\"'\u201c\u201d\u2018\u2019]([^\"'\u201c\u201d\u2018\u2019]{2,40})"
                     "[\"'\u201c\u201d\u2018\u2019]")


def _is_cap(tok):
    """Tu co viet hoa khong -- xet ky tu chu CAI DAU tien.

    Dung .isupper() chu KHONG dung lop ky tu regex: dai `A-Z\u00c0-\u1ef9` chua ca chu
    THUONG tieng Viet (d gach, a co dau...), nen `[A-Z\u00c0-\u1ef9]` nhan nham
    "dung", "den", "dang" la danh tu rieng.
    """
    for ch in tok:
        if ch.isalpha():
            return ch.isupper()
    return False


def heuristic_anchors(text):
    """Do anchor khi khong co LLM.

    Danh tu rieng tieng Viet duoc viet hoa, nhung dau menh de cung vay -- nen bo
    qua tu dau moi menh de va cac tu chuc nang hay dung o vi tri do. Kem chinh
    xac hon LLM, nhung du de khong mat kha nang dinh tuyen khi API chet.
    """
    found = [q.strip() for q in _QUOTED.findall(text or "")]

    for sent in _SENT_SPLIT.split(text or ""):
        toks = [t for t in re.split(r"\s+", sent.strip()) if t]
        run = []
        for pos, tok in enumerate(toks):
            word = tok.strip(",.:;()[]\"'")
            if pos > 0 and _is_cap(word) and word.lower() not in _VI_STOP_CAP:
                run.append(word)
            else:
                if run:
                    found.append(" ".join(run))
                run = []
        if run:
            found.append(" ".join(run))

    # so hieu / bien so / nam cu the
    found += re.findall(r"\b\d{2,}[A-Za-z0-9\-/.]*\b", text or "")

    seen, out = set(), []
    for a in found:
        a = a.strip(" ,.")
        if len(a) > 1 and a.lower() not in seen:
            seen.add(a.lower())
            out.append(a)
    return out


# Tu noi chi THU TU THOI GIAN trong tieng Viet. Co mat = truy van mo ta mot
# chuoi su kien, bat ke buoc dich co tach duoc menh de hay khong.
_SEQ_MARKERS = (
    "sau đó", "sau do", "tiếp theo", "tiep theo", "tiếp đến", "tiep den",
    "kế tiếp", "ke tiep", "rồi", "roi ", "cuối cùng", "cuoi cung",
    "đầu tiên", "dau tien", "trước đó", "truoc do", "sau cùng", "sau cung",
    "lúc đầu", "luc dau", "về sau", "ve sau",
)


def count_events(text):
    """Uoc luong so su kien tu chinh van ban tieng Viet.

    KHONG dua vao danh sach menh de da tach: buoc tach co the that bai (dich may
    hong, LLM chet) va khi do so menh de bang 0 -- do la loi cua buoc tach, chu
    khong phai bang chung rang truy van khong co chuoi su kien.
    """
    t = (text or "").lower()
    sents = [x for x in _SENT_SPLIT.split(t) if len(x.strip()) > 3]
    markers = sum(1 for m in _SEQ_MARKERS if m in t)
    return max(len(sents), markers + 1)


def classify(text, clauses=None):
    """(kind, anchors, ly_do) khi khong co LLM.

    Quyet dinh chinh dua vao ANCHOR: co chi tiet hiem thi mot menh de da chot
    duoc video, khong can dong hang. Khong co anchor thi xet xem truy van co mo
    ta mot chuoi su kien khong.
    """
    anchors = heuristic_anchors(text)
    if anchors:
        return "anchored", anchors, f"do duoc anchor: {', '.join(anchors[:3])}"

    # KHONG co anchor -> truy van THUAN THI GIAC, bat ke co may su kien.
    #
    # Ban truoc tra ve "anchored" cho truy van mot menh de khong anchor
    # ("nguoi dan ong va con cho") -- va "anchored" keo theo trong so NANG VAN
    # BAN, tuc la nguoc hoan toan. Loi do den tu viec gop hai quyet dinh doc lap:
    #
    #   co anchor khong   -> can nang VAN BAN hay nang THI GIAC
    #   co may su kien    -> co can dong hang thoi gian (DP) hay khong
    #
    # So su kien chi quyet dinh cai thu hai; `needs_dp` da lo phan do.
    n = max(count_events(text), len(clauses or []))
    if n >= 2:
        return ("generic_chain", [],
                f"khong co danh tu rieng nao; {n} su kien noi tiep nhau nen chi "
                f"thu tu thoi gian moi phan biet duoc")
    return ("generic_chain", [],
            "khong co danh tu rieng nao -- truy van thuan thi giac, "
            "kenh van ban khong co gi de bam vao")


def decompose(query_vi, query_en=None, use_llm=True, kind=None):
    """-> DecomposedQuery. Khong bao gio nem exception.

    Thu tu uu tien khi quyet dinh `kind`:

        1. `kind` nguoi dung truyen vao   -- cao nhat, khong ai ghi de
        2. nguoi dung tu tach menh de     -- tu tach ra nhieu dong tuc la da noi
                                             day la chuoi su kien co thu tu
        3. LLM phan loai
        4. bo do anchor (khi khong co LLM)

    Nguoi dung ngoi truoc man hinh nhin thay ket qua, con LLM thi khong -- nen
    khi hai ben khac y, nghe nguoi dung.
    """
    query_vi = (query_vi or "").strip()
    forced = kind if kind in ("anchored", "generic_chain") else None

    if query_en and query_en.strip():
        cl = [c for c in query_en.strip().split("\n") if c.strip()]
        auto_kind, anchors, why = classify(query_vi or query_en, cl)
        if forced:
            k, src = forced, "nguoi_dung"
            why = "nguoi dung chon truc tiep"
        elif len(cl) >= 2:
            # Tu tay tach thanh nhieu dong theo thu tu = da noi ro day la chuoi
            # su kien. Ton trong y do do thay vi de bo phan loai doan lai.
            k, src = "generic_chain", "nguoi_dung"
            why = f"nguoi dung tu tach {len(cl)} menh de theo thu tu"
        else:
            k, src, = auto_kind, "heuristic"
        return DecomposedQuery(
            query_vi=query_vi, query_en=query_en.strip(), clauses_en=cl,
            anchors=anchors, kind=k, kind_why=why, kind_source=src,
            source="nguoi_dung_nhap")

    if not query_vi:
        return DecomposedQuery(query_vi="", query_en=None, degraded=True,
                               source="rong", reason="truy van rong")

    # --- bac 1: LLM ------------------------------------------------------
    if use_llm:
        cached = _dcache_get(query_vi)
        res = (_CachedResult(cached) if cached is not None else
               get_client().generate(_PROMPT.format(q=query_vi), tier=DECOMPOSE_TIER))
        if res.ok:
            if cached is None:
                _dcache_put(query_vi, res.text)
            data = _parse_json(res.text)
            if data:
                clauses = [str(c).strip() for c in (data.get("clauses_en") or [])
                           if str(c).strip()]
                vi = str(data.get("query_vi") or query_vi).strip()
                if clauses:
                    # anchors gio la object {text, channel}. Nhan CA hai dang:
                    # ban cu tra ve mang chuoi phang.
                    anchors, anchor_channels = [], {}
                    for a in (data.get("anchors") or []):
                        if isinstance(a, dict):
                            t = str(a.get("text") or "").strip()
                            ch = str(a.get("channel") or "any").strip().lower()
                            if ch not in ("ocr", "asr", "any"):
                                ch = "any"
                        else:
                            t, ch = str(a).strip(), "any"
                        if t:
                            anchors.append(t)
                            anchor_channels[t] = ch
                    question = str(data.get("question") or "").strip() or None
                    k = str(data.get("kind") or "").strip()
                    if k in ("anchored", "generic_chain"):
                        # "reasoning" o dau (anh huong duoc quyet dinh), "why" la
                        # ten cu -- nhan ca hai de cache doi khong vo.
                        why = (str(data.get("reasoning") or data.get("why") or "")
                               .strip() or None)
                        src = "llm"
                    else:
                        # Model tra ve nhan la -> tu phan loai lai, khong doan bua.
                        k, anchors2, why = classify(query_vi, clauses)
                        anchors = anchors or anchors2
                        src = "heuristic"
                    if not anchors:
                        anchor_channels = {}
                    if forced:
                        k, src = forced, "nguoi_dung"
                        why = "nguoi dung chon truc tiep"
                    return DecomposedQuery(
                        query_vi=vi or query_vi,
                        query_en="\n".join(clauses),
                        clauses_en=clauses,
                        anchors=anchors, kind=k, kind_why=why, kind_source=src,
                        anchor_channels=anchor_channels, question=question,
                        source="llm", model=res.model)
            fail_reason = "model tra ve JSON khong doc duoc"
        else:
            fail_reason = res.reason or "loi_api"
    else:
        fail_reason = "use_llm=False"

    # --- bac 2: chi dich -------------------------------------------------
    en = _translate(query_vi)
    if en:
        # Dich may khong tach menh de, nhung van phan loai duoc bang bo do anchor.
        clauses = [c.strip() for c in re.split(r"[.;]+", en) if len(c.strip()) > 3]
        k, anchors, why = classify(query_vi, clauses or [en])
        src = "heuristic"
        if forced:
            k, src, why = forced, "nguoi_dung", "nguoi dung chon truc tiep"
        return DecomposedQuery(
            query_vi=query_vi, query_en="\n".join(clauses) if clauses else en,
            clauses_en=clauses or [en],
            anchors=anchors, kind=k, kind_why=why, kind_source=src,
            degraded=True, source="dich_may", reason=fail_reason)

    # --- bac 3: khong-LLM ------------------------------------------------
    # query_en de TRONG co y -- xem docstring dau file.
    k, anchors, why = classify(query_vi, [])
    src = "heuristic"
    if forced:
        k, src, why = forced, "nguoi_dung", "nguoi dung chon truc tiep"
    return DecomposedQuery(
        query_vi=query_vi, query_en=None, clauses_en=[],
        anchors=anchors, kind=k, kind_why=why, kind_source=src,
        degraded=True, source="khong_llm",
        reason=f"{fail_reason}; dich may cung that bai -> bo kenh siglip")
