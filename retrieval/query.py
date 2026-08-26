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
from dataclasses import dataclass, field
from typing import List, Optional

from retrieval.llm_client import get_client

DECOMPOSE_TIER = "pro"

_PROMPT = """Ban la bo phan ra truy van cho mot he thong tim kiem video tin tuc Viet Nam.

Truy van cua nguoi dung (tieng Viet): {q}

Hay tra ve DUNG mot object JSON, khong kem giai thich, theo dang:
{{
  "clauses_en": ["...", "..."],
  "query_vi": "...",
  "anchors": ["...", "..."],
  "kind": "generic_chain" | "anchored",
  "why": "mot cau ngan"
}}

Quy tac:
- "clauses_en": tach truy van thanh 1-5 menh de THI GIAC doc lap, THEO DUNG THU
  TU THOI GIAN trong truy van. Moi menh de la mot cau tieng Anh ngan mo ta thu
  NHIN THAY duoc trong khung hinh.
- "query_vi": giu nguyen tieng Viet co dau, giu lai TAT CA danh tu rieng va con
  so trong truy van goc -- day la phan co gia tri nhat cho tim kiem van ban.
- "anchors": nhung chi tiet HIEM, de nhan ra, gan nhu chac chan chi xuat hien o
  dung mot video: ten rieng, ten to chuc, chu tren bien hieu/banner, so hieu,
  dia danh cu the. KHONG ke cac hanh dong hay vat the pho bien.
- "kind": phan loai truy van
    "anchored"      = co it nhat mot anchor du hiem de mot minh no chot duoc
                      video. Vi du: bang hieu "London Zoo", doan "Nhan Nghia
                      Duong", bien so xe cu the.
    "generic_chain" = moi menh de deu la hanh dong/canh PHO BIEN, rieng le thi
                      hang nghin video co the khop; chi co THU TU va khoang cach
                      THOI GIAN giua chung moi phan biet duoc. Vi du: "nguoi
                      dung duoi nuoc roi den -> keo luoi ca luc binh minh ->
                      nhom nguoi den quay phim".
  Khi phan van, chon "generic_chain": dong hang theo thoi gian chi ton them mot
  chut tinh toan, con bo sot anchor thi mat hut ket qua.
- Khong bia them chi tiet khong co trong truy van goc.
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


def _translate(text):
    """Bac 2: chi dich, khong phan ra. Tra None neu ket qua khong dang tin."""
    try:
        from deep_translator import GoogleTranslator

        out = GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        return None
    out = (out or "").strip()
    return out if _looks_like_translation(text, out) else None


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

    n = max(count_events(text), len(clauses or []))
    if n >= 2:
        return ("generic_chain", [],
                f"khong co danh tu rieng nao; {n} su kien noi tiep nhau nen chi "
                f"thu tu thoi gian moi phan biet duoc")
    return "anchored", [], "chi mot su kien, khong co gi de dong hang"


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
        res = get_client().generate(_PROMPT.format(q=query_vi), tier=DECOMPOSE_TIER)
        if res.ok:
            data = _parse_json(res.text)
            if data:
                clauses = [str(c).strip() for c in (data.get("clauses_en") or [])
                           if str(c).strip()]
                vi = str(data.get("query_vi") or query_vi).strip()
                if clauses:
                    anchors = [str(a).strip()
                               for a in (data.get("anchors") or []) if str(a).strip()]
                    k = str(data.get("kind") or "").strip()
                    if k in ("anchored", "generic_chain"):
                        why = str(data.get("why") or "").strip() or None
                        src = "llm"
                    else:
                        # Model tra ve nhan la -> tu phan loai lai, khong doan bua.
                        k, anchors2, why = classify(query_vi, clauses)
                        anchors = anchors or anchors2
                        src = "heuristic"
                    if forced:
                        k, src = forced, "nguoi_dung"
                        why = "nguoi dung chon truc tiep"
                    return DecomposedQuery(
                        query_vi=vi or query_vi,
                        query_en="\n".join(clauses),
                        clauses_en=clauses,
                        anchors=anchors, kind=k, kind_why=why, kind_source=src,
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
