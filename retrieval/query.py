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
  "query_vi": "..."
}}

Quy tac:
- "clauses_en": tach truy van thanh 1-4 menh de THI GIAC doc lap, moi menh de la
  mot cau tieng Anh ngan mo ta thu NHIN THAY duoc trong khung hinh. Bo cac y
  khong the nhin thay (ten rieng, ngay thang, con so thong ke).
- "query_vi": giu nguyen tieng Viet co dau, giu lai TAT CA danh tu rieng va con
  so trong truy van goc -- day la phan co gia tri nhat cho tim kiem van ban.
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

    def as_dict(self):
        return {
            "query_vi": self.query_vi,
            "query_en": self.query_en,
            "clauses_en": self.clauses_en,
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


def decompose(query_vi, query_en=None, use_llm=True):
    """-> DecomposedQuery. Khong bao gio nem exception.

    `query_en` do nguoi dung tu nhap luon duoc ton trong: da co thi khong goi
    LLM va khong dich lai.
    """
    query_vi = (query_vi or "").strip()

    if query_en and query_en.strip():
        return DecomposedQuery(
            query_vi=query_vi, query_en=query_en.strip(),
            clauses_en=[c for c in query_en.strip().split("\n") if c.strip()],
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
                    return DecomposedQuery(
                        query_vi=vi or query_vi,
                        query_en="\n".join(clauses),
                        clauses_en=clauses,
                        source="llm", model=res.model)
            fail_reason = "model tra ve JSON khong doc duoc"
        else:
            fail_reason = res.reason or "loi_api"
    else:
        fail_reason = "use_llm=False"

    # --- bac 2: chi dich -------------------------------------------------
    en = _translate(query_vi)
    if en:
        return DecomposedQuery(
            query_vi=query_vi, query_en=en, clauses_en=[en],
            degraded=True, source="dich_may", reason=fail_reason)

    # --- bac 3: khong-LLM ------------------------------------------------
    # query_en de TRONG co y -- xem docstring dau file.
    return DecomposedQuery(
        query_vi=query_vi, query_en=None, clauses_en=[],
        degraded=True, source="khong_llm",
        reason=f"{fail_reason}; dich may cung that bai -> bo kenh siglip")
