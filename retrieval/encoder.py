"""Text encoder SigLIP -- phai khop dung model da dung de trich features.

Hai lo i im lang duoc chan o day:

1. Sai model. `features/*.npy` la 1152-d cua google/siglip-so400m-patch14-384.
   Encoder khac (Nomic, CLIPv2, ...) van cho ra vector chuan hoa va cosine van
   ra so dep, nhung ket qua vo nghia. `dim` duoc kiem tra o
   ArtifactStore.assert_encoder_matches().

2. Thieu padding="max_length". SiglipTextModel duoc huan luyen voi chuoi dem
   den du 64 token; dem kieu "longest" cho ra vector lech ma KHONG bao loi.
   Tham so nay duoc dat cung trong encode_texts(), khong cho ghi de.
"""

import numpy as np

from retrieval.config import SIGLIP_MODEL

# SigLIP so400m dung context 64 token.
MAX_LENGTH = 64


class SigLipTextEncoder:
    """Nap lazy: chi tai model khi thuc su encode lan dau."""

    def __init__(self, model_name=SIGLIP_MODEL, device=None):
        self.model_name = model_name
        self.device = device
        self.dim = None
        self._model = None
        self._tokenizer = None
        self.n_truncated = 0        # so chuoi da bi cat vi qua MAX_LENGTH
        self._cache = {}
        self.load_error = None

    # ------------------------------------------------------------------
    @property
    def available(self):
        try:
            self._ensure()
            return True
        except Exception:
            return False

    def _ensure(self):
        if self._model is not None:
            return
        if self.load_error is not None:
            raise RuntimeError(self.load_error)
        try:
            import torch
            from transformers import AutoTokenizer, SiglipTextModel

            self._torch = torch
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = SiglipTextModel.from_pretrained(self.model_name)
            self._model.eval().to(self.device)
            self.dim = int(self._model.config.hidden_size)
        except Exception as e:
            self.load_error = (
                f"Khong nap duoc SigLIP text encoder '{self.model_name}': "
                f"{type(e).__name__}: {e}. Kenh siglip se khong chay. "
                f"Tai model bang: huggingface-cli download {self.model_name}"
            )
            raise RuntimeError(self.load_error) from e

    # ------------------------------------------------------------------
    def n_tokens(self, text):
        """So token cua mot chuoi. Dung de KHONG bao gio gui qua MAX_LENGTH.

        SigLIP cat cut o 64 token va cat AM THAM. Do tren bo eval: truy van 4
        menh de gop lai dai 93 token, bi cat mat toan bo chi tiet phan biet
        ("cot moc dinh do", "vat mau xanh la") va tut tu hang 9 xuong khong tim
        thay. Khong co ham nay thi khong ai biet dieu do dang xay ra.
        """
        self._ensure()
        # Tokenizer canh bao "sequence length is longer than..." moi lan DO mot
        # chuoi dai. O day dang do co chu y de chia nho, khong phai dang ma hoa
        # -- de canh bao lot ra thi nguoi van hanh tuong dang co loi.
        import logging

        lg = logging.getLogger("transformers.tokenization_utils_base")
        lvl = lg.level
        lg.setLevel(logging.ERROR)
        try:
            return len(self._tokenizer(str(text))["input_ids"])
        finally:
            lg.setLevel(lvl)

    def would_truncate(self, text):
        return self.n_tokens(text) > MAX_LENGTH

    def pack_text(self, text, budget=MAX_LENGTH):
        """Chuoi dai -> danh sach manh, moi manh vua `budget` token.

        Cat o ranh gioi CAU chu khong cat giua chung. Cat giua chung la dieu
        tokenizer tu lam khi qua han, va no vut mat phan duoi ma khong bao gi.
        """
        text = str(text or "").strip()
        if not text:
            return []
        if self.n_tokens(text) <= budget:
            return [text]

        import re as _re

        sents = [x.strip() for x in _re.split(r"(?<=[.;!?])\s+|\n+", text) if x.strip()]
        if not sents:
            sents = [text]
        out, cur = [], []
        for sent in sents:
            trial = " ".join(cur + [sent])
            if cur and self.n_tokens(trial) > budget:
                out.append(" ".join(cur))
                cur = [sent]
            else:
                cur.append(sent)
        if cur:
            out.append(" ".join(cur))
        # Mot cau don le van co the qua dai -> danh phai cat theo tu.
        final = []
        for chunk in out:
            if self.n_tokens(chunk) <= budget:
                final.append(chunk)
                continue
            words, buf = chunk.split(), []
            for w in words:
                if buf and self.n_tokens(" ".join(buf + [w])) > budget:
                    final.append(" ".join(buf))
                    buf = [w]
                else:
                    buf.append(w)
            if buf:
                final.append(" ".join(buf))
        return final or [text]

    def encode_query(self, text, budget=MAX_LENGTH):
        """MOT vector cho mot truy van dai bat ky -- LOI VAO DUY NHAT nen dung.

        Goi thang encode_texts() voi chuoi dai hon 64 token thi tokenizer CAT
        CUT va khong bao gi ca. Do tren du lieu that: cau hoi Q/A tieng Viet dai
        194-536 token, tuc phan lon noi dung bi vut di.

        O day chuoi duoc chia thanh cac manh vua han muc roi lay trung binh
        vector. Vi tich vo huong la tuyen tinh, X . mean(Q) = mean(X . Q), nen
        ket qua bang dung voi "cham diem tung manh roi lay trung binh" -- tuc la
        phep HOI giua cac phan cua mo ta, khong phai phep tuyen.
        """
        chunks = self.pack_text(text, budget)
        if not chunks:
            return None
        V = self.encode_texts(chunks)
        if not len(V):
            return None
        v = V.mean(axis=0)
        n = float((v ** 2).sum() ** 0.5)
        return v / n if n > 1e-9 else v

    def encode_texts(self, texts):
        """-> (n, dim) float32 da L2-normalize, cung khong gian voi features."""
        if isinstance(texts, str):
            texts = [texts]
        texts = [t for t in texts if t and t.strip()]
        if not texts:
            return np.zeros((0, self.dim or 0), dtype=np.float32)

        self._ensure()
        torch = self._torch

        todo = [t for t in texts if t not in self._cache]
        if todo:
            batch = self._tokenizer(
                todo,
                padding="max_length",   # BAT BUOC -- xem docstring dau file
                max_length=MAX_LENGTH,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            # Dem so lan bi cat cut de /diagnostics noi duoc ra.
            for t in todo:
                if len(self._tokenizer(t)["input_ids"]) > MAX_LENGTH:
                    self.n_truncated += 1
            with torch.no_grad():
                out = self._model(**batch).pooler_output
                out = out / out.norm(dim=-1, keepdim=True)
            arr = out.detach().cpu().numpy().astype(np.float32)
            for t, v in zip(todo, arr):
                self._cache[t] = v

        return np.stack([self._cache[t] for t in texts]).astype(np.float32)

    def status(self):
        return {
            "model": self.model_name,
            "loaded": self._model is not None,
            "dim": self.dim,
            "device": self.device,
            "max_length": MAX_LENGTH,
            "n_bi_cat_cut": self.n_truncated,
            "error": self.load_error,
        }
