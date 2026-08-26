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
            "error": self.load_error,
        }
