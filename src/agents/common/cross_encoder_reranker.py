"""Cross-encoder reranker dùng chung cho retrieval.

Model được lazy-load và cache theo process vì chatbot chỉ cần một instance.
Failure không làm sập RAG: caller sẽ giữ thứ tự RRF original-first, tuyệt đối
không quay lại lexical reranker vốn là nguyên nhân gây topic drift.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from typing import Sequence

logger = logging.getLogger(__name__)

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
DEFAULT_CROSS_ENCODER_REVISION = "1427fd652930e4ba29e8149678df786c240d8825"
_shared_reranker = None
_shared_reranker_lock = threading.Lock()


class CrossEncoderReranker:
    """Chấm trực tiếp từng cặp (query gốc, passage) trong miền [0, 1]."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        revision: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        max_length: int | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.model_name = model_name or os.getenv(
            "CROSS_ENCODER_MODEL", DEFAULT_CROSS_ENCODER_MODEL
        )
        self.revision = revision or os.getenv(
            "CROSS_ENCODER_REVISION", DEFAULT_CROSS_ENCODER_REVISION
        )
        self.device = device or os.getenv("CROSS_ENCODER_DEVICE") or "cpu"
        self.batch_size = batch_size or int(os.getenv("CROSS_ENCODER_BATCH_SIZE", "8"))
        self.max_length = max_length or int(os.getenv("CROSS_ENCODER_MAX_LENGTH", "384"))
        if enabled is None:
            enabled = os.getenv("CROSS_ENCODER_ENABLED", "true").casefold() not in {
                "0",
                "false",
                "no",
            }
        self.enabled = enabled
        self._model = None
        self._load_failed = False
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self.enabled and not self._load_failed

    def _load(self):
        if self._model is not None or self._load_failed or not self.enabled:
            return self._model
        with self._lock:
            if self._model is not None or self._load_failed:
                return self._model
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(
                    self.model_name,
                    revision=self.revision,
                    device=self.device,
                    max_length=self.max_length,
                    trust_remote_code=False,
                )
                logger.info(
                    "Cross encoder đã sẵn sàng: %s @ %s (%s)",
                    self.model_name,
                    self.revision[:8],
                    self.device,
                )
            except Exception:
                self._load_failed = True
                logger.warning(
                    "Không tải được cross encoder %s; giữ thứ tự RRF original-first.",
                    self.model_name,
                    exc_info=True,
                )
        return self._model

    def score(self, query: str, passages: Sequence[str]) -> list[float] | None:
        """Trả probability-like score; ``None`` nghĩa là dùng fallback RRF."""

        if not passages:
            return []
        model = self._load()
        if model is None:
            return None
        try:
            raw_scores = model.predict(
                [(query, passage) for passage in passages],
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            flattened = raw_scores.reshape(-1).tolist()
            return [
                1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, float(score)))))
                for score in flattened
            ]
        except Exception:
            logger.warning(
                "Cross encoder chấm điểm thất bại; giữ thứ tự RRF original-first.",
                exc_info=True,
            )
            return None


def get_cross_encoder_reranker() -> CrossEncoderReranker:
    """Một model dùng chung cho mọi pipeline embedding trong cùng process."""

    global _shared_reranker
    if _shared_reranker is None:
        with _shared_reranker_lock:
            if _shared_reranker is None:
                _shared_reranker = CrossEncoderReranker()
    return _shared_reranker
