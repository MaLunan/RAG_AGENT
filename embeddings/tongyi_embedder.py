# embeddings/tongyi_embedder.py
import logging
import time

from langchain_community.embeddings import DashScopeEmbeddings

from core.config import settings

logger = logging.getLogger(__name__)


class TongyiEmbedder:
    """封装 DashScope Tongyi Embedding，支持指数退避重试（最多 3 次）。"""

    MAX_RETRIES = 3

    def __init__(self) -> None:
        self._embedder = DashScopeEmbeddings(
            model=settings.tongyi_embedding_model,
            dashscope_api_key=settings.dashscope_api_key,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量向量化文本列表。"""
        return self._with_retry(lambda: self._embedder.embed_documents(texts))

    def embed_query(self, text: str) -> list[float]:
        """向量化单条查询文本。"""
        return self._with_retry(lambda: self._embedder.embed_query(text))

    def _with_retry(self, fn):
        """指数退避重试：1s → 2s → 4s，3 次失败后抛出最后一次异常。"""
        delay = 1
        last_exc: Exception | None = None
        for attempt in range(self.MAX_RETRIES):
            try:
                return fn()
            except Exception as exc:
                last_exc = exc
                if attempt < self.MAX_RETRIES - 1:
                    logger.warning(
                        "Embedding attempt %d/%d failed: %s. Retrying in %ds...",
                        attempt + 1,
                        self.MAX_RETRIES,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= 2
        raise last_exc
