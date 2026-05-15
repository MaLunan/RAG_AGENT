# retrieval/retriever.py
from embeddings.tongyi_embedder import TongyiEmbedder
from vectorstore.qdrant_store import QdrantStore
from core.config import settings


class Retriever:
    """语义检索：将 query 向量化后在 Qdrant 中检索最相关结果。"""

    def __init__(self, embedder: TongyiEmbedder, store: QdrantStore) -> None:
        self._embedder = embedder
        self._store = store

    def search(
        self,
        query: str,
        collection: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[dict]:
        """
        执行语义检索。

        Args:
            query: 自然语言查询字符串。
            collection: Qdrant collection 名称。
            top_k: 返回结果数，默认读取 settings.top_k。
            score_threshold: 相似度阈值，默认读取 settings.score_threshold。

        Returns:
            List of dicts with keys: content, source, score, metadata.
        """
        effective_top_k = top_k if top_k is not None else settings.top_k
        effective_threshold = (
            score_threshold if score_threshold is not None else settings.score_threshold
        )

        query_vector = self._embedder.embed_query(query)
        return self._store.search(
            query_vector=query_vector,
            collection=collection,
            top_k=effective_top_k,
            score_threshold=effective_threshold,
        )
