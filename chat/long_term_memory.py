# chat/long_term_memory.py
import uuid

from langchain_core.documents import Document
from qdrant_client.models import PointStruct

from core.config import settings


class LongTermMemory:
    """Qdrant 长期向量记忆：存储历史 Q&A，支持跨会话语义关联。"""

    COLLECTION = "_longmem"

    def __init__(self, store, embedder) -> None:
        self._store = store
        self._embedder = embedder

    def save(self, session_id: str, query: str, answer: str) -> None:
        """将一轮 Q&A 向量化后存入长期记忆。"""
        text = f"Q: {query}\nA: {answer}"
        vector = self._embedder.embed_query(text)
        self._store.ensure_collection(self.COLLECTION)
        doc = Document(
            page_content=text,
            metadata={"session_id": session_id, "source": self.COLLECTION},
        )
        self._store.upsert(chunks=[doc], vectors=[vector], collection=self.COLLECTION)

    def search(self, query: str, session_id: str, top_k: int = 3) -> list[str]:
        """检索与当前问题最相关的历史 Q&A（限定同 session）。"""
        vector = self._embedder.embed_query(query)
        results = self._store.search(
            query_vector=vector,
            collection=self.COLLECTION,
            top_k=top_k * 3,  # 多取几条，过滤后仍有 top_k 条
            score_threshold=0.6,
        )
        return [
            r["content"]
            for r in results
            if r["metadata"].get("session_id") == session_id
        ][:top_k]
