# core/dependencies.py
from functools import lru_cache

from qdrant_client import QdrantClient

from core.config import settings


@lru_cache
def get_qdrant_client() -> QdrantClient:
    """返回 Qdrant client 单例（进程级缓存）。"""
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


@lru_cache
def get_store():
    """返回 QdrantStore 单例。"""
    from vectorstore.qdrant_store import QdrantStore
    return QdrantStore(client=get_qdrant_client())


@lru_cache
def get_embedder():
    """返回 TongyiEmbedder 单例。"""
    from embeddings.tongyi_embedder import TongyiEmbedder
    return TongyiEmbedder()


@lru_cache
def get_retriever():
    """返回 Retriever 单例。"""
    from retrieval.retriever import Retriever
    return Retriever(embedder=get_embedder(), store=get_store())
