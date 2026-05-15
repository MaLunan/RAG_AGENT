# vectorstore/qdrant_store.py
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from langchain_core.documents import Document

from core.config import settings


class QdrantStore:
    """封装 Qdrant 向量库操作：建 collection、MD5 查重、upsert、删除、检索。"""

    def __init__(self, client: QdrantClient) -> None:
        self._client = client

    def ensure_collection(self, collection: str) -> None:
        """若 collection 不存在则自动创建（余弦相似度）。"""
        existing_names = [c.name for c in self._client.get_collections().collections]
        if collection not in existing_names:
            self._client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=settings.vector_size, distance=Distance.COSINE),
            )

    def md5_exists(self, file_md5: str, collection: str) -> bool:
        """检查该 MD5 是否已在 collection 中存在（去重判断）。
        若 collection 不存在则返回 False。"""
        try:
            results, _ = self._client.scroll(
                collection_name=collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="file_md5", match=MatchValue(value=file_md5))]
                ),
                limit=1,
            )
            return len(results) > 0
        except Exception:
            # Collection 尚不存在，视为未入库
            return False

    def upsert(
        self,
        chunks: list[Document],
        vectors: list[list[float]],
        collection: str,
    ) -> None:
        """将 chunks 及对应向量写入 Qdrant。自动创建 collection。"""
        self.ensure_collection(collection)
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                # text 字段存储原始内容，其余 metadata 平铺
                payload={"text": chunk.page_content, **chunk.metadata},
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        self._client.upsert(collection_name=collection, points=points)

    def delete_by_md5(self, file_md5: str, collection: str) -> int:
        """删除 collection 中所有匹配该 MD5 的 point，返回删除数量。"""
        try:
            scroll_result, _ = self._client.scroll(
                collection_name=collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="file_md5", match=MatchValue(value=file_md5))]
                ),
                limit=10000,
            )
            point_ids = [p.id for p in scroll_result]
            if point_ids:
                self._client.delete(
                    collection_name=collection,
                    points_selector=point_ids,
                )
            return len(point_ids)
        except Exception:
            return 0

    def search(
        self,
        query_vector: list[float],
        collection: str,
        top_k: int,
        score_threshold: float,
    ) -> list[dict]:
        """语义检索，返回高于阈值的 top-k 结果。"""
        hits = self._client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return [
            {
                "content": hit.payload.get("text", ""),
                "source": hit.payload.get("source", ""),
                "score": hit.score,
                # metadata 中排除 text 字段（避免重复）
                "metadata": {k: v for k, v in hit.payload.items() if k != "text"},
            }
            for hit in hits
        ]

    def list_collections(self) -> list[str]:
        """列出所有 collection 名称。"""
        return [c.name for c in self._client.get_collections().collections]
