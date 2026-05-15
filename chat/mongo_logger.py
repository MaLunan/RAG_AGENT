# chat/mongo_logger.py
from datetime import datetime, timezone

import motor.motor_asyncio

from core.config import settings


class MongoLogger:
    """异步写入 MongoDB，记录每一轮对话（不阻塞响应）。"""

    def __init__(self):
        client = motor.motor_asyncio.AsyncIOMotorClient(settings.mongo_url)
        self._col = client[settings.mongo_db_name]["conversations"]

    async def log(
        self, session_id: str, query: str, answer: str, sources: list[dict]
    ) -> None:
        await self._col.insert_one({
            "session_id": session_id,
            "query": query,
            "answer": answer,
            "sources": [s.get("source", "") for s in sources],
            "created_at": datetime.now(timezone.utc),
        })

    async def read_logs(self, session_id: str, limit: int = 500) -> list[dict]:
        """读取指定 session 的永久对话日志。"""
        cursor = self._col.find(
            {"session_id": session_id},
            {"_id": 0},
            sort=[("created_at", 1)],
        )
        return await cursor.to_list(length=limit)
