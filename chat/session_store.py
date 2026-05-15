# chat/session_store.py
from langchain_community.chat_message_histories import RedisChatMessageHistory
from core.config import settings


def get_session_history(session_id: str) -> RedisChatMessageHistory:
    """返回指定 session 的 Redis 对话历史（LangChain 标准接口）。"""
    return RedisChatMessageHistory(
        session_id=session_id,
        url=settings.redis_url,
        ttl=86400 * 7,  # 7天自动过期
    )
