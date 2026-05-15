# api/chat.py
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from chat.session_store import get_session_history
from core.dependencies import get_chat_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    session_id: str
    query: str
    collection: str
    top_k: int = 5
    score_threshold: float = 0.3


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[dict]


@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest, engine=Depends(get_chat_engine)):
    """多轮问答接口。"""
    return engine.chat(
        session_id=req.session_id,
        query=req.query,
        collection=req.collection,
        top_k=req.top_k,
        score_threshold=req.score_threshold,
    )


@router.get("/{session_id}/history")
def get_history(session_id: str):
    """查看指定 session 的对话历史。"""
    history = get_session_history(session_id)
    messages = []
    for msg in history.messages:
        messages.append({
            "role": "human" if msg.type == "human" else "ai",
            "content": msg.content,
        })
    return {"session_id": session_id, "messages": messages}


@router.delete("/{session_id}")
def clear_session(session_id: str):
    """清空指定 session 的对话历史。"""
    history = get_session_history(session_id)
    history.clear()
    return {"session_id": session_id, "status": "cleared"}


@router.get("/{session_id}/logs")
async def get_logs(session_id: str, engine=Depends(get_chat_engine)):
    """从 MongoDB 获取永久对话日志。"""
    try:
        logs = await engine._mongo_logger.read_logs(session_id)
        # Convert datetime to ISO string for JSON serialization
        for log in logs:
            if "created_at" in log:
                log["created_at"] = log["created_at"].isoformat()
        return {"session_id": session_id, "logs": logs}
    except Exception as e:
        logger.warning(f"Failed to fetch logs: {e}")
        raise HTTPException(status_code=503, detail="Log storage unavailable")
