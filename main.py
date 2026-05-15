# main.py
import logging

from fastapi import FastAPI

from api.documents import router as documents_router

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="RAGAgent",
    description="RAG 知识库 API：文档上传、语义检索",
    version="0.1.0",
)

# 内存 task 状态存储（进程重启后清空，首版可接受）
app.state.tasks: dict = {}

app.include_router(documents_router)


@app.get("/health")
async def health():
    """健康检查，验证 Qdrant 连接可用。"""
    from core.dependencies import get_qdrant_client
    try:
        client = get_qdrant_client()
        client.get_collections()
        return {"status": "ok", "qdrant": "connected"}
    except Exception as exc:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content={"status": "error", "qdrant": str(exc)}
        )


@app.get("/")
async def root():
    return {"message": "RAGAgent API is running"}
