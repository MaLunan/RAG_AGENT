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
    """健康检查，可用于验证 Qdrant 连接。"""
    return {"status": "ok"}


@app.get("/")
async def root():
    return {"message": "RAGAgent API is running"}
