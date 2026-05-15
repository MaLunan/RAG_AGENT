# api/documents.py
import logging
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from core.config import settings
from core.dependencies import get_embedder, get_retriever, get_store
from document_processing.pipeline import DocumentPipeline, compute_md5
from embeddings.tongyi_embedder import TongyiEmbedder
from retrieval.retriever import Retriever
from vectorstore.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

SUPPORTED_EXTENSIONS = {
    ".pdf", ".PDF",
    ".docx", ".DOCX",
    ".md", ".MD", ".markdown", ".MARKDOWN",
    ".csv", ".CSV",
    ".txt", ".TXT",
}

MAX_UPLOAD_BYTES = settings.max_upload_size_mb * 1024 * 1024


class SearchRequest(BaseModel):
    query: str
    collection: str = "default"
    top_k: int = Field(default_factory=lambda: settings.top_k)
    score_threshold: float = Field(default_factory=lambda: settings.score_threshold)


def _process_document(
    tmp_path: str,
    file_name: str,
    file_md5: str,
    collection: str,
    task_id: str,
    tasks: dict,
    store: QdrantStore,
    embedder: TongyiEmbedder,
    in_flight_md5s: set,
) -> None:
    """后台异步处理：pipeline → embed → upsert。失败时回滚并标记状态。"""
    try:
        pipeline = DocumentPipeline()
        chunks = pipeline.process(Path(tmp_path), file_md5, collection)

        texts = [c.page_content for c in chunks]
        vectors = embedder.embed_documents(texts)

        # 二次 MD5 检查：若同名文件已存在则先删旧版本（文件更新场景）
        if store.md5_exists(file_md5, collection):
            store.delete_by_md5(file_md5, collection)

        store.upsert(chunks, vectors, collection)
        tasks[task_id]["status"] = "done"
    except Exception as exc:
        logger.error("Processing failed for %s: %s", file_name, exc)
        # 回滚：删除可能已写入的部分 chunks
        try:
            store.delete_by_md5(file_md5, collection)
        except Exception:
            pass
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(exc)
    finally:
        # 无论成功或失败，都要从 in_flight 集合中移除，允许后续上传
        in_flight_md5s.discard(file_md5)
        Path(tmp_path).unlink(missing_ok=True)


@router.post("/upload")
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    collection: str = Form(default="default"),
    store: QdrantStore = Depends(get_store),
    embedder: TongyiEmbedder = Depends(get_embedder),
):
    """
    上传文档并异步处理入库。

    - 校验文件大小（默认 50MB 上限）
    - 校验文件格式（pdf/docx/md/csv/txt）
    - MD5 去重（已存在返回 409）
    - 后台 Pipeline 处理，立即返回 task_id
    """
    content = await file.read()

    # 1. 文件大小校验
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.max_upload_size_mb}MB.",
        )

    # 2. 文件格式校验
    suffix = Path(file.filename or "").suffix
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{suffix}'. Supported: pdf, docx, md, csv, txt.",
        )

    # 3. MD5 去重（先查 in_flight 再查 Qdrant，避免并发上传竞态）
    file_md5 = compute_md5(content)
    in_flight_md5s: set = request.app.state.in_flight_md5s
    if file_md5 in in_flight_md5s or store.md5_exists(file_md5, collection):
        raise HTTPException(
            status_code=409,
            detail={"status": "duplicate", "file_md5": file_md5},
        )
    # 标记为处理中，防止同一文件在后台任务完成前被重复提交
    in_flight_md5s.add(file_md5)

    # 4. 写入临时文件（BackgroundTask 读取后自动删除）
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(content)
    tmp.close()

    # 5. 注册后台任务
    task_id = str(uuid.uuid4())
    request.app.state.tasks[task_id] = {"status": "processing", "error": None}

    background_tasks.add_task(
        _process_document,
        tmp_path=tmp.name,
        file_name=file.filename or "unknown",
        file_md5=file_md5,
        collection=collection,
        task_id=task_id,
        tasks=request.app.state.tasks,
        store=store,
        embedder=embedder,
        in_flight_md5s=in_flight_md5s,
    )

    return {"task_id": task_id, "status": "processing"}


@router.get("/status/{task_id}")
async def get_task_status(task_id: str, request: Request):
    """查询文档处理任务状态。"""
    task = request.app.state.tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    return {"task_id": task_id, **task}


@router.post("/search")
async def search_documents(
    body: SearchRequest,
    retriever: Retriever = Depends(get_retriever),
):
    """语义检索，返回相似度高于阈值的 top-k 文档片段。"""
    results = retriever.search(
        query=body.query,
        collection=body.collection,
        top_k=body.top_k,
        score_threshold=body.score_threshold,
    )
    return {"results": results}


@router.delete("/{file_md5}")
async def delete_document(
    file_md5: str,
    collection: str = "default",
    store: QdrantStore = Depends(get_store),
):
    """通过 MD5 删除 collection 中该文档的所有 chunks。"""
    deleted = store.delete_by_md5(file_md5, collection)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"deleted_chunks": deleted}


@router.get("/collections")
async def list_collections(store: QdrantStore = Depends(get_store)):
    """列出所有 Qdrant collection 名称。"""
    return {"collections": store.list_collections()}
