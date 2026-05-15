# document_processing/pipeline.py
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document

from .cleaners.text_cleaner import TextCleaner
from .chunkers.smart_chunker import SmartChunker
from .loaders.base import BaseLoader
from .loaders.csv_loader import CSVLoader
from .loaders.markdown_loader import MarkdownLoader
from .loaders.pdf_loader import PDFLoader
from .loaders.text_loader import TextLoader
from .loaders.word_loader import WordLoader
from core.config import settings


def _build_loader_registry() -> dict[str, BaseLoader]:
    """将每个 loader 注册到其支持的文件后缀上。"""
    registry: dict[str, BaseLoader] = {}
    for loader in [PDFLoader(), WordLoader(), MarkdownLoader(), CSVLoader(), TextLoader()]:
        for ext in loader.supported_extensions:
            registry[ext] = loader
    return registry


_LOADERS: dict[str, BaseLoader] = _build_loader_registry()


def compute_md5(file_bytes: bytes) -> str:
    """计算文件字节流的 MD5 哈希（32 位小写十六进制）。"""
    return hashlib.md5(file_bytes).hexdigest()


def get_loader(extension: str) -> BaseLoader:
    """根据文件后缀返回对应 Loader，不支持的格式抛出 ValueError。"""
    loader = _LOADERS.get(extension)
    if not loader:
        raise ValueError(f"Unsupported file extension: {extension}")
    return loader


class DocumentPipeline:
    """串联 loader → cleaner → chunker，将文件转换为带元数据的 chunk 列表。"""

    def __init__(
        self,
        cleaner: TextCleaner | None = None,
        chunker: SmartChunker | None = None,
    ):
        self._cleaner = cleaner or TextCleaner()
        self._chunker = chunker or SmartChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def process(self, file_path: Path, file_md5: str, collection: str) -> list[Document]:
        """
        完整处理流程：load → clean → chunk → 注入元数据。

        Args:
            file_path: 待处理文件路径（临时文件）。
            file_md5:  文件的 MD5 哈希，用于去重和溯源。
            collection: 目标 Qdrant collection 名称。

        Returns:
            List[Document]，每个 chunk 附带完整元数据。

        Raises:
            ValueError: 文件无可提取文本（空文档或扫描版 PDF）。
        """
        loader = get_loader(file_path.suffix)
        docs = loader.load(file_path)

        # 早失败：空文档不进入后续流程
        total_text = " ".join(d.page_content for d in docs).strip()
        if not total_text:
            raise ValueError(f"No extractable text found in '{file_path.name}'")

        cleaned = self._cleaner.clean(docs)
        chunks = self._chunker.chunk(cleaned)

        now = datetime.now(timezone.utc).isoformat()
        for chunk in chunks:
            chunk.metadata.update(
                {
                    "file_md5": file_md5,
                    "collection": collection,
                    "created_at": now,
                }
            )

        return chunks
