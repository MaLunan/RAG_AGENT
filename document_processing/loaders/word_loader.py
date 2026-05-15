# document_processing/loaders/word_loader.py
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader
from .base import BaseLoader


class WordLoader(BaseLoader):
    """加载 Word (.docx) 文件。"""

    @property
    def supported_extensions(self) -> set[str]:
        return {".docx", ".DOCX"}

    def load(self, file_path: Path) -> list[Document]:
        loader = Docx2txtLoader(str(file_path))
        docs = loader.load()
        for doc in docs:
            doc.metadata.update({"source": file_path.name, "file_type": "docx"})
        return docs
