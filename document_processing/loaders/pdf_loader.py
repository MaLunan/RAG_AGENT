# document_processing/loaders/pdf_loader.py
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from .base import BaseLoader
from core.config import settings


class PDFLoader(BaseLoader):
    """加载 PDF 文件，自动检测扫描版（无文本层）并抛出明确错误。"""

    @property
    def supported_extensions(self) -> set[str]:
        return {".pdf", ".PDF"}

    def load(self, file_path: Path) -> list[Document]:
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()

        total_text = " ".join(doc.page_content for doc in docs).strip()
        if len(total_text) < settings.min_text_length_for_scanned_detection:
            raise ValueError(
                f"PDF '{file_path.name}' appears to be a scanned document (no text layer). "
                "OCR support is not available in this version."
            )

        for doc in docs:
            doc.metadata.update({"source": file_path.name, "file_type": "pdf"})
        return docs
