# document_processing/loaders/text_loader.py
import chardet
from pathlib import Path
from langchain_core.documents import Document
from .base import BaseLoader


class TextLoader(BaseLoader):
    """加载纯文本 (.txt) 文件，自动检测编码（支持 GBK / UTF-8 等）。"""

    @property
    def supported_extensions(self) -> set[str]:
        return {".txt", ".TXT"}

    def load(self, file_path: Path) -> list[Document]:
        raw = file_path.read_bytes()
        detected = chardet.detect(raw)
        encoding = detected.get("encoding") or "utf-8"
        text = raw.decode(encoding, errors="replace")
        return [
            Document(
                page_content=text,
                metadata={"source": file_path.name, "file_type": "txt"},
            )
        ]
