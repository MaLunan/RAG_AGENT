# document_processing/loaders/markdown_loader.py
from pathlib import Path
from langchain_core.documents import Document
from .base import BaseLoader


class MarkdownLoader(BaseLoader):
    """加载 Markdown 文件，保留原始文本内容。"""

    @property
    def supported_extensions(self) -> set[str]:
        return {".md", ".MD", ".markdown", ".MARKDOWN"}

    def load(self, file_path: Path) -> list[Document]:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        return [
            Document(
                page_content=text,
                metadata={"source": file_path.name, "file_type": "md"},
            )
        ]
