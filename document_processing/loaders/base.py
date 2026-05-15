# document_processing/loaders/base.py
from abc import ABC, abstractmethod
from pathlib import Path
from langchain_core.documents import Document


class BaseLoader(ABC):
    """所有格式加载器的抽象基类。每个子类负责一种文件格式。"""

    @abstractmethod
    def load(self, file_path: Path) -> list[Document]:
        """加载文件，返回附带元数据的 Document 列表。"""
        ...

    @property
    @abstractmethod
    def supported_extensions(self) -> set[str]:
        """该加载器支持的文件后缀（含点，如 {'.txt', '.TXT'}）。"""
        ...
