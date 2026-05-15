# document_processing/loaders/csv_loader.py
import csv
import chardet
import io
from pathlib import Path
from langchain_core.documents import Document
from .base import BaseLoader


class CSVLoader(BaseLoader):
    """加载 CSV 文件，自动检测编码。每行转为一个 Document。"""

    @property
    def supported_extensions(self) -> set[str]:
        return {".csv", ".CSV"}

    def load(self, file_path: Path) -> list[Document]:
        raw = file_path.read_bytes()
        detected = chardet.detect(raw)
        encoding = detected.get("encoding") or "utf-8"
        text = raw.decode(encoding, errors="replace")

        reader = csv.DictReader(io.StringIO(text))
        documents = []
        for i, row in enumerate(reader):
            # 将每行字段拼接为 "key: value" 格式
            content = "\n".join(f"{k}: {v}" for k, v in row.items() if v and v.strip())
            if content.strip():
                documents.append(
                    Document(
                        page_content=content,
                        metadata={"source": file_path.name, "file_type": "csv", "row": i},
                    )
                )
        return documents
