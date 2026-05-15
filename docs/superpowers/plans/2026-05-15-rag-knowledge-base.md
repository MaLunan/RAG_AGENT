# RAG 知识库 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建基于 LangChain + Qdrant + Tongyi 的 RAG 知识库文档处理 Pipeline，通过 FastAPI REST API 提供文档上传（含 MD5 去重）、异步处理与语义检索能力。

**Architecture:** 分层 Pipeline 模块架构，loaders → cleaner → chunker → embedder → qdrant_store，每层单一职责、独立可测；API 层用 FastAPI BackgroundTasks 处理耗时操作，task 状态存 app.state 内存字典。

**Tech Stack:** Python 3.12+, FastAPI, LangChain, langchain-community, qdrant-client, tongyi(DashScope), pypdf, python-docx, chardet, pytest, httpx

---

## 文件清单

| 操作 | 路径 | 职责 |
|------|------|------|
| Modify | `pyproject.toml` | 添加所有依赖 |
| Modify | `main.py` | 注册路由、初始化 app.state.tasks、health 端点 |
| Create | `core/__init__.py` | 空 |
| Create | `core/config.py` | Pydantic Settings，读取 .env |
| Create | `core/dependencies.py` | FastAPI 依赖注入（Qdrant client、embedder、store、retriever 单例） |
| Create | `api/__init__.py` | 空 |
| Create | `api/documents.py` | 所有文档相关路由（upload/status/search/delete/collections） |
| Create | `document_processing/__init__.py` | 空 |
| Create | `document_processing/loaders/__init__.py` | 空 |
| Create | `document_processing/loaders/base.py` | BaseLoader 抽象类 |
| Create | `document_processing/loaders/text_loader.py` | TXT 加载（chardet 编码） |
| Create | `document_processing/loaders/csv_loader.py` | CSV 加载（chardet 编码，逐行转 Document） |
| Create | `document_processing/loaders/markdown_loader.py` | Markdown 加载 |
| Create | `document_processing/loaders/pdf_loader.py` | PDF 加载（扫描件检测） |
| Create | `document_processing/loaders/word_loader.py` | Word(.docx) 加载 |
| Create | `document_processing/cleaners/__init__.py` | 空 |
| Create | `document_processing/cleaners/text_cleaner.py` | 去噪、空白规范化、空行过滤 |
| Create | `document_processing/chunkers/__init__.py` | 空 |
| Create | `document_processing/chunkers/smart_chunker.py` | RecursiveCharacterTextSplitter + chunk_index 注入 |
| Create | `document_processing/pipeline.py` | 串联 loader→cleaner→chunker，MD5 计算，格式分发 |
| Create | `embeddings/__init__.py` | 空 |
| Create | `embeddings/tongyi_embedder.py` | DashScope 封装，指数退避重试 |
| Create | `vectorstore/__init__.py` | 空 |
| Create | `vectorstore/qdrant_store.py` | Qdrant 增删查（upsert/search/delete_by_md5/md5_exists） |
| Create | `retrieval/__init__.py` | 空 |
| Create | `retrieval/retriever.py` | embed_query → qdrant search → 返回结果列表 |
| Create | `tests/__init__.py` | 空 |
| Create | `tests/test_text_cleaner.py` | TextCleaner 单元测试 |
| Create | `tests/test_smart_chunker.py` | SmartChunker 单元测试 |
| Create | `tests/test_loaders.py` | 各 Loader 单元测试（临时文件） |
| Create | `tests/test_pipeline.py` | DocumentPipeline 单元测试（mock loader） |
| Create | `tests/test_qdrant_store.py` | QdrantStore 单元测试（mock qdrant-client） |
| Create | `tests/test_retriever.py` | Retriever 单元测试（mock embedder + store） |
| Create | `tests/test_documents_api.py` | API 集成测试（TestClient + mock 依赖） |
| Create | `.env.example` | 环境变量模板 |

---

## Task 1: 项目依赖与目录骨架

**Files:**
- Modify: `pyproject.toml`
- Create: 所有 `__init__.py` 文件（见文件清单）
- Create: `.env.example`

- [ ] **Step 1: 更新 pyproject.toml**

```toml
[project]
name = "ragagent"
version = "0.1.0"
description = "RAG knowledge base with LangChain + Qdrant + Tongyi"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn>=0.30.0",
    "pydantic-settings>=2.3.0",
    "langchain>=0.3.0",
    "langchain-community>=0.3.0",
    "langchain-text-splitters>=0.3.0",
    "qdrant-client>=1.9.0",
    "pypdf>=4.3.0",
    "pymupdf>=1.24.0",
    "python-docx>=1.1.0",
    "docx2txt>=0.8",
    "chardet>=5.2.0",
    "dashscope>=1.20.0",
    "python-multipart>=0.0.9",
    "unstructured>=0.14.0",
]

[dependency-groups]
dev = [
    "pytest>=8.2.0",
    "pytest-asyncio>=0.23.0",
    "httpx>=0.27.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

- [ ] **Step 2: 安装依赖**

```bash
uv sync
```

Expected: 依赖解析并安装完成，无报错。

- [ ] **Step 3: 创建目录骨架与 __init__.py**

```bash
mkdir -p core api document_processing/loaders document_processing/cleaners document_processing/chunkers embeddings vectorstore retrieval tests
touch core/__init__.py api/__init__.py \
  document_processing/__init__.py \
  document_processing/loaders/__init__.py \
  document_processing/cleaners/__init__.py \
  document_processing/chunkers/__init__.py \
  embeddings/__init__.py vectorstore/__init__.py \
  retrieval/__init__.py tests/__init__.py
```

- [ ] **Step 4: 创建 .env.example**

```bash
# .env.example
DASHSCOPE_API_KEY=your_dashscope_api_key_here
TONGYI_EMBEDDING_MODEL=text-embedding-v3
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
DEFAULT_COLLECTION=default
MAX_UPLOAD_SIZE_MB=50
SCORE_THRESHOLD=0.7
TOP_K=5
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

- [ ] **Step 5: 复制 .env.example 为 .env 并填写真实 key**

```bash
cp .env.example .env
# 编辑 .env，填入真实的 DASHSCOPE_API_KEY
```

- [ ] **Step 6: Commit**

```bash
git init
git add pyproject.toml .env.example core/ api/ document_processing/ embeddings/ vectorstore/ retrieval/ tests/
git commit -m "chore: scaffold project structure and add dependencies"
```

---

## Task 2: Core 配置与依赖注入

**Files:**
- Create: `core/config.py`
- Create: `core/dependencies.py`

- [ ] **Step 1: 编写 core/config.py**

```python
# core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """统一配置，所有值均可通过环境变量或 .env 文件覆盖。"""

    # Tongyi / DashScope
    dashscope_api_key: str = ""
    tongyi_embedding_model: str = "text-embedding-v3"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    default_collection: str = "default"

    # 上传限制
    max_upload_size_mb: int = 50

    # 检索参数
    score_threshold: float = 0.7
    top_k: int = 5

    # 分块参数
    chunk_size: int = 1000
    chunk_overlap: int = 200
    # 扫描版 PDF 判断阈值：解析后总字符数低于此值视为扫描件
    min_text_length_for_scanned_detection: int = 50

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
```

- [ ] **Step 2: 编写 core/dependencies.py**

```python
# core/dependencies.py
from functools import lru_cache
from qdrant_client import QdrantClient
from core.config import settings


@lru_cache
def get_qdrant_client() -> QdrantClient:
    """返回 Qdrant client 单例。"""
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


# 注意：get_store / get_embedder / get_retriever 在各自模块实现后再导入，
# 避免循环依赖。见 Task 11/12/13 中对 dependencies.py 的补充。
```

- [ ] **Step 3: 验证配置可正常加载**

```bash
uv run python -c "from core.config import settings; print(settings.qdrant_url)"
```

Expected: 打印 `http://localhost:6333`（或 .env 中的值）。

- [ ] **Step 4: Commit**

```bash
git add core/config.py core/dependencies.py
git commit -m "feat: add core config and dependency injection scaffolding"
```

---

## Task 3: TextCleaner（TDD）

**Files:**
- Create: `document_processing/cleaners/text_cleaner.py`
- Create: `tests/test_text_cleaner.py`

- [ ] **Step 1: 先写失败测试**

```python
# tests/test_text_cleaner.py
import pytest
from langchain_core.documents import Document
from document_processing.cleaners.text_cleaner import TextCleaner


def test_removes_null_bytes_and_control_chars():
    cleaner = TextCleaner()
    doc = Document(page_content="hello\x00world\x01test\x1f", metadata={})
    result = cleaner.clean([doc])
    assert result[0].page_content == "helloworld test"


def test_normalizes_multiple_spaces():
    cleaner = TextCleaner()
    doc = Document(page_content="hello    world\t\there", metadata={})
    result = cleaner.clean([doc])
    assert result[0].page_content == "hello world here"


def test_normalizes_excessive_newlines():
    cleaner = TextCleaner()
    doc = Document(page_content="line1\n\n\n\n\nline2", metadata={})
    result = cleaner.clean([doc])
    assert result[0].page_content == "line1\n\nline2"


def test_strips_whitespace_only_lines():
    cleaner = TextCleaner()
    doc = Document(page_content="line1\n   \n  \nline2", metadata={})
    result = cleaner.clean([doc])
    assert result[0].page_content == "line1\nline2"


def test_preserves_metadata():
    cleaner = TextCleaner()
    meta = {"source": "test.pdf", "page": 1}
    doc = Document(page_content="hello world", metadata=meta)
    result = cleaner.clean([doc])
    assert result[0].metadata == meta


def test_handles_empty_content():
    cleaner = TextCleaner()
    doc = Document(page_content="   \n\t\n  ", metadata={})
    result = cleaner.clean([doc])
    assert result[0].page_content == ""


def test_returns_new_document_objects():
    cleaner = TextCleaner()
    doc = Document(page_content="hello", metadata={})
    result = cleaner.clean([doc])
    assert result[0] is not doc
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
uv run pytest tests/test_text_cleaner.py -v
```

Expected: `ImportError` 或 `ModuleNotFoundError`（文件未创建）。

- [ ] **Step 3: 实现 TextCleaner**

```python
# document_processing/cleaners/text_cleaner.py
import re
from langchain_core.documents import Document


class TextCleaner:
    """清洗文档文本：去除噪声字符、规范化空白、过滤空行。"""

    def clean(self, documents: list[Document]) -> list[Document]:
        """对所有 Document 应用清洗，返回新的 Document 对象列表。"""
        return [self._clean_doc(doc) for doc in documents]

    def _clean_doc(self, doc: Document) -> Document:
        text = doc.page_content

        # 去除 null 字节和控制字符（保留 \n \t）
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # 将多个空格/Tab 合并为单个空格
        text = re.sub(r"[ \t]+", " ", text)

        # 将连续 3 行以上的换行压缩为 2 行
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 过滤仅含空白的行
        lines = [line.strip() for line in text.split("\n")]
        lines = [line for line in lines if line]
        text = "\n".join(lines).strip()

        return Document(page_content=text, metadata=doc.metadata)
```

- [ ] **Step 4: 运行测试，确认全部通过**

```bash
uv run pytest tests/test_text_cleaner.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add document_processing/cleaners/text_cleaner.py tests/test_text_cleaner.py
git commit -m "feat: add TextCleaner with noise removal and whitespace normalization"
```

---

## Task 4: SmartChunker（TDD）

**Files:**
- Create: `document_processing/chunkers/smart_chunker.py`
- Create: `tests/test_smart_chunker.py`

- [ ] **Step 1: 先写失败测试**

```python
# tests/test_smart_chunker.py
from langchain_core.documents import Document
from document_processing.chunkers.smart_chunker import SmartChunker


def test_splits_long_text_into_multiple_chunks():
    chunker = SmartChunker(chunk_size=100, chunk_overlap=0)
    doc = Document(page_content="a" * 300, metadata={"source": "test.txt"})
    chunks = chunker.chunk([doc])
    assert len(chunks) > 1


def test_short_text_stays_single_chunk():
    chunker = SmartChunker(chunk_size=1000, chunk_overlap=0)
    doc = Document(page_content="hello world", metadata={})
    chunks = chunker.chunk([doc])
    assert len(chunks) == 1


def test_chunk_index_is_sequential_from_zero():
    chunker = SmartChunker(chunk_size=100, chunk_overlap=0)
    doc = Document(page_content="a" * 350, metadata={})
    chunks = chunker.chunk([doc])
    indices = [c.metadata["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))


def test_chunk_index_continues_across_documents():
    chunker = SmartChunker(chunk_size=100, chunk_overlap=0)
    docs = [
        Document(page_content="a" * 200, metadata={}),
        Document(page_content="b" * 200, metadata={}),
    ]
    chunks = chunker.chunk(docs)
    indices = [c.metadata["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))


def test_preserves_source_metadata_in_all_chunks():
    chunker = SmartChunker(chunk_size=100, chunk_overlap=0)
    doc = Document(page_content="a" * 300, metadata={"source": "doc.pdf", "page": 2})
    chunks = chunker.chunk([doc])
    assert all(c.metadata["source"] == "doc.pdf" for c in chunks)
    assert all(c.metadata["page"] == 2 for c in chunks)
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
uv run pytest tests/test_smart_chunker.py -v
```

Expected: `ImportError` 或 `ModuleNotFoundError`。

- [ ] **Step 3: 实现 SmartChunker**

```python
# document_processing/chunkers/smart_chunker.py
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class SmartChunker:
    """将 Document 列表切分为 chunk，并注入 chunk_index 元数据。"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # 中文优先按段落、句子分割
            separators=["\n\n", "\n", "。", ".", "！", "？", " ", ""],
        )

    def chunk(self, documents: list[Document]) -> list[Document]:
        """切分并为每个 chunk 注入连续递增的 chunk_index。"""
        result = []
        chunk_index = 0
        for doc in documents:
            chunks = self._splitter.split_documents([doc])
            for chunk in chunks:
                chunk.metadata["chunk_index"] = chunk_index
                chunk_index += 1
                result.append(chunk)
        return result
```

- [ ] **Step 4: 运行测试，确认全部通过**

```bash
uv run pytest tests/test_smart_chunker.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add document_processing/chunkers/smart_chunker.py tests/test_smart_chunker.py
git commit -m "feat: add SmartChunker with chunk_index metadata injection"
```

---

## Task 5: BaseLoader + TextLoader + CSVLoader（TDD）

**Files:**
- Create: `document_processing/loaders/base.py`
- Create: `document_processing/loaders/text_loader.py`
- Create: `document_processing/loaders/csv_loader.py`
- Create: `tests/test_loaders.py`（部分）

- [ ] **Step 1: 实现 BaseLoader 抽象类**

```python
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
```

- [ ] **Step 2: 先写 TextLoader 和 CSVLoader 的失败测试**

```python
# tests/test_loaders.py
import csv
import tempfile
from pathlib import Path
import pytest
from document_processing.loaders.text_loader import TextLoader
from document_processing.loaders.csv_loader import CSVLoader


# ── TextLoader ──────────────────────────────────────────────

class TestTextLoader:
    def test_loads_utf8_text_file(self, tmp_path):
        f = tmp_path / "sample.txt"
        f.write_text("Hello World\n第二行", encoding="utf-8")
        docs = TextLoader().load(f)
        assert len(docs) == 1
        assert "Hello World" in docs[0].page_content
        assert docs[0].metadata["source"] == "sample.txt"
        assert docs[0].metadata["file_type"] == "txt"

    def test_loads_gbk_text_file(self, tmp_path):
        f = tmp_path / "gbk.txt"
        f.write_bytes("中文内容".encode("gbk"))
        docs = TextLoader().load(f)
        assert "中文内容" in docs[0].page_content

    def test_supported_extensions(self):
        assert ".txt" in TextLoader().supported_extensions
        assert ".TXT" in TextLoader().supported_extensions


# ── CSVLoader ───────────────────────────────────────────────

class TestCSVLoader:
    def test_loads_csv_rows_as_documents(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("name,age\nAlice,30\nBob,25", encoding="utf-8")
        docs = CSVLoader().load(f)
        assert len(docs) == 2
        assert "Alice" in docs[0].page_content
        assert docs[0].metadata["file_type"] == "csv"
        assert docs[0].metadata["row"] == 0

    def test_skips_empty_rows(self, tmp_path):
        f = tmp_path / "empty.csv"
        f.write_text("name,value\nAlice,100\n,\nBob,200", encoding="utf-8")
        docs = CSVLoader().load(f)
        # 空行（name 和 value 都为空）应被跳过
        assert len(docs) == 2

    def test_supported_extensions(self):
        assert ".csv" in CSVLoader().supported_extensions
```

- [ ] **Step 3: 运行测试，确认失败**

```bash
uv run pytest tests/test_loaders.py -v
```

Expected: `ImportError`。

- [ ] **Step 4: 实现 TextLoader**

```python
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
```

- [ ] **Step 5: 实现 CSVLoader**

```python
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
```

- [ ] **Step 6: 运行测试，确认通过**

```bash
uv run pytest tests/test_loaders.py -v
```

Expected: 所有 TextLoader 和 CSVLoader 测试通过。

- [ ] **Step 7: Commit**

```bash
git add document_processing/loaders/ tests/test_loaders.py
git commit -m "feat: add BaseLoader, TextLoader, CSVLoader with encoding detection"
```

---

## Task 6: MarkdownLoader + PDFLoader + WordLoader（TDD）

**Files:**
- Create: `document_processing/loaders/markdown_loader.py`
- Create: `document_processing/loaders/pdf_loader.py`
- Create: `document_processing/loaders/word_loader.py`
- Modify: `tests/test_loaders.py`（追加测试）

- [ ] **Step 1: 追加失败测试到 tests/test_loaders.py**

```python
# 追加到 tests/test_loaders.py 末尾
from document_processing.loaders.markdown_loader import MarkdownLoader
from document_processing.loaders.pdf_loader import PDFLoader
from document_processing.loaders.word_loader import WordLoader


class TestMarkdownLoader:
    def test_loads_markdown_file(self, tmp_path):
        f = tmp_path / "readme.md"
        f.write_text("# Title\n\nSome content here.", encoding="utf-8")
        docs = MarkdownLoader().load(f)
        assert len(docs) >= 1
        assert docs[0].metadata["file_type"] == "md"
        assert docs[0].metadata["source"] == "readme.md"

    def test_supported_extensions(self):
        exts = MarkdownLoader().supported_extensions
        assert ".md" in exts
        assert ".markdown" in exts


class TestPDFLoader:
    def test_raises_on_scanned_pdf(self, tmp_path):
        """扫描版 PDF（无文本层）应抛出 ValueError，提示需要 OCR。"""
        import pypdf
        from pypdf import PdfWriter
        writer = PdfWriter()
        writer.add_blank_page(width=595, height=842)
        f = tmp_path / "scanned.pdf"
        with open(f, "wb") as fh:
            writer.write(fh)
        with pytest.raises(ValueError, match="scanned"):
            PDFLoader().load(f)

    def test_supported_extensions(self):
        assert ".pdf" in PDFLoader().supported_extensions


class TestWordLoader:
    def test_loads_docx_file(self, tmp_path):
        from docx import Document as DocxDocument
        f = tmp_path / "test.docx"
        doc = DocxDocument()
        doc.add_paragraph("Hello from Word")
        doc.save(str(f))
        docs = WordLoader().load(f)
        assert len(docs) >= 1
        assert "Hello from Word" in docs[0].page_content
        assert docs[0].metadata["file_type"] == "docx"

    def test_supported_extensions(self):
        assert ".docx" in WordLoader().supported_extensions
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
uv run pytest tests/test_loaders.py::TestMarkdownLoader tests/test_loaders.py::TestPDFLoader tests/test_loaders.py::TestWordLoader -v
```

Expected: `ImportError`。

- [ ] **Step 3: 实现 MarkdownLoader**

```python
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
```

- [ ] **Step 4: 实现 PDFLoader**

```python
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
```

- [ ] **Step 5: 实现 WordLoader**

```python
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
```

- [ ] **Step 6: 运行全部 loader 测试**

```bash
uv run pytest tests/test_loaders.py -v
```

Expected: 全部通过。

- [ ] **Step 7: Commit**

```bash
git add document_processing/loaders/markdown_loader.py \
        document_processing/loaders/pdf_loader.py \
        document_processing/loaders/word_loader.py \
        tests/test_loaders.py
git commit -m "feat: add MarkdownLoader, PDFLoader (scan detection), WordLoader"
```

---

## Task 7: DocumentPipeline（TDD）

**Files:**
- Create: `document_processing/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: 先写失败测试**

```python
# tests/test_pipeline.py
import hashlib
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from langchain_core.documents import Document
from document_processing.pipeline import DocumentPipeline, compute_md5, get_loader


class TestComputeMd5:
    def test_returns_32_char_hex_string(self):
        result = compute_md5(b"hello world")
        assert len(result) == 32
        assert all(c in "0123456789abcdef" for c in result)

    def test_same_bytes_same_md5(self):
        assert compute_md5(b"data") == compute_md5(b"data")

    def test_different_bytes_different_md5(self):
        assert compute_md5(b"data1") != compute_md5(b"data2")


class TestGetLoader:
    def test_returns_loader_for_supported_extension(self):
        loader = get_loader(".txt")
        assert loader is not None

    def test_raises_for_unsupported_extension(self):
        with pytest.raises(ValueError, match="Unsupported"):
            get_loader(".xyz")


class TestDocumentPipeline:
    def test_processes_txt_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello World\nSecond line here", encoding="utf-8")
        pipeline = DocumentPipeline()
        chunks = pipeline.process(f, "abc123md5", "default")
        assert len(chunks) >= 1
        assert all(c.metadata["file_md5"] == "abc123md5" for c in chunks)
        assert all(c.metadata["collection"] == "default" for c in chunks)
        assert all("created_at" in c.metadata for c in chunks)

    def test_raises_for_empty_document(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("   \n\t\n  ", encoding="utf-8")
        pipeline = DocumentPipeline()
        with pytest.raises(ValueError, match="No extractable text"):
            pipeline.process(f, "abc123", "default")
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
uv run pytest tests/test_pipeline.py -v
```

Expected: `ImportError`。

- [ ] **Step 3: 实现 pipeline.py**

```python
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
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
uv run pytest tests/test_pipeline.py -v
```

Expected: 全部通过。

- [ ] **Step 5: Commit**

```bash
git add document_processing/pipeline.py tests/test_pipeline.py
git commit -m "feat: add DocumentPipeline orchestrating loader→cleaner→chunker"
```

---

## Task 8: TongyiEmbedder（TDD with mock）

**Files:**
- Create: `embeddings/tongyi_embedder.py`
- Create: `tests/test_tongyi_embedder.py`

- [ ] **Step 1: 先写失败测试**

```python
# tests/test_tongyi_embedder.py
from unittest.mock import MagicMock, patch
import pytest
from embeddings.tongyi_embedder import TongyiEmbedder


class TestTongyiEmbedder:
    def test_embed_documents_calls_underlying_embedder(self):
        with patch("embeddings.tongyi_embedder.DashScopeEmbeddings") as MockEmbed:
            mock_instance = MagicMock()
            mock_instance.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
            MockEmbed.return_value = mock_instance

            embedder = TongyiEmbedder()
            result = embedder.embed_documents(["text1", "text2"])

            assert result == [[0.1, 0.2], [0.3, 0.4]]
            mock_instance.embed_documents.assert_called_once_with(["text1", "text2"])

    def test_embed_query_returns_vector(self):
        with patch("embeddings.tongyi_embedder.DashScopeEmbeddings") as MockEmbed:
            mock_instance = MagicMock()
            mock_instance.embed_query.return_value = [0.5, 0.6, 0.7]
            MockEmbed.return_value = mock_instance

            embedder = TongyiEmbedder()
            result = embedder.embed_query("search query")

            assert result == [0.5, 0.6, 0.7]

    def test_retries_on_exception_and_succeeds(self):
        """第一次失败，第二次成功，应返回第二次结果。"""
        with patch("embeddings.tongyi_embedder.DashScopeEmbeddings") as MockEmbed:
            mock_instance = MagicMock()
            mock_instance.embed_query.side_effect = [
                RuntimeError("rate limit"),
                [0.1, 0.2],
            ]
            MockEmbed.return_value = mock_instance

            with patch("embeddings.tongyi_embedder.time.sleep"):  # 不真实等待
                embedder = TongyiEmbedder()
                result = embedder.embed_query("query")

            assert result == [0.1, 0.2]

    def test_raises_after_max_retries(self):
        """连续 3 次失败后应抛出异常。"""
        with patch("embeddings.tongyi_embedder.DashScopeEmbeddings") as MockEmbed:
            mock_instance = MagicMock()
            mock_instance.embed_query.side_effect = RuntimeError("timeout")
            MockEmbed.return_value = mock_instance

            with patch("embeddings.tongyi_embedder.time.sleep"):
                embedder = TongyiEmbedder()
                with pytest.raises(RuntimeError, match="timeout"):
                    embedder.embed_query("query")
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
uv run pytest tests/test_tongyi_embedder.py -v
```

Expected: `ImportError`。

- [ ] **Step 3: 实现 TongyiEmbedder**

```python
# embeddings/tongyi_embedder.py
import logging
import time

from langchain_community.embeddings import DashScopeEmbeddings

from core.config import settings

logger = logging.getLogger(__name__)


class TongyiEmbedder:
    """封装 DashScope Tongyi Embedding，支持指数退避重试（最多 3 次）。"""

    MAX_RETRIES = 3

    def __init__(self) -> None:
        self._embedder = DashScopeEmbeddings(
            model=settings.tongyi_embedding_model,
            dashscope_api_key=settings.dashscope_api_key,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量向量化文本列表。"""
        return self._with_retry(lambda: self._embedder.embed_documents(texts))

    def embed_query(self, text: str) -> list[float]:
        """向量化单条查询文本。"""
        return self._with_retry(lambda: self._embedder.embed_query(text))

    def _with_retry(self, fn):
        """指数退避重试：1s → 2s → 4s，3 次失败后抛出最后一次异常。"""
        delay = 1
        last_exc: Exception | None = None
        for attempt in range(self.MAX_RETRIES):
            try:
                return fn()
            except Exception as exc:
                last_exc = exc
                if attempt < self.MAX_RETRIES - 1:
                    logger.warning(
                        "Embedding attempt %d/%d failed: %s. Retrying in %ds...",
                        attempt + 1,
                        self.MAX_RETRIES,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= 2
        raise last_exc
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
uv run pytest tests/test_tongyi_embedder.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add embeddings/tongyi_embedder.py tests/test_tongyi_embedder.py
git commit -m "feat: add TongyiEmbedder with exponential backoff retry"
```

---

## Task 9: QdrantStore（TDD with mock）

**Files:**
- Create: `vectorstore/qdrant_store.py`
- Create: `tests/test_qdrant_store.py`

- [ ] **Step 1: 先写失败测试**

```python
# tests/test_qdrant_store.py
from unittest.mock import MagicMock, patch
import pytest
from langchain_core.documents import Document
from vectorstore.qdrant_store import QdrantStore


def make_store() -> tuple[QdrantStore, MagicMock]:
    """返回 (store, mock_client) 元组，方便测试中断言 client 调用。"""
    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = []
    store = QdrantStore(client=mock_client)
    return store, mock_client


class TestEnsureCollection:
    def test_creates_collection_when_not_exists(self):
        store, mock_client = make_store()
        mock_client.get_collections.return_value.collections = []
        store.ensure_collection("new_col")
        mock_client.create_collection.assert_called_once()

    def test_skips_creation_when_already_exists(self):
        store, mock_client = make_store()
        existing = MagicMock()
        existing.name = "existing"
        mock_client.get_collections.return_value.collections = [existing]
        store.ensure_collection("existing")
        mock_client.create_collection.assert_not_called()


class TestMd5Exists:
    def test_returns_true_when_md5_found(self):
        store, mock_client = make_store()
        mock_client.scroll.return_value = ([MagicMock()], None)
        assert store.md5_exists("abc123", "default") is True

    def test_returns_false_when_md5_not_found(self):
        store, mock_client = make_store()
        mock_client.scroll.return_value = ([], None)
        assert store.md5_exists("abc123", "default") is False

    def test_returns_false_when_collection_not_exists(self):
        store, mock_client = make_store()
        mock_client.scroll.side_effect = Exception("collection not found")
        assert store.md5_exists("abc123", "new_collection") is False


class TestUpsert:
    def test_calls_client_upsert_with_correct_payload(self):
        store, mock_client = make_store()
        chunks = [Document(page_content="hello", metadata={"source": "f.txt", "file_md5": "md5"})]
        vectors = [[0.1, 0.2, 0.3]]
        store.upsert(chunks, vectors, "default")
        mock_client.upsert.assert_called_once()
        call_kwargs = mock_client.upsert.call_args[1]
        assert call_kwargs["collection_name"] == "default"
        point = call_kwargs["points"][0]
        assert point.payload["text"] == "hello"
        assert point.payload["file_md5"] == "md5"


class TestDeleteByMd5:
    def test_deletes_points_and_returns_count(self):
        store, mock_client = make_store()
        p1, p2 = MagicMock(id="id1"), MagicMock(id="id2")
        mock_client.scroll.return_value = ([p1, p2], None)
        count = store.delete_by_md5("abc", "default")
        assert count == 2
        mock_client.delete.assert_called_once()

    def test_returns_zero_when_not_found(self):
        store, mock_client = make_store()
        mock_client.scroll.return_value = ([], None)
        count = store.delete_by_md5("nope", "default")
        assert count == 0
        mock_client.delete.assert_not_called()


class TestSearch:
    def test_returns_formatted_results(self):
        store, mock_client = make_store()
        hit = MagicMock()
        hit.payload = {"text": "content", "source": "file.pdf", "page": 1}
        hit.score = 0.95
        mock_client.search.return_value = [hit]
        results = store.search([0.1, 0.2], "default", top_k=3, score_threshold=0.7)
        assert len(results) == 1
        assert results[0]["content"] == "content"
        assert results[0]["score"] == 0.95
        assert "text" not in results[0]["metadata"]  # text 不重复放进 metadata
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
uv run pytest tests/test_qdrant_store.py -v
```

Expected: `ImportError`。

- [ ] **Step 3: 实现 QdrantStore**

```python
# vectorstore/qdrant_store.py
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from langchain_core.documents import Document


class QdrantStore:
    """封装 Qdrant 向量库操作：建 collection、MD5 查重、upsert、删除、检索。"""

    # tongyi-embedding-vision-plus 向量维度，实际以 DashScope 文档为准
    VECTOR_SIZE = 1536

    def __init__(self, client: QdrantClient) -> None:
        self._client = client

    def ensure_collection(self, collection: str) -> None:
        """若 collection 不存在则自动创建（余弦相似度）。"""
        existing_names = [c.name for c in self._client.get_collections().collections]
        if collection not in existing_names:
            self._client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=self.VECTOR_SIZE, distance=Distance.COSINE),
            )

    def md5_exists(self, file_md5: str, collection: str) -> bool:
        """检查该 MD5 是否已在 collection 中存在（去重判断）。
        若 collection 不存在则返回 False。"""
        try:
            results, _ = self._client.scroll(
                collection_name=collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="file_md5", match=MatchValue(value=file_md5))]
                ),
                limit=1,
            )
            return len(results) > 0
        except Exception:
            # Collection 尚不存在，视为未入库
            return False

    def upsert(
        self,
        chunks: list[Document],
        vectors: list[list[float]],
        collection: str,
    ) -> None:
        """将 chunks 及对应向量写入 Qdrant。自动创建 collection。"""
        self.ensure_collection(collection)
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                # text 字段存储原始内容，其余 metadata 平铺
                payload={"text": chunk.page_content, **chunk.metadata},
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        self._client.upsert(collection_name=collection, points=points)

    def delete_by_md5(self, file_md5: str, collection: str) -> int:
        """删除 collection 中所有匹配该 MD5 的 point，返回删除数量。"""
        scroll_result, _ = self._client.scroll(
            collection_name=collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="file_md5", match=MatchValue(value=file_md5))]
            ),
            limit=10000,
        )
        point_ids = [p.id for p in scroll_result]
        if point_ids:
            self._client.delete(
                collection_name=collection,
                points_selector=point_ids,
            )
        return len(point_ids)

    def search(
        self,
        query_vector: list[float],
        collection: str,
        top_k: int,
        score_threshold: float,
    ) -> list[dict]:
        """语义检索，返回高于阈值的 top-k 结果。"""
        hits = self._client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return [
            {
                "content": hit.payload.get("text", ""),
                "source": hit.payload.get("source", ""),
                "score": hit.score,
                # metadata 中排除 text 字段（避免重复）
                "metadata": {k: v for k, v in hit.payload.items() if k != "text"},
            }
            for hit in hits
        ]

    def list_collections(self) -> list[str]:
        """列出所有 collection 名称。"""
        return [c.name for c in self._client.get_collections().collections]
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
uv run pytest tests/test_qdrant_store.py -v
```

Expected: 全部通过。

- [ ] **Step 5: Commit**

```bash
git add vectorstore/qdrant_store.py tests/test_qdrant_store.py
git commit -m "feat: add QdrantStore with MD5 dedup, upsert, delete, search"
```

---

## Task 10: Retriever（TDD with mock）

**Files:**
- Create: `retrieval/retriever.py`
- Create: `tests/test_retriever.py`

- [ ] **Step 1: 先写失败测试**

```python
# tests/test_retriever.py
from unittest.mock import MagicMock
from retrieval.retriever import Retriever


class TestRetriever:
    def _make_retriever(self):
        mock_embedder = MagicMock()
        mock_store = MagicMock()
        mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_store.search.return_value = [
            {"content": "result text", "source": "file.pdf", "score": 0.9, "metadata": {}}
        ]
        return Retriever(embedder=mock_embedder, store=mock_store), mock_embedder, mock_store

    def test_embeds_query_and_calls_search(self):
        retriever, mock_embedder, mock_store = self._make_retriever()
        results = retriever.search("test query", collection="default", top_k=3, score_threshold=0.7)
        mock_embedder.embed_query.assert_called_once_with("test query")
        mock_store.search.assert_called_once_with(
            query_vector=[0.1, 0.2, 0.3],
            collection="default",
            top_k=3,
            score_threshold=0.7,
        )
        assert results[0]["content"] == "result text"

    def test_uses_default_top_k_and_threshold_from_settings(self):
        retriever, mock_embedder, mock_store = self._make_retriever()
        retriever.search("query", collection="default")
        call_kwargs = mock_store.search.call_args[1]
        assert isinstance(call_kwargs["top_k"], int)
        assert isinstance(call_kwargs["score_threshold"], float)

    def test_returns_empty_list_when_no_results(self):
        retriever, mock_embedder, mock_store = self._make_retriever()
        mock_store.search.return_value = []
        results = retriever.search("obscure query", collection="default")
        assert results == []
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
uv run pytest tests/test_retriever.py -v
```

Expected: `ImportError`。

- [ ] **Step 3: 实现 Retriever**

```python
# retrieval/retriever.py
from embeddings.tongyi_embedder import TongyiEmbedder
from vectorstore.qdrant_store import QdrantStore
from core.config import settings


class Retriever:
    """语义检索：将 query 向量化后在 Qdrant 中检索最相关结果。"""

    def __init__(self, embedder: TongyiEmbedder, store: QdrantStore) -> None:
        self._embedder = embedder
        self._store = store

    def search(
        self,
        query: str,
        collection: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[dict]:
        """
        执行语义检索。

        Args:
            query: 自然语言查询字符串。
            collection: Qdrant collection 名称。
            top_k: 返回结果数，默认读取 settings.top_k。
            score_threshold: 相似度阈值，默认读取 settings.score_threshold。

        Returns:
            List of dicts with keys: content, source, score, metadata.
        """
        effective_top_k = top_k if top_k is not None else settings.top_k
        effective_threshold = (
            score_threshold if score_threshold is not None else settings.score_threshold
        )

        query_vector = self._embedder.embed_query(query)
        return self._store.search(
            query_vector=query_vector,
            collection=collection,
            top_k=effective_top_k,
            score_threshold=effective_threshold,
        )
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
uv run pytest tests/test_retriever.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add retrieval/retriever.py tests/test_retriever.py
git commit -m "feat: add Retriever wrapping embed_query + qdrant search"
```

---

## Task 11: 完善 dependencies.py（单例注入）

**Files:**
- Modify: `core/dependencies.py`

- [ ] **Step 1: 补全 dependencies.py**

```python
# core/dependencies.py
from functools import lru_cache

from qdrant_client import QdrantClient

from core.config import settings


@lru_cache
def get_qdrant_client() -> QdrantClient:
    """返回 Qdrant client 单例（进程级缓存）。"""
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


@lru_cache
def get_store():
    """返回 QdrantStore 单例。"""
    from vectorstore.qdrant_store import QdrantStore
    return QdrantStore(client=get_qdrant_client())


@lru_cache
def get_embedder():
    """返回 TongyiEmbedder 单例。"""
    from embeddings.tongyi_embedder import TongyiEmbedder
    return TongyiEmbedder()


@lru_cache
def get_retriever():
    """返回 Retriever 单例。"""
    from retrieval.retriever import Retriever
    return Retriever(embedder=get_embedder(), store=get_store())
```

- [ ] **Step 2: 验证导入无报错**

```bash
uv run python -c "from core.dependencies import get_store, get_embedder, get_retriever; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add core/dependencies.py
git commit -m "feat: complete dependency injection with lru_cache singletons"
```

---

## Task 12: Documents API（TDD with TestClient）

**Files:**
- Create: `api/documents.py`
- Create: `tests/test_documents_api.py`

- [ ] **Step 1: 先写失败测试**

```python
# tests/test_documents_api.py
import io
from unittest.mock import MagicMock, AsyncMock
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

from api.documents import router
from core.dependencies import get_store, get_embedder, get_retriever


def make_test_app():
    """构造测试用 FastAPI app，覆盖依赖为 mock。"""
    app = FastAPI()
    app.state.tasks = {}
    app.include_router(router)

    mock_store = MagicMock()
    mock_store.md5_exists.return_value = False
    mock_store.upsert.return_value = None
    mock_store.list_collections.return_value = ["default"]
    mock_store.delete_by_md5.return_value = 5
    mock_store.search.return_value = []

    mock_embedder = MagicMock()
    mock_embedder.embed_documents.return_value = [[0.1] * 10]

    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [
        {"content": "some result", "source": "file.pdf", "score": 0.9, "metadata": {}}
    ]

    app.dependency_overrides[get_store] = lambda: mock_store
    app.dependency_overrides[get_embedder] = lambda: mock_embedder
    app.dependency_overrides[get_retriever] = lambda: mock_retriever

    return app, mock_store, mock_embedder, mock_retriever


class TestUpload:
    def test_upload_txt_returns_processing_status(self):
        app, mock_store, _, _ = make_test_app()
        client = TestClient(app)
        response = client.post(
            "/documents/upload",
            files={"file": ("test.txt", b"Hello World content", "text/plain")},
            data={"collection": "default"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "processing"
        assert "task_id" in body

    def test_upload_rejects_unsupported_format(self):
        app, _, _, _ = make_test_app()
        client = TestClient(app)
        response = client.post(
            "/documents/upload",
            files={"file": ("file.exe", b"binary", "application/octet-stream")},
        )
        assert response.status_code == 400

    def test_upload_rejects_duplicate_md5(self):
        app, mock_store, _, _ = make_test_app()
        mock_store.md5_exists.return_value = True
        client = TestClient(app)
        response = client.post(
            "/documents/upload",
            files={"file": ("test.txt", b"Hello World", "text/plain")},
        )
        assert response.status_code == 409

    def test_upload_rejects_oversized_file(self):
        app, _, _, _ = make_test_app()
        # 临时将 MAX_UPLOAD_BYTES 设置极小，测试限制逻辑
        import api.documents as docs_module
        original = docs_module.MAX_UPLOAD_BYTES
        docs_module.MAX_UPLOAD_BYTES = 5
        client = TestClient(app)
        response = client.post(
            "/documents/upload",
            files={"file": ("test.txt", b"This content is too long", "text/plain")},
        )
        docs_module.MAX_UPLOAD_BYTES = original
        assert response.status_code == 413


class TestStatus:
    def test_returns_task_status(self):
        app, _, _, _ = make_test_app()
        app.state.tasks["task-abc"] = {"status": "done", "error": None}
        client = TestClient(app)
        response = client.get("/documents/status/task-abc")
        assert response.status_code == 200
        assert response.json()["status"] == "done"

    def test_returns_404_for_unknown_task(self):
        app, _, _, _ = make_test_app()
        client = TestClient(app)
        response = client.get("/documents/status/nonexistent")
        assert response.status_code == 404


class TestSearch:
    def test_returns_search_results(self):
        app, _, _, _ = make_test_app()
        client = TestClient(app)
        response = client.post(
            "/documents/search",
            json={"query": "test query", "collection": "default", "top_k": 3, "score_threshold": 0.7},
        )
        assert response.status_code == 200
        assert "results" in response.json()


class TestDelete:
    def test_delete_returns_deleted_count(self):
        app, mock_store, _, _ = make_test_app()
        client = TestClient(app)
        response = client.delete("/documents/abc123md5?collection=default")
        assert response.status_code == 200
        assert response.json()["deleted_chunks"] == 5

    def test_delete_returns_404_when_not_found(self):
        app, mock_store, _, _ = make_test_app()
        mock_store.delete_by_md5.return_value = 0
        client = TestClient(app)
        response = client.delete("/documents/nonexistent?collection=default")
        assert response.status_code == 404


class TestListCollections:
    def test_returns_collection_names(self):
        app, _, _, _ = make_test_app()
        client = TestClient(app)
        response = client.get("/documents/collections")
        assert response.status_code == 200
        assert "collections" in response.json()
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
uv run pytest tests/test_documents_api.py -v
```

Expected: `ImportError`。

- [ ] **Step 3: 实现 api/documents.py**

```python
# api/documents.py
import logging
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from core.config import settings
from core.dependencies import get_embedder, get_retriever, get_store
from document_processing.pipeline import DocumentPipeline, compute_md5, get_loader
from embeddings.tongyi_embedder import TongyiEmbedder
from retrieval.retriever import Retriever
from vectorstore.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

SUPPORTED_EXTENSIONS = {
    ".pdf", ".PDF",
    ".docx", ".DOCX",
    ".md", ".MD", ".markdown",
    ".csv", ".CSV",
    ".txt", ".TXT",
}

MAX_UPLOAD_BYTES = settings.max_upload_size_mb * 1024 * 1024


class SearchRequest(BaseModel):
    query: str
    collection: str = "default"
    top_k: int = 5
    score_threshold: float = 0.7


async def _process_document(
    tmp_path: str,
    file_name: str,
    file_md5: str,
    collection: str,
    task_id: str,
    tasks: dict,
    store: QdrantStore,
    embedder: TongyiEmbedder,
) -> None:
    """后台异步处理：pipeline → embed → upsert。失败时回滚并标记状态。"""
    try:
        pipeline = DocumentPipeline()
        chunks = pipeline.process(Path(tmp_path), file_md5, collection)

        texts = [c.page_content for c in chunks]
        vectors = embedder.embed_documents(texts)

        # 二次 MD5 检查防并发竞态（若已存在则先删旧版本）
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

    # 3. MD5 去重
    file_md5 = compute_md5(content)
    if store.md5_exists(file_md5, collection):
        raise HTTPException(
            status_code=409,
            detail={"status": "duplicate", "file_md5": file_md5},
        )

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
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
uv run pytest tests/test_documents_api.py -v
```

Expected: 全部通过。

- [ ] **Step 5: Commit**

```bash
git add api/documents.py tests/test_documents_api.py
git commit -m "feat: add documents REST API with upload, status, search, delete"
```

---

## Task 13: 更新 main.py 并运行全量测试

**Files:**
- Modify: `main.py`

- [ ] **Step 1: 更新 main.py**

```python
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
```

- [ ] **Step 2: 运行全量测试**

```bash
uv run pytest tests/ -v
```

Expected: 所有测试通过，0 failures。

- [ ] **Step 3: 验证 FastAPI 启动无报错**

```bash
uv run uvicorn main:app --reload --port 8000
```

Expected: `INFO: Application startup complete.`

- [ ] **Step 4: 访问 API 文档确认路由注册正确**

浏览器打开 `http://localhost:8000/docs`，确认可见：
- `POST /documents/upload`
- `GET /documents/status/{task_id}`
- `POST /documents/search`
- `DELETE /documents/{file_md5}`
- `GET /documents/collections`
- `GET /health`

- [ ] **Step 5: Final commit**

```bash
git add main.py
git commit -m "feat: wire up FastAPI app with documents router and health endpoint"
```

---

## 附录：环境变量说明

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DASHSCOPE_API_KEY` | 阿里云 DashScope API Key（必填） | 无 |
| `TONGYI_EMBEDDING_MODEL` | Embedding 模型 ID | `text-embedding-v3` |
| `QDRANT_URL` | Qdrant 服务地址 | `http://localhost:6333` |
| `QDRANT_API_KEY` | Qdrant API Key（云端部署时填写） | 空 |
| `MAX_UPLOAD_SIZE_MB` | 单文件最大上传大小 | `50` |
| `CHUNK_SIZE` | 分块大小（字符数） | `1000` |
| `CHUNK_OVERLAP` | 分块重叠（字符数） | `200` |
| `SCORE_THRESHOLD` | 检索相似度阈值 | `0.7` |
| `TOP_K` | 默认返回结果数 | `5` |

> **注意：** `TONGYI_EMBEDDING_MODEL` 的实际值需对照 DashScope 控制台确认，`tongyi-embedding-vision-plus` 是展示名，API 调用时使用的 model ID 可能不同（如 `multimodal-embedding-one-peace-v1`）。
