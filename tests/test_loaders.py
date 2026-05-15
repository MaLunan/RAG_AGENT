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
