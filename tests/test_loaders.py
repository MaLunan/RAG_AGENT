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
