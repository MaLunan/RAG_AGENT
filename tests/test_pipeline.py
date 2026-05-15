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
