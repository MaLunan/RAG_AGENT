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
