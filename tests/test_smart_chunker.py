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
