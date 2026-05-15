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
