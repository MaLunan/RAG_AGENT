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
