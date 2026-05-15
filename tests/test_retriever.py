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
