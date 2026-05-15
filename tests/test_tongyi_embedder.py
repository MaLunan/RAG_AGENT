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
