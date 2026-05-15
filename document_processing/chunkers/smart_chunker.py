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
