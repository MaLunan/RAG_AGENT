# chat/rag_chain.py
import asyncio
import logging
from typing import Any

from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_models import ChatTongyi
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables.history import RunnableWithMessageHistory

from chat.long_term_memory import LongTermMemory
from chat.mongo_logger import MongoLogger
from chat.session_store import get_session_history
from core.config import settings

logger = logging.getLogger(__name__)


class _QdrantLCRetriever(BaseRetriever):
    """将自定义 Retriever 包装为 LangChain BaseRetriever。"""

    retriever: Any
    collection: str
    top_k: int
    score_threshold: float

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        results = self.retriever.search(
            query=query,
            collection=self.collection,
            top_k=self.top_k,
            score_threshold=self.score_threshold,
        )
        return [
            Document(page_content=r["content"], metadata=r["metadata"])
            for r in results
        ]


class ChatEngine:
    """多轮 RAG 对话引擎：整合 LangChain RAG 链、Redis 会话记忆、MongoDB 日志、Qdrant 长期记忆。"""

    def __init__(self, retriever, embedder, store) -> None:
        self._retriever = retriever
        self._long_term = LongTermMemory(store=store, embedder=embedder)
        self._mongo_logger = MongoLogger()
        self._llm = ChatTongyi(
            model=settings.tongyi_llm_model,
            dashscope_api_key=settings.dashscope_api_key,
        )

    def _build_chain(
        self, collection: str, top_k: int, score_threshold: float
    ) -> RunnableWithMessageHistory:
        lc_retriever = _QdrantLCRetriever(
            retriever=self._retriever,
            collection=collection,
            top_k=top_k,
            score_threshold=score_threshold,
        )

        # 1. 将带历史的问题改写为独立完整问题
        contextualize_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "根据对话历史，将用户问题改写为一个独立完整的问题（保持原意，无需历史也能理解）。"
                "若已独立则直接返回原问题。",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(
            self._llm, lc_retriever, contextualize_prompt
        )

        # 2. 基于检索内容回答
        qa_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "你是知识库问答助手。以下是从知识库中检索到的相关文档片段：\n\n"
                "{context}\n\n"
                "请根据上述文档内容，直接提取并回答用户的问题。\n"
                "- 答案要简洁准确，直接给出用户想要的具体信息（如姓名、邮箱、电话、日期等）\n"
                "- 文档中的信息可能以换行、竖线或空格分隔，请仔细识别\n"
                "- 只有在文档中确实找不到相关信息时，才回复【知识库中未找到相关信息】",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        qa_chain = create_stuff_documents_chain(self._llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

        return RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def chat(
        self,
        session_id: str,
        query: str,
        collection: str,
        top_k: int,
        score_threshold: float,
    ) -> dict:
        """执行多轮问答，返回 answer 和 sources。"""
        chain = self._build_chain(collection, top_k, score_threshold)
        result = chain.invoke(
            {"input": query},
            config={"configurable": {"session_id": session_id}},
        )
        answer = result["answer"]
        sources = [
            {
                "source": d.metadata.get("source", ""),
                "content": d.page_content[:200],
            }
            for d in result.get("context", [])
        ]

        # 异步持久化（不阻塞响应）
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._mongo_logger.log(session_id, query, answer, sources))
        except RuntimeError:
            # 没有运行中的事件循环（单元测试环境）
            pass
        except Exception:
            logger.warning("MongoDB logging failed, continuing without log.")

        try:
            self._long_term.save(session_id, query, answer)
        except Exception:
            logger.warning("Long-term memory save failed, continuing.")

        return {"session_id": session_id, "answer": answer, "sources": sources}
