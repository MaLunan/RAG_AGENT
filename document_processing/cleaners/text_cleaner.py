# document_processing/cleaners/text_cleaner.py
import re
from langchain_core.documents import Document


class TextCleaner:
    """清洗文档文本：去除噪声字符、规范化空白、过滤空行。"""

    def clean(self, documents: list[Document]) -> list[Document]:
        """对所有 Document 应用清洗，返回新的 Document 对象列表。"""
        return [self._clean_doc(doc) for doc in documents]

    def _clean_doc(self, doc: Document) -> Document:
        text = doc.page_content

        # 去除 null 字节（\x00）和 DEL（\x7f）
        text = re.sub(r"[\x00\x7f]", "", text)

        # 将其余控制字符（保留 \n \t）替换为空格
        text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f]", " ", text)

        # 将多个空格/Tab 合并为单个空格
        text = re.sub(r"[ \t]+", " ", text)

        # 处理各行：
        # - 真正空行（原始为 ""）：保留但压缩连续多个为单个空行
        # - 仅含空白的行（原始非空但 strip 后为空）：直接删除
        # - 有内容的行：strip 后保留
        lines = text.split("\n")
        result_lines = []
        prev_empty = False
        for line in lines:
            stripped = line.strip()
            if line == "":
                # 真正空行：压缩连续空行
                if not prev_empty:
                    result_lines.append("")
                prev_empty = True
            elif stripped == "":
                # 仅含空白的行：丢弃
                pass
            else:
                result_lines.append(stripped)
                prev_empty = False

        # 去除首尾的空行
        while result_lines and result_lines[0] == "":
            result_lines.pop(0)
        while result_lines and result_lines[-1] == "":
            result_lines.pop()

        text = "\n".join(result_lines)

        return Document(page_content=text, metadata=doc.metadata)
