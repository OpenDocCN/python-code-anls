# `.\DB-GPT-src\dbgpt\rag\text_splitter\pre_text_splitter.py`

```py
"""Pre text splitter."""
# 从模块导入所需的类型和类
from typing import Iterable, List

from dbgpt.core import Chunk, Document  # 导入核心类 Chunk 和 Document
from dbgpt.rag.text_splitter.text_splitter import TextSplitter  # 导入文本分割器类 TextSplitter


class PreTextSplitter(TextSplitter):
    """Split text by pre separator."""

    def __init__(self, pre_separator: str, text_splitter_impl: TextSplitter):
        """Create the pre text splitter instance.

        Args:
            pre_separator: pre separator
            text_splitter_impl: text splitter impl
        """
        self.pre_separator = pre_separator  # 初始化预分隔符属性
        self._impl = text_splitter_impl  # 初始化文本分割器实现属性

    def split_text(self, text: str, **kwargs) -> List[str]:
        """Split text by pre separator."""
        return self._impl.split_text(text)  # 调用文本分割器实现的文本分割方法

    def split_documents(self, documents: Iterable[Document], **kwargs) -> List[Chunk]:
        """Split documents by pre separator."""

        def generator() -> Iterable[Document]:
            for doc in documents:
                # 使用单个文档分割函数分割每个文档，并生成文档块
                yield from _single_document_split(doc, pre_separator=self.pre_separator)

        return self._impl.split_documents(generator())
        
def _single_document_split(
    document: Document, pre_separator: str
) -> Iterable[Document]:
    origin_content = document.content  # 获取文档的原始内容
    for i, content in enumerate(origin_content.split(pre_separator)):
        metadata = document.metadata.copy()  # 复制文档的元数据
        if "source" in metadata:
            metadata["source"] = metadata["source"] + "_pre_split_" + str(i)  # 更新源信息键
        yield Chunk(content=content, metadata=metadata)  # 返回带有内容和元数据的文档块
```