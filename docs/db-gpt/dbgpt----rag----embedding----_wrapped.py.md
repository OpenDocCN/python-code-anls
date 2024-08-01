# `.\DB-GPT-src\dbgpt\rag\embedding\_wrapped.py`

```py
"""Wraps the third-party language model embeddings to the common interface."""

# 引入类型检查和列表类型
from typing import TYPE_CHECKING, List

# 引入自定义的Embeddings类
from dbgpt.core import Embeddings

# 如果是类型检查模式，引入LangChainEmbeddings类
if TYPE_CHECKING:
    from langchain.embeddings.base import (
        Embeddings as LangChainEmbeddings,  # mypy: ignore
    )

# 继承自Embeddings类，将第三方语言模型嵌入封装成通用接口
class WrappedEmbeddings(Embeddings):
    """Wraps the third-party language model embeddings to the common interface."""

    def __init__(self, embeddings: "LangChainEmbeddings") -> None:
        """Create a new WrappedEmbeddings."""
        self._embeddings = embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        # 调用内部嵌入对象的embed_documents方法，将文本嵌入为向量列表
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        # 调用内部嵌入对象的embed_query方法，将查询文本嵌入为向量
        return self._embeddings.embed_query(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs."""
        # 异步调用内部嵌入对象的aembed_documents方法，将文本异步嵌入为向量列表
        return await self._embeddings.aembed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text."""
        # 异步调用内部嵌入对象的aembed_query方法，将查询文本异步嵌入为向量
        return await self._embeddings.aembed_query(text)
```