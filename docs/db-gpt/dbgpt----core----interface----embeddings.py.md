# `.\DB-GPT-src\dbgpt\core\interface\embeddings.py`

```py
"""Interface for embedding models."""

import asyncio  # 引入异步处理模块
from abc import ABC, abstractmethod  # 引入抽象基类和抽象方法的装饰器
from typing import List  # 引入类型提示


class RerankEmbeddings(ABC):
    """Interface for rerank models."""

    @abstractmethod
    def predict(self, query: str, candidates: List[str]) -> List[float]:
        """Predict the scores of the candidates."""

    async def apredict(self, query: str, candidates: List[str]) -> List[float]:
        """Asynchronously predict the scores of the candidates."""
        return await asyncio.get_running_loop().run_in_executor(
            None, self.predict, query, candidates
        )


class Embeddings(ABC):
    """Interface for embedding models.

    Refer to `Langchain Embeddings <https://github.com/langchain-ai/langchain/tree/
    master/libs/langchain/langchain/embeddings>`_.
    """

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs."""
        return await asyncio.get_running_loop().run_in_executor(
            None, self.embed_documents, texts
        )

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text."""
        return await asyncio.get_running_loop().run_in_executor(
            None, self.embed_query, text
        )
```