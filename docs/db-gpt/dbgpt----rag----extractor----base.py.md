# `.\DB-GPT-src\dbgpt\rag\extractor\base.py`

```py
"""Base Extractor Base class."""
# 导入必要的模块
from abc import ABC, abstractmethod
from typing import List

from dbgpt.core import Chunk, LLMClient

# 定义抽象基类 Extractor
class Extractor(ABC):
    """Base Extractor Base class.

    It's apply for Summary Extractor, Keyword Extractor, Triplets Extractor, Question
    Extractor, etc.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize the Extractor."""
        # 初始化 Extractor 类，传入 LLMClient 对象
        self._llm_client = llm_client

    def extract(self, chunks: List[Chunk]) -> str:
        """Return extracted metadata from chunks.

        Args:
            chunks (List[Chunk]): extract metadata from chunks
        """
        # 提取元数据信息并返回
        return self._extract(chunks)

    async def aextract(self, chunks: List[Chunk]) -> str:
        """Async Extracts chunks.

        Args:
            chunks (List[Chunk]): extract metadata from chunks
        """
        # 异步提取元数据信息并返回
        return await self._aextract(chunks)

    @abstractmethod
    def _extract(self, chunks: List[Chunk]) -> str:
        """Return extracted metadata from chunks.

        Args:
            chunks (List[Chunk]): extract metadata from chunks
        """

    @abstractmethod
    async def _aextract(self, chunks: List[Chunk]) -> str:
        """Async Extracts chunks.

        Args:
            chunks (List[Chunk]): extract metadata from chunks
        """
```