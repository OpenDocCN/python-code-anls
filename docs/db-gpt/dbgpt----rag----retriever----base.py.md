# `.\DB-GPT-src\dbgpt\rag\retriever\base.py`

```py
"""Base retriever module."""
from abc import ABC, abstractmethod   # 导入ABC抽象基类和abstractmethod装饰器
from enum import Enum   # 导入枚举类型Enum
from typing import Any, Dict, List, Optional   # 导入类型提示相关的模块

from dbgpt.core import Chunk   # 从dbgpt.core模块导入Chunk类
from dbgpt.storage.vector_store.filters import MetadataFilters   # 从dbgpt.storage.vector_store.filters导入MetadataFilters类


class RetrieverStrategy(str, Enum):
    """Retriever strategy.

    Args:
        - EMBEDDING: embedding retriever
        - KEYWORD: keyword retriever
        - HYBRID: hybrid retriever
    """
    
    EMBEDDING = "embedding"   # 枚举项：嵌入检索器
    GRAPH = "graph"   # 枚举项：图检索器
    KEYWORD = "keyword"   # 枚举项：关键词检索器
    HYBRID = "hybrid"   # 枚举项：混合检索器


class BaseRetriever(ABC):
    """Base retriever."""

    def load_document(self, chunks: List[Chunk], **kwargs: Dict[str, Any]) -> List[str]:
        """Load document in vector database.

        Args:
            - chunks: document chunks.
        Return chunk ids.
        """
        raise NotImplementedError   # 抽象方法，子类需实现具体逻辑

    def retrieve(
        self, query: str, filters: Optional[MetadataFilters] = None
    ) -> List[Chunk]:
        """Retrieve knowledge chunks.

        Args:
            query (str): query text.
            filters: (Optional[MetadataFilters]) metadata filters.

        Returns:
            List[Chunk]: list of chunks
        """
        return self._retrieve(query, filters)   # 调用具体实现的检索方法

    async def aretrieve(
        self, query: str, filters: Optional[MetadataFilters] = None
    ) -> List[Chunk]:
        """Retrieve knowledge chunks asynchronously.

        Args:
            query (str): async query text.
            filters: (Optional[MetadataFilters]) metadata filters.

        Returns:
            List[Chunk]: list of chunks
        """
        return await self._aretrieve(query, filters)   # 异步调用具体实现的检索方法

    def retrieve_with_scores(
        self,
        query: str,
        score_threshold: float,
        filters: Optional[MetadataFilters] = None,
    ) -> List[Chunk]:
        """Retrieve knowledge chunks with score.

        Args:
            query (str): query text.
            score_threshold (float): score threshold.
            filters: (Optional[MetadataFilters]) metadata filters.

        Returns:
            List[Chunk]: list of chunks
        """
        return self._retrieve_with_score(query, score_threshold, filters)   # 调用具体实现的带分数的检索方法

    async def aretrieve_with_scores(
        self,
        query: str,
        score_threshold: float,
        filters: Optional[MetadataFilters] = None,
    ) -> List[Chunk]:
        """Retrieve knowledge chunks with score asynchronously.

        Args:
            query (str): query text
            score_threshold (float): score threshold
            filters: (Optional[MetadataFilters]) metadata filters.

        Returns:
            List[Chunk]: list of chunks
        """
        return await self._aretrieve_with_score(query, score_threshold, filters)   # 异步调用具体实现的带分数的检索方法

    @abstractmethod
    def _retrieve(
        self, query: str, filters: Optional[MetadataFilters] = None
    ) -> List[Chunk]:
        """Abstract method to retrieve knowledge chunks.

        Args:
            query (str): query text.
            filters: (Optional[MetadataFilters]) metadata filters.

        Returns:
            List[Chunk]: list of chunks
        """
        ...
    ) -> List[Chunk]:
        """
        Retrieve knowledge chunks.

        Args:
            query (str): query text
            filters: (Optional[MetadataFilters]) metadata filters.

        Returns:
            List[Chunk]: list of chunks
        """
        
    @abstractmethod
    async def _aretrieve(
        self, query: str, filters: Optional[MetadataFilters] = None
    ) -> List[Chunk]:
        """
        Async Retrieve knowledge chunks.

        Args:
            query (str): query text
            filters: (Optional[MetadataFilters]) metadata filters.

        Returns:
            List[Chunk]: list of chunks
        """

    @abstractmethod
    def _retrieve_with_score(
        self,
        query: str,
        score_threshold: float,
        filters: Optional[MetadataFilters] = None,
    ) -> List[Chunk]:
        """
        Retrieve knowledge chunks with score.

        Args:
            query (str): query text
            score_threshold (float): score threshold
            filters: (Optional[MetadataFilters]) metadata filters.

        Returns:
            List[Chunk]: list of chunks
        """

    @abstractmethod
    async def _aretrieve_with_score(
        self,
        query: str,
        score_threshold: float,
        filters: Optional[MetadataFilters] = None,
    ) -> List[Chunk]:
        """
        Async Retrieve knowledge chunks with score.

        Args:
            query (str): query text
            score_threshold (float): score threshold
            filters: (Optional[MetadataFilters]) metadata filters.

        Returns:
            List[Chunk]: list of chunks
        """
```