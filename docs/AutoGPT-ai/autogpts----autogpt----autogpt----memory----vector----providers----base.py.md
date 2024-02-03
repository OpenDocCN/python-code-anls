# `.\AutoGPT\autogpts\autogpt\autogpt\memory\vector\providers\base.py`

```py
import abc
import functools
import logging
from typing import MutableSet, Sequence

import numpy as np

from autogpt.config.config import Config

from .. import MemoryItem, MemoryItemRelevance
from ..utils import Embedding, get_embedding

logger = logging.getLogger(__name__)

# 定义一个名为VectorMemoryProvider的类，实现了MutableSet接口，用于管理MemoryItem对象集合
class VectorMemoryProvider(MutableSet[MemoryItem]):
    @abc.abstractmethod
    # 初始化方法，接受一个Config对象作为参数
    def __init__(self, config: Config):
        pass

    # 根据查询获取最相关的MemoryItem对象
    def get(self, query: str, config: Config) -> MemoryItemRelevance | None:
        """
        Gets the data from the memory that is most relevant to the given query.

        Args:
            query: The query used to retrieve information.
            config: The config Object.

        Returns: The most relevant Memory
        """
        result = self.get_relevant(query, 1, config)
        return result[0] if result else None

    # 获取与查询最相关的前k个MemoryItem对象
    def get_relevant(
        self, query: str, k: int, config: Config
    ) -> Sequence[MemoryItemRelevance]:
        """
        Returns the top-k most relevant memories for the given query

        Args:
            query: the query to compare stored memories to
            k: the number of relevant memories to fetch
            config: The config Object.

        Returns:
            list[MemoryItemRelevance] containing the top [k] relevant memories
        """
        if len(self) < 1:
            return []

        logger.debug(
            f"Searching for {k} relevant memories for query '{query}'; "
            f"{len(self)} memories in index"
        )

        # 计算MemoryItem对象与查询的相关性得分
        relevances = self.score_memories_for_relevance(query, config)
        logger.debug(f"Memory relevance scores: {[str(r) for r in relevances]}")

        # 取出得分最高的前k个MemoryItem对象的索引
        top_k_indices = np.argsort([r.score for r in relevances])[-k:][::-1]

        return [relevances[i] for i in top_k_indices]

    # 计算MemoryItem对象与查询的相关性得分
    def score_memories_for_relevance(
        self, for_query: str, config: Config
    # 返回每个内存项在索引中的相关性
    def get_relevances(self, for_query: Query, config: Config) -> Sequence[MemoryItemRelevance]:
        """
        Returns MemoryItemRelevance for every memory in the index.
        Implementations may override this function for performance purposes.
        """
        # 获取查询的嵌入向量
        e_query: Embedding = get_embedding(for_query, config)
        # 返回每个内存项对于查询的相关性
        return [m.relevance_for(for_query, e_query) for m in self]

    # 获取内存索引的统计信息
    def get_stats(self) -> tuple[int, int]:
        """
        Returns:
            tuple (n_memories: int, n_chunks: int): the stats of the memory index
        """
        # 返回内存索引中的内存项数量和所有内存块的总数
        return len(self), functools.reduce(lambda t, m: t + len(m.e_chunks), self, 0)
```