# `.\DB-GPT-src\dbgpt\storage\full_text\base.py`

```py
"""Full text store base class."""
import logging  # 导入日志模块
from abc import abstractmethod  # 导入抽象方法模块
from concurrent.futures import Executor  # 导入Executor模块
from typing import List, Optional  # 导入类型提示模块

from dbgpt.core import Chunk  # 导入Chunk类
from dbgpt.rag.index.base import IndexStoreBase  # 导入IndexStoreBase类
from dbgpt.storage.vector_store.filters import MetadataFilters  # 导入MetadataFilters类
from dbgpt.util.executor_utils import blocking_func_to_async  # 导入blocking_func_to_async函数

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class FullTextStoreBase(IndexStoreBase):
    """Graph store base class."""

    def __init__(self, executor: Optional[Executor] = None):
        """Initialize vector store."""
        super().__init__(executor)  # 调用父类的初始化方法，设置executor参数

    @abstractmethod
    def load_document(self, chunks: List[Chunk]) -> List[str]:
        """Load document in index database.

        Args:
            chunks(List[Chunk]): document chunks.
        Return:
            List[str]: chunk ids.
        """
        # 抽象方法，子类需要实现从数据库加载文档的逻辑

    async def aload_document(self, chunks: List[Chunk]) -> List[str]:
        """Async load document in index database.

        Args:
            chunks(List[Chunk]): document chunks.
        Return:
            List[str]: chunk ids.
        """
        return await blocking_func_to_async(self._executor, self.load_document, chunks)
        # 使用异步方式加载文档到索引数据库，并返回文档的chunk ids

    @abstractmethod
    def similar_search_with_scores(
        self,
        text,
        topk,
        score_threshold: float,
        filters: Optional[MetadataFilters] = None,
    ) -> List[Chunk]:
        """Similar search with scores in index database.

        Args:
            text(str): The query text.
            topk(int): The number of similar documents to return.
            score_threshold(int): score_threshold: Optional, a floating point value
                between 0 to 1
            filters(Optional[MetadataFilters]): metadata filters.
        """
        # 抽象方法，实现在索引数据库中带有分数的相似搜索

    @abstractmethod
    def delete_by_ids(self, ids: str) -> List[str]:
        """Delete docs.

        Args:
            ids(str): The vector ids to delete, separated by comma.
        """
        # 抽象方法，删除指定id的文档

    def delete_vector_name(self, index_name: str):
        """Delete name."""
        # 删除指定名称的向量名称
```