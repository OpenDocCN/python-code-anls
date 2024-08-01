# `.\DB-GPT-src\dbgpt\storage\full_text\opensearch.py`

```py
"""OpenSearch index store."""
# 导入所需的模块和类
from typing import List, Optional

from dbgpt.core import Chunk  # 导入Chunk类
from dbgpt.rag.index.base import IndexStoreBase  # 导入IndexStoreBase类
from dbgpt.storage.vector_store.filters import MetadataFilters  # 导入MetadataFilters类


class OpenSearch(IndexStoreBase):
    """OpenSearch index store."""

    def load_document(self, chunks: List[Chunk]) -> List[str]:
        """Load document in index database.

        Args:
            chunks(List[Chunk]): document chunks.
        Return:
            List[str]: chunk ids.
        """
        pass  # 空函数，用于加载文档到索引数据库中

    def aload_document(self, chunks: List[Chunk]) -> List[str]:
        """Async load document in index database.

        Args:
            chunks(List[Chunk]): document chunks.
        Return:
            List[str]: chunk ids.
        """
        pass  # 异步加载文档到索引数据库中

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
            score_threshold(int): Optional, a floating point value between 0 to 1
            filters(Optional[MetadataFilters]): metadata filters.
        Return:
            List[Chunk]: The similar documents.
        """
        pass  # 在索引数据库中进行带分数的相似搜索

    def delete_by_ids(self, ids: str):
        """Delete docs.

        Args:
            ids(str): The vector ids to delete, separated by comma.
        """
        pass  # 根据给定的ID删除文档

    def delete_vector_name(self, index_name: str):
        """Delete name.

        Args:
            index_name(str): The name of the vector to delete.
        """
        pass  # 删除指定名称的向量
```