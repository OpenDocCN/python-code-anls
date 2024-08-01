# `.\DB-GPT-src\dbgpt\storage\knowledge_graph\base.py`

```py
"""
Knowledge graph base class.
"""
# 引入日志模块，用于记录和输出程序运行时的信息
import logging
# 从abc模块中导入ABC抽象基类和abstractmethod装饰器，用于定义抽象方法
from abc import ABC, abstractmethod
# 从typing模块中导入List和Optional类型，用于类型提示
from typing import List, Optional

# 从dbgpt._private.pydantic中导入ConfigDict类
from dbgpt._private.pydantic import ConfigDict
# 从dbgpt.rag.index.base中导入IndexStoreBase和IndexStoreConfig类
from dbgpt.rag.index.base import IndexStoreBase, IndexStoreConfig
# 从dbgpt.storage.graph_store.graph中导入Graph类
from dbgpt.storage.graph_store.graph import Graph

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


class KnowledgeGraphConfig(IndexStoreConfig):
    """
    Knowledge graph config.

    Inherits from IndexStoreConfig for extended configuration options.
    """
    # 定义模型配置属性，允许任意类型，额外字段允许
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class KnowledgeGraphBase(IndexStoreBase, ABC):
    """
    Knowledge graph base class.

    Abstract base class for defining operations on a knowledge graph.
    """

    @abstractmethod
    def query_graph(self, limit: Optional[int] = None) -> Graph:
        """
        Abstract method to query the knowledge graph.

        Args:
            limit (Optional[int]): Maximum number of results to return.

        Returns:
            Graph: The queried graph data.
        """

    def delete_by_ids(self, ids: str) -> List[str]:
        """
        Delete document by ids.

        Args:
            ids (str): IDs of documents to delete.

        Raises:
            Exception: Always raises an exception as delete is not supported.

        Returns:
            List[str]: List of IDs that were intended to be deleted.
        """
        # 抛出异常，因为知识图谱不支持删除操作
        raise Exception("Delete document not supported by knowledge graph")
```