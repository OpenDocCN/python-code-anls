# `.\DB-GPT-src\dbgpt\storage\graph_store\factory.py`

```py
"""Connector for vector store."""
import logging  # 导入日志模块
from typing import Tuple, Type  # 导入类型提示所需的类和函数

from dbgpt.storage import graph_store  # 导入图存储模块
from dbgpt.storage.graph_store.base import GraphStoreBase, GraphStoreConfig  # 导入图存储基类和配置类

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class GraphStoreFactory:
    """Factory for graph store."""

    @staticmethod
    def create(graph_store_type: str, graph_store_configure=None) -> GraphStoreBase:
        """Create a GraphStore instance.

        Args:
            - graph_store_type: graph store type Memory, TuGraph, Neo4j
            - graph_store_config: graph store config
        """
        store_cls, cfg_cls = GraphStoreFactory.__find_type(graph_store_type)  # 根据传入的图存储类型查找对应的存储类和配置类

        try:
            config = cfg_cls()  # 创建配置类的实例
            if graph_store_configure:
                graph_store_configure(config)  # 如果传入了配置函数，则用其配置配置类实例
            return store_cls(config)  # 使用配置类实例化存储类，并返回存储类实例
        except Exception as e:
            logger.error("create graph store failed: %s", e)  # 记录创建图存储失败的错误信息
            raise e  # 抛出异常

    @staticmethod
    def __find_type(graph_store_type: str) -> Tuple[Type, Type]:
        """Find the appropriate store and config classes based on the provided graph_store_type.

        Args:
            - graph_store_type: The type of graph store to create (e.g., Memory, TuGraph, Neo4j)

        Returns:
            Tuple containing the store class and config class associated with the given type.

        Raises:
            Exception: If the graph store type is not supported.
        """
        for t in graph_store.__all__:  # 遍历所有图存储类型
            if t.lower() == graph_store_type.lower():  # 找到与传入的图存储类型相匹配的类名（不区分大小写）
                store_cls, cfg_cls = getattr(graph_store, t)  # 获取对应的存储类和配置类
                if issubclass(store_cls, GraphStoreBase) and issubclass(
                    cfg_cls, GraphStoreConfig
                ):  # 检查获取的类是否符合基础存储类和配置类的要求
                    return store_cls, cfg_cls  # 返回存储类和配置类

        raise Exception(f"Graph store {graph_store_type} not supported")  # 如果未找到匹配的图存储类型，则抛出异常
```