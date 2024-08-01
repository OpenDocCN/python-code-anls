# `.\DB-GPT-src\dbgpt\storage\graph_store\base.py`

```py
"""Graph store base class."""
import logging  # 导入日志模块，用于记录程序运行时的信息
from abc import ABC, abstractmethod  # 导入ABC和abstractmethod装饰器，用于定义抽象基类和抽象方法
from typing import List, Optional, Tuple  # 导入类型提示相关模块

from dbgpt._private.pydantic import BaseModel, ConfigDict, Field  # 导入pydantic相关模块
from dbgpt.core import Embeddings  # 导入Embeddings类
from dbgpt.storage.graph_store.graph import Direction, Graph  # 导入Direction和Graph类

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


class GraphStoreConfig(BaseModel):
    """Graph store config."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")  # 定义模型配置字典

    name: str = Field(
        default="dbgpt_collection",
        description="The name of graph store, inherit from index store.",
    )  # 图形存储的名称，默认为"dbgpt_collection"，继承自索引存储
    embedding_fn: Optional[Embeddings] = Field(
        default=None,
        description="The embedding function of graph store, optional.",
    )  # 图形存储的嵌入函数，可选项


class GraphStoreBase(ABC):
    """Graph store base class."""

    @abstractmethod
    def insert_triplet(self, sub: str, rel: str, obj: str):
        """Add triplet."""  # 添加三元组的抽象方法声明

    @abstractmethod
    def get_triplets(self, sub: str) -> List[Tuple[str, str]]:
        """Get triplets."""  # 获取三元组的抽象方法声明

    @abstractmethod
    def delete_triplet(self, sub: str, rel: str, obj: str):
        """Delete triplet."""  # 删除三元组的抽象方法声明

    @abstractmethod
    def drop(self):
        """Drop graph."""  # 删除图形的抽象方法声明

    @abstractmethod
    def get_schema(self, refresh: bool = False) -> str:
        """Get schema."""  # 获取模式的抽象方法声明

    @abstractmethod
    def get_full_graph(self, limit: Optional[int] = None) -> Graph:
        """Get full graph."""  # 获取完整图形的抽象方法声明

    @abstractmethod
    def explore(
        self,
        subs: List[str],
        direct: Direction = Direction.BOTH,
        depth: Optional[int] = None,
        fan: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Graph:
        """Explore on graph."""  # 在图形上进行探索的抽象方法声明

    @abstractmethod
    def query(self, query: str, **args) -> Graph:
        """Execute a query."""  # 执行查询的抽象方法声明
```