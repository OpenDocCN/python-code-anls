# `.\DB-GPT-src\dbgpt\storage\knowledge_graph\open_spg.py`

```py
"""OpenSPG class."""
import logging  # 导入日志模块
from typing import List, Optional  # 导入类型提示相关模块

from dbgpt._private.pydantic import ConfigDict  # 导入ConfigDict类
from dbgpt.core import Chunk  # 导入Chunk类
from dbgpt.storage.graph_store.graph import Graph, MemoryGraph  # 导入Graph和MemoryGraph类
from dbgpt.storage.knowledge_graph.base import KnowledgeGraphBase, KnowledgeGraphConfig  # 导入KnowledgeGraphBase和KnowledgeGraphConfig类
from dbgpt.storage.vector_store.filters import MetadataFilters  # 导入MetadataFilters类

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


class OpenSPGConfig(KnowledgeGraphConfig):
    """OpenSPG config."""

    model_config = ConfigDict(arbitrary_types_allowed=True)  # 定义model_config属性为ConfigDict对象


class OpenSPG(KnowledgeGraphBase):
    """OpenSPG class."""

    # todo: add OpenSPG implementation  # 待实现OpenSPG的功能

    def __init__(self, config: OpenSPGConfig):
        """Initialize the OpenSPG with config details."""
        pass  # 初始化方法，暂未实现具体功能

    def load_document(self, chunks: List[Chunk]) -> List[str]:
        """Load document."""
        return []  # 加载文档方法，返回空列表

    def similar_search_with_scores(
        self,
        text,
        topk,
        score_threshold: float,
        filters: Optional[MetadataFilters] = None,
    ) -> List[Chunk]:
        """Similar with scores."""
        return []  # 根据得分进行相似性搜索方法，返回空列表

    def query_graph(self, limit: Optional[int] = None) -> Graph:
        """Query graph."""
        return MemoryGraph()  # 查询图形结构方法，返回MemoryGraph对象

    def delete_vector_name(self, index_name: str):
        """Delete vector name."""
        pass  # 删除向量名称方法，暂未实现具体功能
```