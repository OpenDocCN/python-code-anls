# `.\graphrag\graphrag\vector_stores\base.py`

```py
# 版权声明，声明版权归 Microsoft Corporation 所有，基于 MIT 许可证发布

# 导入必要的模块
from abc import ABC, abstractmethod  # 导入抽象基类和抽象方法装饰器
from dataclasses import dataclass, field  # 导入数据类装饰器和字段定义装饰器
from typing import Any  # 导入通用类型提示

from graphrag.model.types import TextEmbedder  # 导入文本嵌入器类型定义

DEFAULT_VECTOR_SIZE: int = 1536  # 默认向量大小设为 1536

@dataclass
class VectorStoreDocument:
    """存储在向量存储中的文档。"""

    id: str | int
    """文档的唯一标识符"""

    text: str | None
    """文档的文本内容，可以为空"""

    vector: list[float] | None
    """文档的向量表示，可以为空"""

    attributes: dict[str, Any] = field(default_factory=dict)
    """存储任意的额外元数据，如标题、日期范围等"""

@dataclass
class VectorStoreSearchResult:
    """向量存储的搜索结果。"""

    document: VectorStoreDocument
    """找到的文档"""

    score: float
    """相似度分数，范围在0到1之间，越高表示越相似"""

class BaseVectorStore(ABC):
    """向量存储数据访问类的基类。"""

    def __init__(
        self,
        collection_name: str,
        db_connection: Any | None = None,
        document_collection: Any | None = None,
        query_filter: Any | None = None,
        **kwargs: Any,
    ):
        self.collection_name = collection_name
        self.db_connection = db_connection
        self.document_collection = document_collection
        self.query_filter = query_filter
        self.kwargs = kwargs

    @abstractmethod
    def connect(self, **kwargs: Any) -> None:
        """连接到向量存储。"""

    @abstractmethod
    def load_documents(
        self, documents: list[VectorStoreDocument], overwrite: bool = True
    ) -> None:
        """将文档加载到向量存储中。"""

    @abstractmethod
    def similarity_search_by_vector(
        self, query_embedding: list[float], k: int = 10, **kwargs: Any
    ) -> list[VectorStoreSearchResult]:
        """根据向量执行近似最近邻搜索。"""

    @abstractmethod
    def similarity_search_by_text(
        self, text: str, text_embedder: TextEmbedder, k: int = 10, **kwargs: Any
    ) -> list[VectorStoreSearchResult]:
        """根据文本执行近似最近邻搜索。"""

    @abstractmethod
    def filter_by_id(self, include_ids: list[str] | list[int]) -> Any:
        """构建按ID过滤文档的查询过滤器。"""
```