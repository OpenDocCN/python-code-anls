# `.\DB-GPT-src\dbgpt\storage\vector_store\pgvector_store.py`

```py
"""Postgres vector store."""

# 导入日志模块
import logging
# 导入类型提示模块
from typing import List, Optional

# 导入 Pydantic 相关模块
from dbgpt._private.pydantic import ConfigDict, Field
# 导入核心模块 Chunk
from dbgpt.core import Chunk
# 导入向量存储相关模块
from dbgpt.core.awel.flow import Parameter, ResourceCategory, register_resource
# 导入向量存储基类和配置类
from dbgpt.storage.vector_store.base import (
    _COMMON_PARAMETERS,
    VectorStoreBase,
    VectorStoreConfig,
)
# 导入元数据过滤器
from dbgpt.storage.vector_store.filters import MetadataFilters
# 导入国际化函数
from dbgpt.util.i18n_utils import _

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


# 注册 PG 向量存储资源
@register_resource(
    _("PG Vector Store"),
    "pg_vector_store",
    category=ResourceCategory.VECTOR_STORE,
    parameters=[
        *_COMMON_PARAMETERS,
        Parameter.build_from(
            _("Connection String"),
            "connection_string",
            str,
            description=_(
                "The connection string of vector store, if not set, will use "
                "the default connection string."
            ),
            optional=True,
            default=None,
        ),
    ],
    description="PG vector store.",
)
class PGVectorConfig(VectorStoreConfig):
    """PG vector store config."""

    # 允许任意类型的模型配置
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 连接字符串属性，用于连接向量存储
    connection_string: str = Field(
        default=None,
        description="the connection string of vector store, if not set, will use the "
        "default connection string.",
    )


class PGVectorStore(VectorStoreBase):
    """PG vector store.

    To use this, you should have the ``pgvector`` python package installed.
    """

    def __init__(self, vector_store_config: PGVectorConfig) -> None:
        """Create a PGVectorStore instance."""
        try:
            # 尝试导入 langchain 中的 PGVector 类
            from langchain.vectorstores import PGVector  # mypy: ignore
        except ImportError:
            # 若导入失败，抛出 ImportError 异常
            raise ImportError(
                "Please install the `langchain` package to use the PGVector."
            )
        # 调用父类的构造方法
        super().__init__()
        # 设置连接字符串属性
        self.connection_string = vector_store_config.connection_string
        # 设置嵌入函数属性
        self.embeddings = vector_store_config.embedding_fn
        # 设置集合名称属性
        self.collection_name = vector_store_config.name

        # 初始化 PGVector 客户端
        self.vector_store_client = PGVector(
            embedding_function=self.embeddings,  # type: ignore
            collection_name=self.collection_name,
            connection_string=self.connection_string,
        )

    def similar_search(
        self, text: str, topk: int, filters: Optional[MetadataFilters] = None
    ) -> List[Chunk]:
        """Perform similar search in PGVector."""
        # 调用 PGVector 客户端的相似搜索方法
        return self.vector_store_client.similarity_search(text, topk, filters)

    def vector_name_exists(self) -> bool:
        """Check if vector name exists."""
        try:
            # 尝试创建向量集合
            self.vector_store_client.create_collection()
            # 如果成功创建，返回 True
            return True
        except Exception as e:
            # 发生异常时记录错误日志，并返回 False
            logger.error(f"vector_name_exists error, {str(e)}")
            return False
    # 将文档块加载到 PGVector 中
    def load_document(self, chunks: List[Chunk]) -> List[str]:
        """Load document to PGVector.

        Args:
            chunks(List[Chunk]): document chunks.

        Return:
            List[str]: chunk ids.
        """
        # 转换每个文档块为语言链
        lc_documents = [Chunk.chunk2langchain(chunk) for chunk in chunks]
        # 使用向量存储客户端从语言链文档中加载向量
        self.vector_store_client.from_documents(lc_documents)  # type: ignore
        # 返回每个文档块的 ID 列表
        return [str(chunk.chunk_id) for chunk in lc_documents]

    # 根据向量名称删除向量
    def delete_vector_name(self, vector_name: str):
        """Delete vector by name.

        Args:
            vector_name(str): vector name.
        """
        # 调用向量存储客户端的删除集合方法
        return self.vector_store_client.delete_collection()

    # 根据 ID 列表删除向量
    def delete_by_ids(self, ids: str):
        """Delete vector by ids.

        Args:
            ids(str): vector ids, separated by comma.
        """
        # 将逗号分隔的 ID 字符串转换为 ID 列表
        delete_ids = ids.split(",")
        # 调用向量存储客户端的删除方法
        return self.vector_store_client.delete(delete_ids)
```