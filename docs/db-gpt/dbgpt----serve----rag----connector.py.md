# `.\DB-GPT-src\dbgpt\serve\rag\connector.py`

```py
"""Connector for vector store."""

import copy  # 导入复制模块，用于数据的深拷贝操作
import logging  # 导入日志模块，用于记录程序运行时的信息
import os  # 导入操作系统相关模块，用于处理文件路径等操作
from collections import defaultdict  # 导入默认字典模块，用于创建默认字典
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Type, cast  # 导入类型提示模块，用于静态类型检查

from dbgpt.core import Chunk, Embeddings  # 从dbgpt.core模块导入Chunk和Embeddings类
from dbgpt.core.awel.flow import (  # 从dbgpt.core.awel.flow模块导入以下类和函数
    FunctionDynamicOptions,
    OptionValue,
    Parameter,
    ResourceCategory,
    register_resource,
)
from dbgpt.rag.index.base import IndexStoreBase, IndexStoreConfig  # 从dbgpt.rag.index.base模块导入IndexStoreBase和IndexStoreConfig类
from dbgpt.storage.vector_store.base import VectorStoreConfig  # 从dbgpt.storage.vector_store.base模块导入VectorStoreConfig类
from dbgpt.storage.vector_store.filters import MetadataFilters  # 从dbgpt.storage.vector_store.filters模块导入MetadataFilters类
from dbgpt.util.i18n_utils import _  # 导入国际化函数，用于文本国际化

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象

connector: Dict[str, Tuple[Type, Type]] = {}  # 定义一个空的字典connector，用于存储类型和元组信息
pools: DefaultDict[str, Dict] = defaultdict(dict)  # 定义一个默认字典pools，用于存储字符串到字典的映射关系


def _load_vector_options() -> List[OptionValue]:
    """加载向量存储选项的函数"""
    from dbgpt.storage import vector_store  # 导入向量存储模块

    return [
        OptionValue(label=cls, name=cls, value=cls)
        for cls in vector_store.__all__  # 遍历vector_store模块中的所有公开对象
        if issubclass(getattr(vector_store, cls)[0], IndexStoreBase)  # 如果对象是IndexStoreBase的子类
    ]


@register_resource(
    _("Vector Store Connector"),  # 资源的显示名称
    "vector_store_connector",  # 资源的唯一标识符
    category=ResourceCategory.VECTOR_STORE,  # 资源的类别
    parameters=[  # 资源的参数列表
        Parameter.build_from(
            _("Vector Store Type"),  # 参数的显示名称
            "vector_store_type",  # 参数的唯一标识符
            str,  # 参数的类型
            description=_("The type of vector store."),  # 参数的描述信息
            options=FunctionDynamicOptions(func=_load_vector_options),  # 参数的选项，通过函数动态加载
        ),
        Parameter.build_from(
            _("Vector Store Implementation"),  # 参数的显示名称
            "vector_store_config",  # 参数的唯一标识符
            VectorStoreConfig,  # 参数的类型
            description=_("The vector store implementation."),  # 参数的描述信息
            optional=True,  # 参数是否可选
            default=None,  # 参数的默认值
        ),
    ],
    alias=["dbgpt.storage.vector_store.connector.VectorStoreConnector"],  # 资源的别名列表
)
class VectorStoreConnector:
    """向量存储连接器类"""

    def __init__(
        self,
        vector_store_type: str,  # 向量存储的类型
        vector_store_config: Optional[IndexStoreConfig] = None,  # 向量存储的配置信息，默认为None
    ) -> None:
        """Create a VectorStoreConnector instance.

        Args:
            - vector_store_type: vector store type Milvus, Chroma, Weaviate
            - ctx: vector store config params.
        """
        # 检查 vector_store_config 是否为空，若为空则抛出异常
        if vector_store_config is None:
            raise Exception("vector_store_config is required")

        # 将传入的 vector_store_config 赋值给实例变量 _index_store_config
        self._index_store_config = vector_store_config
        # 调用 _register 方法进行注册
        self._register()

        # 根据 vector_store_type 确定连接器类和配置类
        if self._match(vector_store_type):
            self.connector_class, self.config_class = connector[vector_store_type]
        else:
            # 若 vector_store_type 不被支持，则抛出异常
            raise Exception(f"Vector store {vector_store_type} not supported")

        # 记录日志，显示所选用的连接器类
        logger.info(f"VectorStore:{self.connector_class}")

        # 将 vector_store_type 和 embedding_fn 赋值给实例变量 _vector_store_type 和 _embeddings
        self._vector_store_type = vector_store_type
        self._embeddings = vector_store_config.embedding_fn

        # 创建配置字典，从 vector_store_config 中获取相关配置信息
        config_dict = {}
        for key in vector_store_config.to_dict().keys():
            value = getattr(vector_store_config, key)
            if value is not None:
                config_dict[key] = value
        # 将 model_extra 中的额外配置项加入配置字典
        for key, value in vector_store_config.model_extra.items():
            if value is not None:
                config_dict[key] = value

        # 使用配置字典创建配置对象 config
        config = self.config_class(**config_dict)
        try:
            # 尝试从连接池中获取或创建与 vector store 相关的客户端对象 self.client
            if vector_store_type in pools and config.name in pools[vector_store_type]:
                self.client = pools[vector_store_type][config.name]
            else:
                client = self.connector_class(config)
                pools[vector_store_type][config.name] = self.client = client
        except Exception as e:
            # 连接失败时记录错误日志并抛出异常
            logger.error("connect vector store failed: %s", e)
            raise e

    @classmethod
    def from_default(
        cls,
        vector_store_type: Optional[str] = None,
        embedding_fn: Optional[Any] = None,
        vector_store_config: Optional[VectorStoreConfig] = None,
    ) -> "VectorStoreConnector":
        """Initialize default vector store connector."""
        # 获取环境变量中的 VECTOR_STORE_TYPE，若未设置则默认为 "Chroma"
        vector_store_type = vector_store_type or os.getenv(
            "VECTOR_STORE_TYPE", "Chroma"
        )
        # 导入 ChromaVectorConfig 类并创建默认的 vector_store_config 对象
        from dbgpt.storage.vector_store.chroma_store import ChromaVectorConfig

        vector_store_config = vector_store_config or ChromaVectorConfig()
        # 将传入的 embedding_fn 赋值给 vector_store_config 对象的 embedding_fn 属性
        vector_store_config.embedding_fn = embedding_fn
        # 将 vector_store_type 转换为 str 类型
        real_vector_store_type = cast(str, vector_store_type)
        # 返回使用指定参数初始化的 VectorStoreConnector 类的实例
        return cls(real_vector_store_type, vector_store_config)

    @property
    def index_client(self):
        # 返回当前实例的 client 属性作为 index_client 属性
        return self.client
    def load_document(self, chunks: List[Chunk]) -> List[str]:
        """Load document in vector database.

        Args:
            - chunks: document chunks.
        Return chunk ids.
        """
        # 获取每次加载的最大文档块数，如果未配置则默认为 10
        max_chunks_once_load = (
            self._index_store_config.max_chunks_once_load
            if self._index_store_config
            else 10
        )
        # 获取并发线程数的最大限制，如果未配置则默认为 1
        max_threads = (
            self._index_store_config.max_threads if self._index_store_config else 1
        )
        # 调用客户端的加载文档方法，限制每次加载的文档块数和并发线程数
        return self.client.load_document_with_limit(
            chunks,
            max_chunks_once_load,
            max_threads,
        )

    async def aload_document(self, chunks: List[Chunk]) -> List[str]:
        """Async load document in vector database.

        Args:
            - chunks: document chunks.
        Return chunk ids.
        """
        # 获取每次加载的最大文档块数，如果未配置则默认为 10
        max_chunks_once_load = (
            self._index_store_config.max_chunks_once_load
            if self._index_store_config
            else 10
        )
        # 获取并发线程数的最大限制，如果未配置则默认为 1
        max_threads = (
            self._index_store_config.max_threads if self._index_store_config else 1
        )
        # 异步调用客户端的加载文档方法，限制每次加载的文档块数和并发线程数
        return await self.client.aload_document_with_limit(
            chunks, max_chunks_once_load, max_threads
        )

    def similar_search(
        self, doc: str, topk: int, filters: Optional[MetadataFilters] = None
    ) -> List[Chunk]:
        """Similar search in vector database.

        Args:
           - doc: query text
           - topk: topk
           - filters: metadata filters.
        Return:
            - chunks: chunks.
        """
        # 调用客户端的相似搜索方法，传入查询文本、返回结果的数量和可选的元数据过滤器
        return self.client.similar_search(doc, topk, filters)

    def similar_search_with_scores(
        self,
        doc: str,
        topk: int,
        score_threshold: float,
        filters: Optional[MetadataFilters] = None,
    ) -> List[Chunk]:
        """Similar_search_with_score in vector database.

        Return docs and relevance scores in the range [0, 1].

        Args:
            doc(str): query text
            topk(int): return docs nums. Defaults to 4.
            score_threshold(float): score_threshold: Optional, a floating point value
                between 0 to 1 to filter the resulting set of retrieved docs,0 is
                dissimilar, 1 is most similar.
            filters: metadata filters.
        Return:
            - chunks: Return docs and relevance scores in the range [0, 1].
        """
        # 调用客户端的带分数的相似搜索方法，传入查询文本、返回结果的数量、分数阈值和可选的元数据过滤器
        return self.client.similar_search_with_scores(
            doc, topk, score_threshold, filters
        )

    async def asimilar_search_with_scores(
        self,
        doc: str,
        topk: int,
        score_threshold: float,
        filters: Optional[MetadataFilters] = None,
    ) -> List[Chunk]:
        """Async similar_search_with_score in vector database."""
        # 异步调用客户端的带分数的相似搜索方法，传入查询文本、返回结果的数量、分数阈值和可选的元数据过滤器
        return await self.client.asimilar_search_with_scores(
            doc, topk, score_threshold, filters
        )

    @property
    # 返回当前向量存储配置对象
    def vector_store_config(self) -> IndexStoreConfig:
        """Return the vector store config."""
        # 如果向量存储配置对象未设置，则引发数值错误异常
        if not self._index_store_config:
            raise ValueError("vector store config not set.")
        # 返回向量存储配置对象
        return self._index_store_config

    # 检查向量名称是否存在
    def vector_name_exists(self):
        """Whether vector name exists."""
        # 调用客户端方法检查向量名称是否存在
        return self.client.vector_name_exists()

    # 删除指定的向量名称
    def delete_vector_name(self, vector_name: str):
        """Delete vector name.

        Args:
            - vector_name: vector store name
        """
        try:
            # 如果向量名称存在，则调用客户端方法删除它
            if self.vector_name_exists():
                self.client.delete_vector_name(vector_name)
        except Exception as e:
            # 记录删除向量名称失败的错误信息
            logger.error(f"delete vector name {vector_name} failed: {e}")
            # 抛出异常指示删除名称失败
            raise Exception(f"delete name {vector_name} failed")
        # 操作成功返回True
        return True

    # 根据IDs删除向量
    def delete_by_ids(self, ids):
        """Delete vector by ids.

        Args:
            - ids: vector ids
        """
        # 调用客户端方法删除指定IDs的向量
        return self.client.delete_by_ids(ids=ids)

    # 返回当前嵌入（向量）对象，可选类型为Embeddings或None
    @property
    def current_embeddings(self) -> Optional[Embeddings]:
        """Return the current embeddings."""
        # 返回当前嵌入对象
        return self._embeddings

    # 创建一个新的连接器对象
    def new_connector(self, name: str, **kwargs) -> "VectorStoreConnector":
        """Create a new connector.

        New connector based on the current connector.
        """
        # 复制当前的向量存储配置对象
        config = copy.copy(self.vector_store_config)
        # 设置附加参数的属性到配置对象中
        for k, v in kwargs.items():
            if v is not None:
                setattr(config, k, v)
        # 设置连接器名称
        config.name = name

        # 返回一个基于当前连接器类型和配置的新连接器对象
        return self.__class__(self._vector_store_type, config)

    # 检查给定的向量存储类型是否有匹配的连接器
    def _match(self, vector_store_type) -> bool:
        # 返回是否存在给定向量存储类型的连接器
        return bool(connector.get(vector_store_type))

    # 注册向量存储类型
    def _register(self):
        # 导入向量存储模块
        from dbgpt.storage import vector_store

        # 遍历所有导出的向量存储类
        for cls in vector_store.__all__:
            # 获取存储类和配置类
            store_cls, config_cls = getattr(vector_store, cls)
            # 如果存储类是IndexStoreBase的子类且配置类是IndexStoreConfig的子类
            if issubclass(store_cls, IndexStoreBase) and issubclass(
                config_cls, IndexStoreConfig
            ):
                # 将存储类和配置类注册到连接器中
                connector[cls] = (store_cls, config_cls)
```