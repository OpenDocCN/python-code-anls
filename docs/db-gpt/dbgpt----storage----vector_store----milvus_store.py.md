# `.\DB-GPT-src\dbgpt\storage\vector_store\milvus_store.py`

```py
"""Milvus vector store."""  # 文件的顶层注释，说明这个文件是关于Milvus向量存储的
from __future__ import annotations  # 导入未来版本的注解支持，使得类型注解中可以引用类本身

import json  # 导入处理JSON的模块
import logging  # 导入日志记录模块
import os  # 导入操作系统功能的模块
from typing import Any, Iterable, List, Optional  # 导入类型提示模块

from dbgpt._private.pydantic import ConfigDict, Field  # 导入Pydantic相关模块
from dbgpt.core import Chunk, Embeddings  # 导入核心模块中的Chunk和Embeddings类
from dbgpt.core.awel.flow import Parameter, ResourceCategory, register_resource  # 导入awel.flow中的类和函数
from dbgpt.storage.vector_store.base import (  # 导入向量存储基础类和配置类
    _COMMON_PARAMETERS,
    VectorStoreBase,
    VectorStoreConfig,
)
from dbgpt.storage.vector_store.filters import FilterOperator, MetadataFilters  # 导入向量存储的过滤器类
from dbgpt.util import string_utils  # 导入字符串处理工具类
from dbgpt.util.i18n_utils import _  # 导入国际化翻译函数_

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


@register_resource(  # 注册Milvus向量存储作为一个资源
    _("Milvus Vector Store"),  # 资源的显示名称，国际化翻译为Milvus向量存储
    "milvus_vector_store",  # 资源的唯一标识符
    category=ResourceCategory.VECTOR_STORE,  # 资源类别为向量存储
    parameters=[  # 资源的参数列表
        *_COMMON_PARAMETERS,  # 引入共享的参数列表
        Parameter.build_from(  # 构建参数对象
            _("Uri"),  # 参数名称，国际化翻译为Uri
            "uri",  # 参数标识符
            str,  # 参数类型为字符串
            description=_(
                "The uri of milvus store, if not set, will use the default " "uri."
            ),  # 参数的描述信息，国际化翻译为Milvus存储的URI，如果未设置将使用默认的URI
            optional=True,  # 参数为可选
            default="localhost",  # 默认值为localhost
        ),
        Parameter.build_from(  # 构建参数对象
            _("Port"),  # 参数名称，国际化翻译为Port
            "port",  # 参数标识符
            str,  # 参数类型为字符串
            description=_(
                "The port of milvus store, if not set, will use the default " "port."
            ),  # 参数的描述信息，国际化翻译为Milvus存储的端口，如果未设置将使用默认的端口
            optional=True,  # 参数为可选
            default="19530",  # 默认值为19530
        ),
        Parameter.build_from(  # 构建参数对象
            _("Alias"),  # 参数名称，国际化翻译为Alias
            "alias",  # 参数标识符
            str,  # 参数类型为字符串
            description=_(
                "The alias of milvus store, if not set, will use the default " "alias."
            ),  # 参数的描述信息，国际化翻译为Milvus存储的别名，如果未设置将使用默认的别名
            optional=True,  # 参数为可选
            default="default",  # 默认值为default
        ),
        Parameter.build_from(  # 构建参数对象
            _("Primary Field"),  # 参数名称，国际化翻译为Primary Field
            "primary_field",  # 参数标识符
            str,  # 参数类型为字符串
            description=_(
                "The primary field of milvus store, if not set, will use the "
                "default primary field."
            ),  # 参数的描述信息，国际化翻译为Milvus存储的主字段，如果未设置将使用默认的主字段
            optional=True,  # 参数为可选
            default="pk_id",  # 默认值为pk_id
        ),
        Parameter.build_from(  # 构建参数对象
            _("Text Field"),  # 参数名称，国际化翻译为Text Field
            "text_field",  # 参数标识符
            str,  # 参数类型为字符串
            description=_(
                "The text field of milvus store, if not set, will use the "
                "default text field."
            ),  # 参数的描述信息，国际化翻译为Milvus存储的文本字段，如果未设置将使用默认的文本字段
            optional=True,  # 参数为可选
            default="content",  # 默认值为content
        ),
        Parameter.build_from(  # 构建参数对象
            _("Embedding Field"),  # 参数名称，国际化翻译为Embedding Field
            "embedding_field",  # 参数标识符
            str,  # 参数类型为字符串
            description=_(
                "The embedding field of milvus store, if not set, will use the "
                "default embedding field."
            ),  # 参数的描述信息，国际化翻译为Milvus存储的嵌入字段，如果未设置将使用默认的嵌入字段
            optional=True,  # 参数为可选
            default="vector",  # 默认值为vector
        ),
    ],
    description=_("Milvus vector store."),  # 资源的描述信息，国际化翻译为Milvus向量存储
)
class MilvusVectorConfig(VectorStoreConfig):
    """Milvus vector store config."""  # Milvus向量存储配置类的文档字符串
    model_config = ConfigDict(arbitrary_types_allowed=True)  # 配置类的属性model_config，类型为ConfigDict，允许任意类型
    # URI（统一资源标识符）的配置，用于指定 Milvus 存储的地址，默认为 localhost。
    uri: str = Field(
        default="localhost",
        description="The uri of milvus store, if not set, will use the default uri.",
    )
    
    # 端口号的配置，用于指定 Milvus 存储的端口，默认为 19530。
    port: str = Field(
        default="19530",
        description="The port of milvus store, if not set, will use the default port.",
    )
    
    # 别名的配置，用于指定 Milvus 存储的别名，默认为 "default"。
    alias: str = Field(
        default="default",
        description="The alias of milvus store, if not set, will use the default "
        "alias.",
    )
    
    # 主字段的配置，用于指定 Milvus 存储中的主字段，默认为 "pk_id"。
    primary_field: str = Field(
        default="pk_id",
        description="The primary field of milvus store, if not set, will use the "
        "default primary field.",
    )
    
    # 文本字段的配置，用于指定 Milvus 存储中的文本字段，默认为 "content"。
    text_field: str = Field(
        default="content",
        description="The text field of milvus store, if not set, will use the default "
        "text field.",
    )
    
    # 嵌入字段的配置，用于指定 Milvus 存储中的嵌入向量字段，默认为 "vector"。
    embedding_field: str = Field(
        default="vector",
        description="The embedding field of milvus store, if not set, will use the "
        "default embedding field.",
    )
    
    # 元数据字段的配置，用于指定 Milvus 存储中的元数据字段，默认为 "metadata"。
    metadata_field: str = Field(
        default="metadata",
        description="The metadata field of milvus store, if not set, will use the "
        "default metadata field.",
    )
    
    # 安全设置的配置，用于指定 Milvus 存储的安全选项，默认为空字符串。
    secure: str = Field(
        default="",
        description="The secure of milvus store, if not set, will use the default "
        "secure.",
    )
    def similar_search(
        self, text, topk, filters: Optional[MetadataFilters] = None
    ) -> List[dict]:
        """Perform similar search in Milvus."""
        # Perform vector embedding for the input text
        text_vector = self.embedding.embed_query(text)
        # Perform similarity search using Milvus
        results = self.col.query(text_vector, topk=topk, filters=filters)
        return results
    ) -> List[Chunk]:
        """Perform a search on a query string and return results."""
        # 尝试导入 pymilvus 库的 Collection 和 DataType
        try:
            from pymilvus import Collection, DataType
        except ImportError:
            # 如果导入失败，抛出 ValueError 异常
            raise ValueError(
                "Could not import pymilvus python package. "
                "Please install it with `pip install pymilvus`."
            )
        """similar_search in vector database."""
        # 使用给定的集合名称创建 Collection 对象
        self.col = Collection(self.collection_name)
        # 获取集合的 schema（模式）
        schema = self.col.schema
        # 遍历 schema 中的字段
        for x in schema.fields:
            # 将字段名添加到 self.fields 列表中
            self.fields.append(x.name)
            # 如果字段具有自动 ID，则从 self.fields 列表中移除该字段名
            if x.auto_id:
                self.fields.remove(x.name)
            # 如果字段是主键，则将该字段名设置为 self.primary_field
            if x.is_primary:
                self.primary_field = x.name
            # 如果字段类型是 FLOAT_VECTOR 或 BINARY_VECTOR，则将该字段名设置为 self.vector_field
            if x.dtype == DataType.FLOAT_VECTOR or x.dtype == DataType.BINARY_VECTOR:
                self.vector_field = x.name
        # 将 filters 转换为 milvus 表达式过滤器
        milvus_filter_expr = self.convert_metadata_filters(filters) if filters else None
        # 调用 _search 方法执行搜索，返回搜索结果
        _, docs_and_scores = self._search(text, topk, expr=milvus_filter_expr)

        # 根据搜索结果，返回 Chunk 对象列表
        return [
            Chunk(
                metadata=json.loads(doc.metadata.get("metadata", "")),
                content=doc.content,
            )
            for doc, _, _ in docs_and_scores
        ]

    def similar_search_with_scores(
        self,
        text: str,
        topk: int,
        score_threshold: float,
        filters: Optional[MetadataFilters] = None,
    ) -> List[Chunk]:
        """Perform a search on a query string and return results with score.

        For more information about the search parameters, take a look at the pymilvus
        documentation found here:
        https://milvus.io/api-reference/pymilvus/v2.2.6/Collection/search().md

        Args:
            text (str): The query text.
            topk (int): The number of similar documents to return.
            score_threshold (float): Optional, a floating point value between 0 to 1.
            filters (Optional[MetadataFilters]): Optional, metadata filters.
        Returns:
            List[Tuple[Document, float]]: Result doc and score.
        """
        try:
            from pymilvus import Collection, DataType
        except ImportError:
            raise ValueError(
                "Could not import pymilvus python package. "
                "Please install it with `pip install pymilvus`."
            )

        # 创建 Collection 对象，并设置为类的属性
        self.col = Collection(self.collection_name)
        # 获取 Collection 的 schema
        schema = self.col.schema
        # 遍历 schema 中的字段
        for x in schema.fields:
            # 将字段名添加到类的 fields 属性中
            self.fields.append(x.name)
            # 如果字段是自动生成的 ID 字段，则从 fields 属性中移除
            if x.auto_id:
                self.fields.remove(x.name)
            # 如果字段是主键字段，则设置主键字段名为 primary_field
            if x.is_primary:
                self.primary_field = x.name

            # 如果字段是向量类型（FLOAT_VECTOR 或 BINARY_VECTOR），则设置向量字段名为 vector_field
            if x.dtype == DataType.FLOAT_VECTOR or x.dtype == DataType.BINARY_VECTOR:
                self.vector_field = x.name

        # 将元数据过滤器转换为 milvus 的表达式过滤器
        milvus_filter_expr = self.convert_metadata_filters(filters) if filters else None
        # 调用内部 _search 方法执行查询
        _, docs_and_scores = self._search(
            query=text, topk=topk, expr=milvus_filter_expr
        )

        # 检查是否存在不合法的相似度分数，并记录警告日志
        if any(score < 0.0 or score > 1.0 for _, score, id in docs_and_scores):
            logger.warning(
                "similarity score need between" f" 0 and 1, got {docs_and_scores}"
            )

        # 如果指定了 score_threshold，则根据阈值筛选文档
        if score_threshold is not None:
            docs_and_scores = [
                Chunk(
                    metadata=doc.metadata,
                    content=doc.content,
                    score=score,
                    chunk_id=str(id),
                )
                for doc, score, id in docs_and_scores
                if score >= score_threshold
            ]
            # 如果未检索到符合阈值的文档，则记录警告日志
            if len(docs_and_scores) == 0:
                logger.warning(
                    "No relevant docs were retrieved using the relevance score"
                    f" threshold {score_threshold}"
                )
        
        # 返回最终的文档和分数列表
        return docs_and_scores

    def _search(
        self,
        query: str,
        k: int = 4,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        partition_names: Optional[List[str]] = None,
        round_decimal: int = -1,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ):
        """Search in vector database.

        Args:
            query: query text.
            k: topk.
            param: search params.
            expr: search expr.
            partition_names: partition names.
            round_decimal: round decimal.
            timeout: timeout.
            **kwargs: kwargs.
        Returns:
            Tuple[Document, float, int]: Result doc and score.
        """
        self.col.load()
        # 如果未指定搜索参数，则使用默认索引类型的参数
        if param is None:
            index_type = self.col.indexes[0].params["index_type"]
            param = self.index_params_map[index_type].get("params")
        
        # 将查询文本嵌入到向量空间中
        query_vector = self.embedding.embed_query(query)
        
        # 确定结果的元数据字段
        output_fields = self.fields[:]
        output_fields.remove(self.vector_field)
        
        # 在Milvus中进行搜索
        res = self.col.search(
            [query_vector],
            self.vector_field,
            param,
            k,
            expr=expr,
            output_fields=output_fields,
            partition_names=partition_names,
            round_decimal=round_decimal,
            timeout=60,
            **kwargs,
        )
        
        # 构建返回结果列表
        ret = []
        for result in res[0]:
            meta = {x: result.entity.get(x) for x in output_fields}
            ret.append(
                (
                    Chunk(content=meta.pop(self.text_field), metadata=meta),
                    result.distance,
                    result.id,
                )
            )
        
        # 如果未检索到相关文档，记录警告并返回空结果
        if len(ret) == 0:
            logger.warning("No relevant docs were retrieved.")
            return None, []
        
        # 返回第一个结果文档及其相关信息
        return ret[0], ret

    def vector_name_exists(self):
        """Whether vector name exists."""
        try:
            from pymilvus import utility
        except ImportError:
            raise ValueError(
                "Could not import pymilvus python package. "
                "Please install it with `pip install pymilvus`."
            )
        
        # 检查向量集合是否存在
        return utility.has_collection(self.collection_name)

    def delete_vector_name(self, vector_name: str):
        """Delete vector name."""
        try:
            from pymilvus import utility
        except ImportError:
            raise ValueError(
                "Could not import pymilvus python package. "
                "Please install it with `pip install pymilvus`."
            )
        
        # 删除指定的向量集合
        logger.info(f"milvus vector_name:{vector_name} begin delete...")
        utility.drop_collection(self.collection_name)
        return True
    def delete_by_ids(self, ids):
        """Delete vector by ids."""
        try:
            from pymilvus import Collection  # 导入 pymilvus 的 Collection 类
        except ImportError:
            raise ValueError(
                "Could not import pymilvus python package. "
                "Please install it with `pip install pymilvus`."
            )
        self.col = Collection(self.collection_name)  # 创建 Collection 对象
        # milvus 根据 ids 删除向量
        logger.info(f"begin delete milvus ids: {ids}")  # 记录日志，显示开始删除 milvus 中的 ids
        delete_ids = ids.split(",")  # 将 ids 字符串分割为列表
        doc_ids = [int(doc_id) for doc_id in delete_ids]  # 转换为整数类型的列表
        delete_expr = f"{self.primary_field} in {doc_ids}"  # 构造删除表达式
        self.col.delete(delete_expr)  # 在 Collection 中执行删除操作
        return True  # 返回操作成功标志

    def convert_metadata_filters(self, filters: MetadataFilters) -> str:
        """Convert filter to milvus filters.

        Args:
            - filters: metadata filters.
        Returns:
            - metadata_filters: metadata filters.
        """
        metadata_filters = []  # 初始化元数据过滤器列表
        for metadata_filter in filters.filters:  # 遍历输入的元数据过滤器
            if isinstance(metadata_filter.value, str):  # 如果值为字符串类型
                expr = (
                    f"{self.props_field}['{metadata_filter.key}'] "
                    f"{FilterOperator.EQ} '{metadata_filter.value}'"
                )  # 构造等于操作的表达式
                metadata_filters.append(expr)  # 添加到元数据过滤器列表
            elif isinstance(metadata_filter.value, List):  # 如果值为列表类型
                expr = (
                    f"{self.props_field}['{metadata_filter.key}'] "
                    f"{FilterOperator.IN} {metadata_filter.value}"
                )  # 构造包含操作的表达式
                metadata_filters.append(expr)  # 添加到元数据过滤器列表
            else:  # 其他类型的值
                expr = (
                    f"{self.props_field}['{metadata_filter.key}'] "
                    f"{FilterOperator.EQ} {str(metadata_filter.value)}"
                )  # 构造等于操作的表达式（将值转换为字符串）
                metadata_filters.append(expr)  # 添加到元数据过滤器列表
        if len(metadata_filters) > 1:  # 如果有多个过滤器
            metadata_filter_expr = f" {filters.condition} ".join(metadata_filters)  # 使用指定条件连接过滤器表达式
        else:  # 如果只有一个过滤器
            metadata_filter_expr = metadata_filters[0]  # 直接取第一个过滤器表达式
        return metadata_filter_expr  # 返回最终的元数据过滤器表达式
```