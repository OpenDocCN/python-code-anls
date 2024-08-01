# `.\DB-GPT-src\dbgpt\storage\vector_store\chroma_store.py`

```py
"""Chroma vector store."""
# 引入日志模块
import logging
# 引入操作系统相关的模块
import os
# 引入类型提示相关的模块
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

# 引入 Chroma 数据库客户端
from chromadb import PersistentClient
# 引入 Chroma 的配置设置
from chromadb.config import Settings

# 引入 Pydantic 的配置字典和字段
from dbgpt._private.pydantic import ConfigDict, Field
# 引入 Pilot 路径配置
from dbgpt.configs.model_config import PILOT_PATH
# 引入核心 Chunk 模块
from dbgpt.core import Chunk
# 引入 AWEL 流相关的模块
from dbgpt.core.awel.flow import Parameter, ResourceCategory, register_resource
# 引入国际化相关的工具函数
from dbgpt.util.i18n_utils import _

# 引入本地的基类和相关配置
from .base import _COMMON_PARAMETERS, VectorStoreBase, VectorStoreConfig
# 引入过滤器相关的模块
from .filters import FilterOperator, MetadataFilters

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# 定义 Chroma 集合名称常量
CHROMA_COLLECTION_NAME = "langchain"

# 注册 Chroma 向量存储资源
@register_resource(
    _("Chroma Vector Store"),
    "chroma_vector_store",
    category=ResourceCategory.VECTOR_STORE,
    description=_("Chroma vector store."),
    parameters=[
        *_COMMON_PARAMETERS,
        Parameter.build_from(
            _("Persist Path"),
            "persist_path",
            str,
            description=_("the persist path of vector store."),
            optional=True,
            default=None,
        ),
    ],
)
# Chroma 向量存储配置类
class ChromaVectorConfig(VectorStoreConfig):
    """Chroma vector store config."""

    # 模型配置字典，允许任意类型
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 向量存储的持久化路径，可选项
    persist_path: Optional[str] = Field(
        default=os.getenv("CHROMA_PERSIST_PATH", None),
        description="the persist path of vector store.",
    )
    # 集合的元数据，如果未设置，将使用默认元数据
    collection_metadata: Optional[dict] = Field(
        default=None,
        description="the index metadata of vector store, if not set, will use the "
        "default metadata.",
    )

# Chroma 存储类，继承自向量存储基类
class ChromaStore(VectorStoreBase):
    """Chroma vector store."""

    # 初始化方法，创建 ChromaStore 实例
    def __init__(self, vector_store_config: ChromaVectorConfig) -> None:
        """Create a ChromaStore instance.

        Args:
            vector_store_config(ChromaVectorConfig): vector store config.
        """
        # 调用父类的初始化方法
        super().__init__()

        # 将向量存储配置转换为字典形式
        chroma_vector_config = vector_store_config.to_dict(exclude_none=True)
        # 获取 Chroma 的持久化路径
        chroma_path = chroma_vector_config.get(
            "persist_path", os.path.join(PILOT_PATH, "data")
        )
        # 设置持久化目录
        self.persist_dir = os.path.join(
            chroma_path, vector_store_config.name + ".vectordb"
        )
        # 设置嵌入函数（未在提供的代码中找到定义，此处可能需要进一步调整）
        self.embeddings = vector_store_config.embedding_fn
        
        # 设置 Chroma 的配置信息
        chroma_settings = Settings(
            persist_directory=self.persist_dir,  # 持久化目录
            anonymized_telemetry=False,  # 关闭匿名化遥测
        )
        # 创建 Chroma 的持久化客户端
        self._chroma_client = PersistentClient(
            path=self.persist_dir,  # 持久化路径
            settings=chroma_settings  # Chroma 的配置设置
        )

        # 获取集合的元数据，如果未设置，则使用默认值
        collection_metadata = chroma_vector_config.get("collection_metadata") or {
            "hnsw:space": "cosine"
        }
        # 获取或创建 Chroma 集合
        self._collection = self._chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,  # 集合名称
            embedding_function=None,  # 嵌入函数（未在提供的代码中找到定义，此处可能需要进一步调整）
            metadata=collection_metadata  # 集合的元数据
        )
    def similar_search(
        self, text, topk, filters: Optional[MetadataFilters] = None
    ) -> List[Chunk]:
        """Search similar documents."""
        # 记录信息到日志：ChromaStore 相似搜索
        logger.info("ChromaStore similar search")
        # 调用内部方法 _query 进行查询，获取相似文档的结果
        chroma_results = self._query(
            text=text,
            topk=topk,
            filters=filters,
        )
        # 构建并返回 Chunk 对象的列表，每个 Chunk 对象包含文本内容、元数据以及固定的分数 0.0
        return [
            Chunk(content=chroma_result[0], metadata=chroma_result[1] or {}, score=0.0)
            for chroma_result in zip(
                chroma_results["documents"][0],
                chroma_results["metadatas"][0],
            )
        ]

    def similar_search_with_scores(
        self, text, topk, score_threshold, filters: Optional[MetadataFilters] = None
    ) -> List[Chunk]:
        """Search similar documents with scores.

        Chroma similar_search_with_score.
        Return docs and relevance scores in the range [0, 1].
        Args:
            text(str): query text
            topk(int): return docs nums. Defaults to 4.
            score_threshold(float): score_threshold: Optional, a floating point value
                between 0 to 1 to filter the resulting set of retrieved docs,0 is
                dissimilar, 1 is most similar.
            filters(MetadataFilters): metadata filters, defaults to None
        """
        # 记录信息到日志：ChromaStore 带分数的相似搜索
        logger.info("ChromaStore similar search with scores")
        # 调用内部方法 _query 进行查询，获取相似文档的结果
        chroma_results = self._query(
            text=text,
            topk=topk,
            filters=filters,
        )
        # 构建并返回 Chunk 对象的列表，每个 Chunk 对象包含文本内容、元数据和基于相似度计算的分数
        chunks = [
            (
                Chunk(
                    content=chroma_result[0],
                    metadata=chroma_result[1] or {},
                    score=(1 - chroma_result[2]),
                )
            )
            for chroma_result in zip(
                chroma_results["documents"][0],
                chroma_results["metadatas"][0],
                chroma_results["distances"][0],
            )
        ]
        # 根据给定的分数阈值过滤 Chunk 列表，并返回过滤后的结果
        return self.filter_by_score_threshold(chunks, score_threshold)

    def vector_name_exists(self) -> bool:
        """Whether vector name exists."""
        # 记录信息到日志，检查持久化目录是否存在
        logger.info(f"Check persist_dir: {self.persist_dir}")
        # 检查持久化目录是否存在，如果不存在则返回 False
        if not os.path.exists(self.persist_dir):
            return False
        # 获取持久化目录下的文件列表
        files = os.listdir(self.persist_dir)
        # 过滤掉默认文件 'chroma.sqlite3'
        files = list(filter(lambda f: f != "chroma.sqlite3", files))
        # 如果过滤后的文件列表长度大于 0，则说明存在非默认文件，返回 True
        return len(files) > 0

    def load_document(self, chunks: List[Chunk]) -> List[str]:
        """Load document to vector store."""
        # 记录信息到日志：ChromaStore 加载文档
        logger.info("ChromaStore load document")
        # 从传入的 Chunk 列表中提取文本内容、元数据和 ID 列表
        texts = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        # 将元数据转换为适合 Chroma 的格式
        chroma_metadatas = [
            _transform_chroma_metadata(metadata) for metadata in metadatas
        ]
        # 调用内部方法 _add_texts 将文本、元数据和 ID 添加到向量存储中
        self._add_texts(texts=texts, metadatas=chroma_metadatas, ids=ids)
        # 返回添加的文档的 ID 列表
        return ids
    def delete_vector_name(self, vector_name: str):
        """Delete vector name."""
        # 记录日志，指示开始删除指定的向量名称
        logger.info(f"chroma vector_name:{vector_name} begin delete...")
        # 调用_chroma_client对象的方法删除指定集合（collection）
        self._chroma_client.delete_collection(self._collection.name)
        # 清理持久化文件夹中的数据
        self._clean_persist_folder()
        return True

    def delete_by_ids(self, ids):
        """Delete vector by ids."""
        # 记录日志，指示开始按照指定的ids删除向量
        logger.info(f"begin delete chroma ids: {ids}")
        # 将传入的ids字符串按逗号分割成列表
        ids = ids.split(",")
        # 如果ids列表不为空
        if len(ids) > 0:
            # 调用_collection对象的delete方法，删除指定ids对应的数据
            self._collection.delete(ids=ids)

    def convert_metadata_filters(
        self,
        filters: MetadataFilters,
    ) -> dict:
        """Convert metadata filters to Chroma filters.

        Args:
            filters(MetadataFilters): metadata filters.
        Returns:
            dict: Chroma filters.
        """
        # 初始化空的where_filters字典，用于存储转换后的Chroma过滤器
        where_filters = {}
        # 初始化空的filters_list列表，用于存储转换后的Chroma过滤器列表
        filters_list = []
        # 获取MetadataFilters对象的condition属性值
        condition = filters.condition
        # 构建Chroma条件字符串
        chroma_condition = f"${condition.value}"
        # 如果filters对象中存在filters列表
        if filters.filters:
            # 遍历filters中的每个filter对象
            for filter in filters.filters:
                # 如果filter对象有operator属性
                if filter.operator:
                    # 将filter对象转换为Chroma过滤器格式，并添加到filters_list中
                    filters_list.append(
                        {
                            filter.key: {
                                _convert_chroma_filter_operator(
                                    filter.operator
                                ): filter.value
                            }
                        }
                    )
                else:
                    # 将filter对象直接作为键值对添加到filters_list中（类型忽略检查）
                    filters_list.append({filter.key: filter.value})  # type: ignore

        # 如果filters_list中只有一个元素，直接返回该元素
        if len(filters_list) == 1:
            return filters_list[0]
        # 如果filters_list中有多个元素，将它们作为Chroma过滤器的OR条件
        elif len(filters_list) > 1:
            where_filters[chroma_condition] = filters_list
        # 返回组装好的Chroma过滤器
        return where_filters

    def _add_texts(
        self,
        texts: Iterable[str],
        ids: List[str],
        metadatas: Optional[List[Mapping[str, Union[str, int, float, bool]]]] = None,
    ) -> List[str]:
        """Add texts to Chroma collection.

        Args:
            texts(Iterable[str]): texts.
            metadatas(Optional[List[dict]]): metadatas.
            ids(Optional[List[str]]): ids.
        Returns:
            List[str]: ids.
        """
        # 初始化embeddings为None
        embeddings = None
        # 将texts转换为列表
        texts = list(texts)
        # 如果存在embeddings对象，则调用其embed_documents方法获取文本的嵌入向量
        if self.embeddings is not None:
            embeddings = self.embeddings.embed_documents(texts)
        # 如果存在metadatas参数
        if metadatas:
            try:
                # 使用_collection对象的upsert方法，将文本、元数据和嵌入向量插入或更新到Chroma集合中
                self._collection.upsert(
                    metadatas=metadatas,
                    embeddings=embeddings,  # type: ignore
                    documents=texts,
                    ids=ids,
                )
            except ValueError as e:
                # 记录错误日志，指示在插入带有元数据的数据时出现了值错误
                logger.error(f"Error upsert chromadb with metadata: {e}")
        else:
            # 使用_collection对象的upsert方法，将文本和嵌入向量插入或更新到Chroma集合中
            self._collection.upsert(
                embeddings=embeddings,  # type: ignore
                documents=texts,
                ids=ids,
            )
        # 返回已处理的ids列表
        return ids
    def _query(self, text: str, topk: int, filters: Optional[MetadataFilters] = None):
        """Query Chroma collection.

        Args:
            text(str): query text.
            topk(int): topk.
            filters(MetadataFilters): metadata filters.
        Returns:
            dict: query result.
        """
        # 如果查询文本为空字符串，则返回空字典
        if not text:
            return {}
        # 将元数据过滤器转换为适合查询的格式，如果没有提供过滤器则设为None
        where_filters = self.convert_metadata_filters(filters) if filters else None
        # 如果嵌入向量为空，则引发数值错误异常
        if self.embeddings is None:
            raise ValueError("Chroma Embeddings is None")
        # 对查询文本生成查询嵌入向量
        query_embedding = self.embeddings.embed_query(text)
        # 调用集合对象的查询方法，使用查询嵌入向量、topk参数和过滤器进行查询
        return self._collection.query(
            query_embeddings=query_embedding,
            n_results=topk,
            where=where_filters,
        )

    def _clean_persist_folder(self):
        """Clean persist folder."""
        # 递归遍历持久化文件夹中的所有内容
        for root, dirs, files in os.walk(self.persist_dir, topdown=False):
            # 删除所有文件
            for name in files:
                os.remove(os.path.join(root, name))
            # 删除所有子文件夹
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        # 最后删除持久化文件夹本身
        os.rmdir(self.persist_dir)
# 将操作符转换为 Chroma 查询中对应的操作符字符串
def _convert_chroma_filter_operator(operator: str) -> str:
    """Convert operator to Chroma where operator.

    Args:
        operator(str): operator to convert.

    Returns:
        str: Corresponding Chroma where operator string.
        
    Raises:
        ValueError: If the provided operator is not recognized.
    """
    if operator == FilterOperator.EQ:
        return "$eq"  # 等于操作符对应的 Chroma 查询操作符
    elif operator == FilterOperator.NE:
        return "$ne"  # 不等于操作符对应的 Chroma 查询操作符
    elif operator == FilterOperator.GT:
        return "$gt"  # 大于操作符对应的 Chroma 查询操作符
    elif operator == FilterOperator.LT:
        return "$lt"  # 小于操作符对应的 Chroma 查询操作符
    elif operator == FilterOperator.GTE:
        return "$gte"  # 大于等于操作符对应的 Chroma 查询操作符
    elif operator == FilterOperator.LTE:
        return "$lte"  # 小于等于操作符对应的 Chroma 查询操作符
    else:
        raise ValueError(f"Chroma Where operator {operator} not supported")  # 抛出异常，指示不支持的操作符


# 将元数据转换为 Chroma 元数据格式
def _transform_chroma_metadata(
    metadata: Dict[str, Any]
) -> Mapping[str, str | int | float | bool]:
    """Transform metadata to Chroma metadata.

    Args:
        metadata (Dict[str, Any]): Metadata dictionary to transform.

    Returns:
        Mapping[str, str | int | float | bool]: Transformed Chroma metadata dictionary.
    """
    transformed = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            transformed[key] = value  # 保留类型为 str、int、float 或 bool 的元数据项
    return transformed  # 返回转换后的 Chroma 元数据字典
```