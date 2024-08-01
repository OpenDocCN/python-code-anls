# `.\DB-GPT-src\dbgpt\storage\vector_store\elastic_store.py`

```py
"""Elasticsearch vector store."""

# 导入必要的库和模块
from __future__ import annotations  # 允许在类型注释中使用未定义的类

import logging  # 导入日志记录模块
import os  # 导入操作系统相关模块
from typing import List, Optional  # 引入类型提示相关模块

from dbgpt._private.pydantic import Field  # 导入 Pydantic 的 Field 类
from dbgpt.core import Chunk, Embeddings  # 导入核心功能模块
from dbgpt.core.awel.flow import Parameter, ResourceCategory, register_resource  # 导入参数、资源类别和注册资源函数
from dbgpt.storage.vector_store.base import (
    _COMMON_PARAMETERS,  # 导入通用参数列表
    VectorStoreBase,  # 导入向量存储基类
    VectorStoreConfig,  # 导入向量存储配置类
)
from dbgpt.storage.vector_store.filters import MetadataFilters  # 导入元数据过滤器类
from dbgpt.util import string_utils  # 导入字符串工具函数
from dbgpt.util.i18n_utils import _  # 导入国际化函数

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

@register_resource(
    _("ElasticSearch Vector Store"),  # 注册 Elasticsearch 向量存储资源
    "elasticsearch_vector_store",
    category=ResourceCategory.VECTOR_STORE,  # 设置资源类别为向量存储
    parameters=[  # 定义资源参数列表
        *_COMMON_PARAMETERS,  # 引用通用参数列表
        Parameter.build_from(
            _("Uri"),  # 参数名称：Uri
            "uri",
            str,
            description=_(
                "The uri of elasticsearch store, if not set, will use the default "
                "uri."
            ),
            optional=True,  # 参数可选
            default="localhost",  # 默认值为 localhost
        ),
        Parameter.build_from(
            _("Port"),  # 参数名称：Port
            "port",
            str,
            description=_(
                "The port of elasticsearch store, if not set, will use the default "
                "port."
            ),
            optional=True,  # 参数可选
            default="9200",  # 默认端口为 9200
        ),
        Parameter.build_from(
            _("Alias"),  # 参数名称：Alias
            "alias",
            str,
            description=_(
                "The alias of elasticsearch store, if not set, will use the default "
                "alias."
            ),
            optional=True,  # 参数可选
            default="default",  # 默认别名为 default
        ),
        Parameter.build_from(
            _("Index Name"),  # 参数名称：Index Name
            "index_name",
            str,
            description=_(
                "The index name of elasticsearch store, if not set, will use the "
                "default index name."
            ),
            optional=True,  # 参数可选
            default="index_name_test",  # 默认索引名称为 index_name_test
        ),
    ],
    description=_("Elasticsearch vector store."),  # 描述：Elasticsearch 向量存储
)
class ElasticsearchVectorConfig(VectorStoreConfig):
    """Elasticsearch vector store config."""

    class Config:
        """Config for BaseModel."""

        arbitrary_types_allowed = True  # 允许任意类型

    uri: str = Field(
        default="localhost",  # 默认 URI 为 localhost
        description="The uri of elasticsearch store, if not set, will use the default "
        "uri.",
    )
    port: str = Field(
        default="9200",  # 默认端口为 9200
        description="The port of elasticsearch store, if not set, will use the default "
        "port.",
    )

    alias: str = Field(
        default="default",  # 默认别名为 default
        description="The alias of elasticsearch store, if not set, will use the "
        "default "
        "alias.",
    )
    index_name: str = Field(
        default="index_name_test",  # 默认索引名称为 index_name_test
        description="The index name of elasticsearch store, if not set, will use the "
        "default index name.",
    )
    metadata_field: str = Field(
        default="metadata",
        description="The metadata field of elasticsearch store, if not set, will use "
        "the default metadata field.",
    )
    secure: str = Field(
        default="",
        description="The secure of elasticsearch store, if not set, will use the "
        "default secure.",
    )



# 定义 metadata_field 字段，用于指定 Elasticsearch 存储中的元数据字段
metadata_field: str = Field(
    default="metadata",
    description="The metadata field of elasticsearch store, if not set, will use "
    "the default metadata field.",
)

# 定义 secure 字段，用于指定 Elasticsearch 存储的安全设置
secure: str = Field(
    default="",
    description="The secure of elasticsearch store, if not set, will use the "
    "default secure.",
)
class ElasticStore(VectorStoreBase):
    """Elasticsearch vector store."""

    def load_document(
        self,
        chunks: List[Chunk],
    ) -> List[str]:
        """Add text data into ElasticSearch."""
        # 记录日志：加载文档到ElasticSearch
        logger.info("ElasticStore load document")
        try:
            # 尝试导入ElasticsearchStore类
            from langchain.vectorstores.elasticsearch import ElasticsearchStore
        except ImportError:
            # 如果导入失败，抛出错误提示用户安装langchain和elasticsearch包
            raise ValueError(
                "Could not import langchain python package. "
                "Please install it with `pip install langchain` and "
                "`pip install elasticsearch`."
            )
        try:
            # 提取每个Chunk对象的文本内容、元数据和ID
            texts = [chunk.content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            ids = [chunk.chunk_id for chunk in chunks]
            # 如果提供了用户名和密码，使用from_texts方法将文本数据存入Elasticsearch
            if self.username != "" and self.password != "":
                self.db = ElasticsearchStore.from_texts(
                    texts=texts,
                    embedding=self.embedding,  # type: ignore
                    metadatas=metadatas,
                    ids=ids,
                    es_url=f"http://{self.uri}:{self.port}",
                    index_name=self.index_name,
                    distance_strategy="COSINE",
                    query_field="context",
                    vector_query_field="dense_vector",
                    es_user=self.username,
                    es_password=self.password,
                )  # type: ignore
                # 记录成功的日志信息
                logger.info("Elasticsearch save success.......")
                return ids  # 返回文档的ID列表
            else:
                # 如果没有提供用户名和密码，使用from_documents方法将文档数据存入Elasticsearch
                self.db = ElasticsearchStore.from_documents(
                    texts=texts,
                    embedding=self.embedding,  # type: ignore
                    metadatas=metadatas,
                    ids=ids,
                    es_url=f"http://{self.uri}:{self.port}",
                    index_name=self.index_name,
                    distance_strategy="COSINE",
                    query_field="context",
                    vector_query_field="dense_vector",
                )  # type: ignore
                return ids  # 返回文档的ID列表
        except ConnectionError as ce:
            # 捕获连接错误并记录日志
            logger.error(f"ElasticSearch connect failed {ce}")
        except Exception as e:
            # 捕获其他异常并记录日志
            logger.error(f"ElasticSearch load_document failed : {e}")
        return []  # 返回空列表，表示加载文档失败
    def delete_by_ids(self, ids):
        """Delete vector by ids."""
        # 记录日志，显示即将删除 Elasticsearch 中的向量数量
        logger.info(f"begin delete elasticsearch len ids: {len(ids)}")
        # 将传入的 ids 字符串以逗号分隔为列表
        ids = ids.split(",")
        try:
            # 调用 self.db_init.delete 方法删除指定 ids 的数据
            self.db_init.delete(ids=ids)
            # 刷新 Elasticsearch 索引以确保删除操作生效
            self.es_client_python.indices.refresh(index=self.index_name)
        except Exception as e:
            # 若删除操作失败，记录错误日志
            logger.error(f"ElasticSearch delete_by_ids failed : {e}")

    def similar_search(
        self,
        text: str,
        topk: int,
        filters: Optional[MetadataFilters] = None,
    ) -> List[Chunk]:
        """Perform a search on a query string and return results."""
        # 调用内部方法 _search 进行查询操作，返回查询结果
        info_docs = self._search(query=text, topk=topk, filters=filters)
        return info_docs

    def similar_search_with_scores(
        self, text, topk, score_threshold, filters: Optional[MetadataFilters] = None
    ) -> List[Chunk]:
        """Perform a search on a query string and return results with score.

        For more information about the search parameters, take a look at the
        ElasticSearch documentation found here: https://www.elastic.co/.

        Args:
            text (str): The query text.
            topk (int): The number of similar documents to return.
            score_threshold (float): Optional, a floating point value between 0 to 1.
            filters (Optional[MetadataFilters]): Optional, metadata filters.
        Returns:
            List[Chunk]: Result doc and score.
        """
        # 设置查询参数
        query = text
        # 调用内部方法 _search 进行查询操作，返回查询结果
        info_docs = self._search(query=query, topk=topk, filters=filters)
        # 筛选出得分高于 score_threshold 的文档
        docs_and_scores = [
            chunk for chunk in info_docs if chunk.score >= score_threshold
        ]
        # 若未检索到符合条件的文档，记录警告日志
        if len(docs_and_scores) == 0:
            logger.warning(
                "No relevant docs were retrieved using the relevance score"
                f" threshold {score_threshold}"
            )
        return docs_and_scores

    def _search(
        self, query: str, topk: int, filters: Optional[MetadataFilters] = None, **kwargs
    ):
        """Internal method to perform a search and return documents."""
        # 此方法未给出具体实现，仅作为注释说明，应在代码的其余部分进行定义和实现
        pass
    ) -> List[Chunk]:
        """搜索相似文档。

        Args:
            query: 查询文本
            topk: 返回文档数量。默认为4。
            filters: 元数据过滤器。
        Return:
            List[Chunk]: 分块列表
        """
        # 从kwargs中弹出"jieba_tokenize"参数
        jieba_tokenize = kwargs.pop("jieba_tokenize", None)
        # 如果jieba_tokenize为True，则使用jieba进行分词
        if jieba_tokenize:
            try:
                import jieba
                import jieba.analyse
            except ImportError:
                raise ValueError("请使用`pip install jieba`安装它。")
            # 使用textrank算法提取关键词
            query_list = jieba.analyse.textrank(query, topK=20, withWeight=False)
            query = " ".join(query_list)
        # 构建Elasticsearch查询体
        body = {"query": {"match": {"context": query}}}
        # 在Elasticsearch中执行查询
        search_results = self.es_client_python.search(
            index=self.index_name, body=body, size=topk
        )
        search_results = search_results["hits"]["hits"]

        # 如果没有搜索结果，则返回空列表
        if not search_results:
            logger.warning("""未找到ElasticSearch结果。""")
            return []
        info_docs = []
        # 遍历搜索结果，构建Chunk对象列表
        for result in search_results:
            doc_id = result["_id"]
            source = result["_source"]
            context = source["context"]
            metadata = source["metadata"]
            score = result["_score"]
            doc_with_score = Chunk(
                content=context, metadata=metadata, score=score, chunk_id=doc_id
            )
            info_docs.append(doc_with_score)
        return info_docs

    def vector_name_exists(self):
        """检查向量名称是否存在。"""
        return self.es_client_python.indices.exists(index=self.index_name)

    def delete_vector_name(self, vector_name: str):
        """删除向量名称/索引名称。"""
        # 如果索引名称存在，则删除索引
        if self.es_client_python.indices.exists(index=self.index_name):
            self.es_client_python.indices.delete(index=self.index_name)
```