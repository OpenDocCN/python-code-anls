# `.\graphrag\graphrag\vector_stores\azure_ai_search.py`

```py
# 版权声明和许可信息
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""包含 Azure AI Search 向量存储实现的包。"""

# 导入所需的模块和类
import json
from typing import Any

from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery

from graphrag.model.types import TextEmbedder

# 导入基类和常量
from .base import (
    DEFAULT_VECTOR_SIZE,
    BaseVectorStore,
    VectorStoreDocument,
    VectorStoreSearchResult,
)

# AzureAISearch 类，继承自 BaseVectorStore
class AzureAISearch(BaseVectorStore):
    """Azure AI Search 向量存储实现。"""

    index_client: SearchIndexClient

    # 连接到 AzureAI 向量存储
    def connect(self, **kwargs: Any) -> Any:
        """连接到 AzureAI 向量存储。"""
        # 从参数中获取连接相关的信息
        url = kwargs.get("url", None)
        api_key = kwargs.get("api_key", None)
        audience = kwargs.get("audience", None)
        self.vector_size = kwargs.get("vector_size", DEFAULT_VECTOR_SIZE)

        # 设置向量搜索配置文件名称，默认为 "vectorSearchProfile"
        self.vector_search_profile_name = kwargs.get(
            "vector_search_profile_name", "vectorSearchProfile"
        )

        # 如果提供了 URL，则创建 SearchClient 和 SearchIndexClient 实例
        if url:
            # 如果提供了 audience 参数，则作为参数传递给 SearchClient 和 SearchIndexClient
            audience_arg = {"audience": audience} if audience else {}
            # 创建 SearchClient 实例
            self.db_connection = SearchClient(
                endpoint=url,
                index_name=self.collection_name,
                credential=AzureKeyCredential(api_key)
                if api_key
                else DefaultAzureCredential(),
                **audience_arg,
            )
            # 创建 SearchIndexClient 实例
            self.index_client = SearchIndexClient(
                endpoint=url,
                credential=AzureKeyCredential(api_key)
                if api_key
                else DefaultAzureCredential(),
                **audience_arg,
            )
        else:
            # 如果没有提供 URL，则抛出 ValueError 异常
            not_supported_error = "AAISearchDBClient is not supported on local host."
            raise ValueError(not_supported_error)

    def load_documents(
        self, documents: list[VectorStoreDocument], overwrite: bool = True
        # 加载文档到向量存储
    ) -> None:
        """将文档加载到 Azure AI Search 索引中。"""
        # 如果 overwrite 参数为 True，则删除现有同名索引
        if overwrite:
            if self.collection_name in self.index_client.list_index_names():
                self.index_client.delete_index(self.collection_name)

            # 配置向量搜索的算法和配置
            vector_search = VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="HnswAlg",
                        parameters=HnswParameters(
                            metric=VectorSearchAlgorithmMetric.COSINE
                        ),
                    )
                ],
                profiles=[
                    VectorSearchProfile(
                        name=self.vector_search_profile_name,
                        algorithm_configuration_name="HnswAlg",
                    )
                ],
            )

            # 创建新的搜索索引对象
            index = SearchIndex(
                name=self.collection_name,
                fields=[
                    SimpleField(
                        name="id",
                        type=SearchFieldDataType.String,
                        key=True,
                    ),
                    SearchField(
                        name="vector",
                        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                        searchable=True,
                        vector_search_dimensions=self.vector_size,
                        vector_search_profile_name=self.vector_search_profile_name,
                    ),
                    SearchableField(name="text", type=SearchFieldDataType.String),
                    SimpleField(
                        name="attributes",
                        type=SearchFieldDataType.String,
                    ),
                ],
                vector_search=vector_search,
            )

            # 创建或更新搜索索引
            self.index_client.create_or_update_index(
                index,
            )

        # 准备要上传的文档批处理
        batch = [
            {
                "id": doc.id,
                "vector": doc.vector,
                "text": doc.text,
                "attributes": json.dumps(doc.attributes),
            }
            for doc in documents
            if doc.vector is not None
        ]

        # 如果批处理不为空，则上传文档
        if batch and len(batch) > 0:
            self.db_connection.upload_documents(batch)
    def filter_by_id(self, include_ids: list[str] | list[int]) -> Any:
        """Build a query filter to filter documents by a list of ids."""
        # 如果传入的 include_ids 为空或者长度为0，则将查询过滤器设为 None
        if include_ids is None or len(include_ids) == 0:
            self.query_filter = None
            # 返回 self.query_filter 是为了与其他方法保持一致，但实际上不需要返回值
            return self.query_filter

        # 使用逗号连接 include_ids 中的每个 id，构建搜索条件字符串
        id_filter = ",".join([f"{id!s}" for id in include_ids])
        # 将查询过滤器设为符合 Azure Search 查询语法的字符串形式
        self.query_filter = f"search.in(id, '{id_filter}', ',')"

        # 返回 self.query_filter 是为了与其他方法保持一致，但实际上不需要返回值
        # TODO: 在未来的 PR 中重构此处
        return self.query_filter

    def similarity_search_by_vector(
        self, query_embedding: list[float], k: int = 10, **kwargs: Any
    ) -> list[VectorStoreSearchResult]:
        """Perform a vector-based similarity search."""
        # 创建一个 VectorizedQuery 对象，用于向量相似性搜索
        vectorized_query = VectorizedQuery(
            vector=query_embedding, k_nearest_neighbors=k, fields="vector"
        )

        # 使用 self.db_connection 执行向量查询，并获取响应结果
        response = self.db_connection.search(
            vector_queries=[vectorized_query],
        )

        # 将响应中的每个文档转换为 VectorStoreSearchResult 对象的列表，并返回
        return [
            VectorStoreSearchResult(
                document=VectorStoreDocument(
                    id=doc.get("id", ""),
                    text=doc.get("text", ""),
                    vector=doc.get("vector", []),
                    attributes=(json.loads(doc.get("attributes", "{}"))),
                ),
                # 每个文档的余弦相似度得分，范围在0.333到1.000之间
                # 参考：https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking#scores-in-a-hybrid-search-results
                score=doc["@search.score"],
            )
            for doc in response
        ]

    def similarity_search_by_text(
        self, text: str, text_embedder: TextEmbedder, k: int = 10, **kwargs: Any
    ) -> list[VectorStoreSearchResult]:
        """Perform a text-based similarity search."""
        # 使用 text_embedder 将文本转换为查询向量
        query_embedding = text_embedder(text)
        # 如果成功生成了查询向量，则调用 similarity_search_by_vector 进行向量相似性搜索
        if query_embedding:
            return self.similarity_search_by_vector(
                query_embedding=query_embedding, k=k
            )
        # 如果无法生成查询向量，则返回空列表
        return []
```