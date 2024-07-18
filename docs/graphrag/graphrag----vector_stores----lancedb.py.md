# `.\graphrag\graphrag\vector_stores\lancedb.py`

```py
"""The LanceDB vector storage implementation package."""

# 导入必要的库和模块
import lancedb as lancedb  # noqa: I001 (Ruff was breaking on this file imports, even tho they were sorted and passed local tests)
from graphrag.model.types import TextEmbedder

import json
from typing import Any

import pyarrow as pa

# 导入基础向量存储相关的类和方法
from .base import (
    BaseVectorStore,
    VectorStoreDocument,
    VectorStoreSearchResult,
)

# 定义 LanceDBVectorStore 类，继承自 BaseVectorStore 类
class LanceDBVectorStore(BaseVectorStore):
    """The LanceDB vector storage implementation."""

    # 连接到向量存储的方法
    def connect(self, **kwargs: Any) -> Any:
        """Connect to the vector storage."""
        # 获取数据库连接 URI，默认为当前目录下的 lancedb
        db_uri = kwargs.get("db_uri", "./lancedb")
        # 使用 lancedb 模块连接到数据库，忽略类型检查
        self.db_connection = lancedb.connect(db_uri)  # type: ignore

    # 将文档加载到向量存储的方法
    def load_documents(
        self, documents: list[VectorStoreDocument], overwrite: bool = True
    ) -> None:
        """Load documents into vector storage."""
        # 准备要存储的文档数据
        data = [
            {
                "id": document.id,
                "text": document.text,
                "vector": document.vector,
                "attributes": json.dumps(document.attributes),
            }
            for document in documents
            if document.vector is not None
        ]

        # 如果没有有效数据，则设为 None
        if len(data) == 0:
            data = None

        # 定义存储数据的 schema
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float64())),
            pa.field("attributes", pa.string()),
        ])

        # 如果允许覆盖已有数据
        if overwrite:
            # 如果有数据，则创建新的表格或覆盖已有表格
            if data:
                self.document_collection = self.db_connection.create_table(
                    self.collection_name, data=data, mode="overwrite"
                )
            else:
                self.document_collection = self.db_connection.create_table(
                    self.collection_name, schema=schema, mode="overwrite"
                )
        else:
            # 否则，添加数据到已有表格
            self.document_collection = self.db_connection.open_table(
                self.collection_name
            )
            if data:
                self.document_collection.add(data)

    # 根据 ID 进行过滤的方法
    def filter_by_id(self, include_ids: list[str] | list[int]) -> Any:
        """Build a query filter to filter documents by id."""
        # 如果没有包含任何 ID，则查询过滤器设为 None
        if len(include_ids) == 0:
            self.query_filter = None
        else:
            # 否则，根据 ID 类型构建查询过滤器
            if isinstance(include_ids[0], str):
                id_filter = ", ".join([f"'{id}'" for id in include_ids])
                self.query_filter = f"id in ({id_filter})"
            else:
                self.query_filter = (
                    f"id in ({', '.join([str(id) for id in include_ids])})"
                )
        return self.query_filter

    # 根据向量进行相似性搜索的方法
    def similarity_search_by_vector(
        self, query_embedding: list[float], k: int = 10, **kwargs: Any
    def similarity_search_by_vector(
        self, query_embedding: np.ndarray, k: int = 10
    ) -> list[VectorStoreSearchResult]:
        """Perform a vector-based similarity search."""
        # 如果有查询过滤器，使用过滤器对查询进行限制，并返回结果列表
        if self.query_filter:
            docs = (
                self.document_collection.search(query=query_embedding)
                .where(self.query_filter, prefilter=True)
                .limit(k)
                .to_list()
            )
        else:
            # 如果没有查询过滤器，直接返回查询结果列表
            docs = (
                self.document_collection.search(query=query_embedding)
                .limit(k)
                .to_list()
            )
        # 根据查询结果构建 VectorStoreSearchResult 对象的列表，并计算每个结果的得分
        return [
            VectorStoreSearchResult(
                document=VectorStoreDocument(
                    id=doc["id"],
                    text=doc["text"],
                    vector=doc["vector"],
                    attributes=json.loads(doc["attributes"]),
                ),
                score=1 - abs(float(doc["_distance"])),
            )
            for doc in docs
        ]

    def similarity_search_by_text(
        self, text: str, text_embedder: TextEmbedder, k: int = 10, **kwargs: Any
    ) -> list[VectorStoreSearchResult]:
        """Perform a similarity search using a given input text."""
        # 将输入文本转换为查询向量
        query_embedding = text_embedder(text)
        # 如果成功获取到查询向量，则调用 similarity_search_by_vector 方法进行相似性搜索
        if query_embedding:
            return self.similarity_search_by_vector(query_embedding, k)
        # 如果未能获取到有效的查询向量，则返回空列表
        return []
```