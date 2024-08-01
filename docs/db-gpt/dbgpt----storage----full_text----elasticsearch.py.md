# `.\DB-GPT-src\dbgpt\storage\full_text\elasticsearch.py`

```py
"""Elasticsearch document store."""
# 引入所需的模块和类
import json
import os
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import List, Optional

from dbgpt.core import Chunk
from dbgpt.rag.index.base import logger
from dbgpt.storage.full_text.base import FullTextStoreBase
from dbgpt.storage.vector_store.elastic_store import ElasticsearchVectorConfig
from dbgpt.storage.vector_store.filters import MetadataFilters
from dbgpt.util import string_utils
from dbgpt.util.executor_utils import blocking_func_to_async

# 定义 ElasticDocumentConfig 类，继承自 ElasticsearchVectorConfig
class ElasticDocumentConfig(ElasticsearchVectorConfig):
    """Elasticsearch document store config."""

    k1: Optional[float] = 2.0
    b: Optional[float] = 0.75

# 定义 ElasticDocumentStore 类，继承自 FullTextStoreBase
class ElasticDocumentStore(FullTextStoreBase):
    """Elasticsearch index store."""

    def __init__(
        self, es_config: ElasticDocumentConfig, executor: Optional[Executor] = None
    ):
        """Initialize Elasticsearch index store.

        refer https://www.elastic.co/guide/en/elasticsearch/reference/8.9/index-modules-similarity.html
        TF/IDF based similarity that has built-in tf normalization and is supposed to
        work better for short fields (like names). See Okapi_BM25 for more details.
        This similarity has the following options:
        """
        super().__init__()  # 调用父类的初始化方法

        from elasticsearch import Elasticsearch
        
        # 设置 Elasticsearch 的配置信息
        self._es_config = es_config
        self._es_url = es_config.uri or os.getenv("ELASTICSEARCH_URL", "localhost")
        self._es_port = es_config.port or os.getenv("ELASTICSEARCH_PORT", "9200")
        self._es_username = es_config.user or os.getenv("ELASTICSEARCH_USER", "elastic")
        self._es_password = es_config.password or os.getenv("ELASTICSEARCH_PASSWORD", "dbgpt")
        self._index_name = es_config.name.lower()
        
        # 如果索引名称包含中文，转换为十六进制编码作为新索引名称
        if string_utils.contains_chinese(es_config.name):
            bytes_str = es_config.name.encode("utf-8")
            hex_str = bytes_str.hex()
            self._index_name = "dbgpt_" + hex_str
        
        # 设置 BM25 相似度算法的参数 k1 和 b
        self._k1 = es_config.k1 or 2.0
        self._b = es_config.b or 0.75
        
        # 根据配置信息连接 Elasticsearch 客户端
        if self._es_username and self._es_password:
            self._es_client = Elasticsearch(
                hosts=[f"http://{self._es_url}:{self._es_port}"],
                basic_auth=(self._es_username, self._es_password),
            )
        else:
            self._es_client = Elasticsearch(
                hosts=[f"http://{self._es_url}:{self._es_port}"],
            )
        
        # 设置 Elasticsearch 索引的分析器和相似度算法
        self._es_index_settings = {
            "analysis": {"analyzer": {"default": {"type": "standard"}}},
            "similarity": {
                "custom_bm25": {
                    "type": "BM25",
                    "k1": self._k1,
                    "b": self._b,
                }
            },
        }
        
        # 设置 Elasticsearch 索引的映射（mapping）
        self._es_mappings = {
            "properties": {
                "content": {
                    "type": "text",
                    "similarity": "custom_bm25",
                },
                "metadata": {
                    "type": "keyword",
                },
            }
        }
        
        # 如果索引不存在，则创建索引
        if not self._es_client.indices.exists(index=self._index_name):
            self._es_client.indices.create(
                index=self._index_name,
                mappings=self._es_mappings,
                settings=self._es_index_settings,
            )
        
        # 设置线程池执行器
        self._executor = executor or ThreadPoolExecutor()
    def load_document(self, chunks: List[Chunk]) -> List[str]:
        """Load document in elasticsearch.

        Args:
            chunks(List[Chunk]): document chunks.

        Return:
            List[str]: chunk ids.
        """
        try:
            from elasticsearch.helpers import bulk  # 导入 elasticsearch.helpers 模块中的 bulk 函数
        except ImportError:
            raise ValueError("Please install package `pip install elasticsearch`.")  # 如果导入失败，抛出导入错误异常
        es_requests = []  # 初始化空的 Elasticsearch 请求列表
        ids = []  # 初始化空的 chunk id 列表
        contents = [chunk.content for chunk in chunks]  # 提取每个 Chunk 对象的 content 属性
        metadatas = [json.dumps(chunk.metadata) for chunk in chunks]  # 将每个 Chunk 对象的 metadata 属性转换为 JSON 字符串
        chunk_ids = [chunk.chunk_id for chunk in chunks]  # 提取每个 Chunk 对象的 chunk_id 属性
        for i, content in enumerate(contents):
            es_request = {
                "_op_type": "index",  # 操作类型为索引操作
                "_index": self._index_name,  # 索引名称为对象中的 _index_name 属性
                "content": content,  # 文档内容
                "metadata": metadatas[i],  # 文档元数据
                "_id": chunk_ids[i],  # 文档 id
            }
            ids.append(chunk_ids[i])  # 将文档 id 添加到 ids 列表中
            es_requests.append(es_request)  # 将 Elasticsearch 请求添加到请求列表中
        bulk(self._es_client, es_requests)  # 使用 bulk 函数向 Elasticsearch 批量提交索引请求
        self._es_client.indices.refresh(index=self._index_name)  # 刷新指定索引的索引
        return ids  # 返回所有文档的 id 列表

    def similar_search(
        self, text: str, topk: int, filters: Optional[MetadataFilters] = None
    ) -> List[Chunk]:
        """Search similar text.

        Args:
            text(str): text.
            topk(int): topk.
            filters(MetadataFilters): filters.

        Return:
            List[Chunk]: similar text.
        """
        es_query = {"query": {"match": {"content": text}}}  # 构建基于文本内容的 Elasticsearch 查询
        res = self._es_client.search(index=self._index_name, body=es_query)  # 执行查询并获取结果

        chunks = []  # 初始化空的 Chunk 对象列表
        for r in res["hits"]["hits"]:
            chunks.append(
                Chunk(
                    chunk_id=r["_id"],  # 使用查询结果中的 _id 属性作为 Chunk 对象的 chunk_id
                    content=r["_source"]["content"],  # 使用查询结果中的 content 属性作为 Chunk 对象的 content
                    metadata=json.loads(r["_source"]["metadata"]),  # 将查询结果中的 metadata 属性解析为 JSON 对象并作为 Chunk 对象的 metadata
                )
            )
        return chunks[:topk]  # 返回前 topk 个相似的 Chunk 对象

    def similar_search_with_scores(
        self,
        text,
        top_k: int = 10,
        score_threshold: float = 0.3,
        filters: Optional[MetadataFilters] = None,
    ) -> List[Tuple[Chunk, float]]:
        """Search similar text with scores.

        Args:
            text: text to search.
            top_k: maximum number of results to return.
            score_threshold: minimum score threshold for results.
            filters: optional filters to apply.

        Return:
            List[Tuple[Chunk, float]]: list of (similar Chunk, score) tuples.
        """
        es_query = {"query": {"match": {"content": text}}}  # 构建基于文本内容的 Elasticsearch 查询
        res = self._es_client.search(index=self._index_name, body=es_query)  # 执行查询并获取结果

        results = []  # 初始化空的结果列表
        for r in res["hits"]["hits"]:
            chunk = Chunk(
                chunk_id=r["_id"],  # 使用查询结果中的 _id 属性作为 Chunk 对象的 chunk_id
                content=r["_source"]["content"],  # 使用查询结果中的 content 属性作为 Chunk 对象的 content
                metadata=json.loads(r["_source"]["metadata"]),  # 将查询结果中的 metadata 属性解析为 JSON 对象并作为 Chunk 对象的 metadata
            )
            score = r["_score"]  # 获取查询结果的分数
            if score >= score_threshold:  # 如果分数达到或超过阈值
                results.append((chunk, score))  # 将 Chunk 对象和分数作为元组添加到结果列表中
        return results[:top_k]  # 返回前 top_k 个相似的 Chunk 对象和分数的元组列表
    ) -> List[Chunk]:
        """Search similar text with scores.

        Args:
            text(str): 要搜索的文本.
            top_k(int): 返回的前k个结果.
            min_score(float): 最小分数阈值.
            filters(MetadataFilters): 元数据过滤器.

        Return:
            List[Tuple[str, float]]: 包含相似文本及其分数的列表.
        """
        # 构建Elasticsearch查询
        es_query = {"query": {"match": {"content": text}}}
        # 执行查询
        res = self._es_client.search(index=self._index_name, body=es_query)

        chunks_with_scores = []
        # 遍历查询结果
        for r in res["hits"]["hits"]:
            # 根据分数阈值过滤结果
            if r["_score"] >= score_threshold:
                # 将符合条件的结果封装为Chunk对象并加入列表
                chunks_with_scores.append(
                    Chunk(
                        chunk_id=r["_id"],
                        content=r["_source"]["content"],
                        metadata=json.loads(r["_source"]["metadata"]),
                        score=r["_score"],
                    )
                )
        # 如果指定了分数阈值但未找到匹配文档，则记录警告信息
        if score_threshold is not None and len(chunks_with_scores) == 0:
            logger.warning(
                "No relevant docs were retrieved using the relevance score"
                f" threshold {score_threshold}"
            )
        # 返回前top_k个符合条件的结果
        return chunks_with_scores[:top_k]

    async def aload_document(self, chunks: List[Chunk]) -> List[str]:
        """异步加载文档到Elasticsearch中.

        Args:
            chunks(List[Chunk]): 待加载的文档块.
        Return:
            List[str]: 加载成功的文档块ID列表.
        """
        # 调用异步函数将阻塞操作转换为异步执行
        return await blocking_func_to_async(self._executor, self.load_document, chunks)

    def delete_by_ids(self, ids: str) -> List[str]:
        """根据文档ID列表删除文档.

        Args:
            ids(List[str]): 待删除的文档ID列表.
        Return:
            List[str]: 返回已删除的文档ID列表.
        """
        # 将输入的ID字符串按逗号分隔为列表
        id_list = ids.split(",")
        # 构建批量删除的请求体
        bulk_body = [
            {"delete": {"_index": self._index_name, "_id": doc_id}}
            for doc_id in id_list
        ]
        # 执行批量删除操作
        self._es_client.bulk(body=bulk_body)
        # 返回已删除的文档ID列表
        return id_list

    def delete_vector_name(self, index_name: str):
        """根据索引名称删除索引.

        Args:
            index_name(str): 要删除的索引名称.
        """
        # 调用Elasticsearch客户端删除指定索引
        self._es_client.indices.delete(index=self._index_name)
```