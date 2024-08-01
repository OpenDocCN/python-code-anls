# `.\DB-GPT-src\dbgpt\rag\retriever\bm25.py`

```py
"""
BM25 retriever.
"""
# 导入所需模块
import json
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import Any, List, Optional

# 导入自定义模块
from dbgpt.app.base import logger
from dbgpt.core import Chunk
from dbgpt.rag.retriever.base import BaseRetriever
from dbgpt.rag.retriever.rerank import DefaultRanker, Ranker
from dbgpt.rag.retriever.rewrite import QueryRewrite
from dbgpt.storage.vector_store.filters import MetadataFilters
from dbgpt.util.executor_utils import blocking_func_to_async

# 定义 BM25Retriever 类，继承自 BaseRetriever
class BM25Retriever(BaseRetriever):
    """
    BM25 retriever.

    refer https://www.elastic.co/guide/en/elasticsearch/reference/8.9/
    index-modules-similarity.html;
    TF/IDF based similarity that has built-in tf normalization and is supposed to
    work better for short fields (like names). See Okapi_BM25 for more details.
    """

    # 初始化方法，接受多个可选参数和默认值
    def __init__(
        self,
        top_k: int = 4,
        es_index: str = "dbgpt",
        es_client: Any = None,
        query_rewrite: Optional[QueryRewrite] = None,
        rerank: Optional[Ranker] = None,
        k1: Optional[float] = 2.0,
        b: Optional[float] = 0.75,
        executor: Optional[Executor] = None,
        """Create BM25Retriever.

        Args:
            top_k (int): top k results to retrieve
            es_index (str): name of the Elasticsearch index to use
            es_client (Any): client object for interacting with Elasticsearch
            query_rewrite (Optional[QueryRewrite]): optional query rewriting component
            rerank (Ranker): reranking strategy to use
            k1 (Optional[float]): controls non-linear term frequency normalization (saturation), default is 2.0
            b (Optional[float]): controls document length normalization in TF values, default is 0.75
            executor (Optional[Executor]): executor for handling concurrent tasks

        Returns:
            BM25Retriever: initialized BM25 retriever instance
        """
        super().__init__()
        self._top_k = top_k  # Initialize top_k attribute with the provided value
        self._query_rewrite = query_rewrite  # Store query rewriting component
        try:
            from elasticsearch import Elasticsearch  # Attempt to import Elasticsearch module
        except ImportError:
            raise ImportError(
                "please install elasticsearch using `pip install elasticsearch`"
            )  # Raise ImportError if Elasticsearch module is not found
        self._es_client: Elasticsearch = es_client  # Assign Elasticsearch client to instance variable

        self._es_mappings = {
            "properties": {
                "content": {
                    "type": "text",
                    "similarity": "custom_bm25",
                }
            }
        }  # Define Elasticsearch index mappings for 'content' field with custom BM25 similarity

        self._es_index_settings = {
            "analysis": {"analyzer": {"default": {"type": "standard"}}},
            "similarity": {
                "custom_bm25": {
                    "type": "BM25",
                    "k1": k1,
                    "b": b,
                }
            },
        }  # Define Elasticsearch index settings including custom BM25 similarity parameters

        self._index_name = es_index  # Store the Elasticsearch index name

        if not self._es_client.indices.exists(index=self._index_name):
            self._es_client.indices.create(
                index=self._index_name,
                mappings=self._es_mappings,
                settings=self._es_index_settings,
            )  # Create the Elasticsearch index if it does not exist

        self._rerank = rerank or DefaultRanker(self._top_k)  # Assign the provided rerank strategy or default
        self._executor = executor or ThreadPoolExecutor()  # Assign the provided executor or default ThreadPoolExecutor

    def _retrieve(
        self, query: str, filters: Optional[MetadataFilters] = None
    ) -> List[Chunk]:
        """Retrieve knowledge chunks.

        Args:
            query (str): query text to search for
            filters: optional filters to apply to the query results

        Return:
            List[Chunk]: list of retrieved knowledge chunks
        """
        es_query = {"query": {"match": {"content": query}}}  # Define Elasticsearch query for matching content field
        res = self._es_client.search(index=self._index_name, body=es_query)  # Execute search query

        chunks = []
        for r in res["hits"]["hits"]:
            chunks.append(
                Chunk(
                    chunk_id=r["_id"],  # Extract document ID as chunk_id
                    content=r["_source"]["content"],  # Extract content from document source
                    metadata=json.loads(r["_source"]["metadata"]),  # Extract metadata from document source
                )
            )  # Create Chunk objects and append to chunks list
        return chunks[: self._top_k]  # Return top-k chunks based on the retrieval limit

    def _retrieve_with_score(
        self,
        query: str,
        score_threshold: float,
        filters: Optional[MetadataFilters] = None,
    async def _aretrieve_with_score(
        self,
        query: str,
        score_threshold: float,
        filters: Optional[MetadataFilters] = None,
    ) -> List[Chunk]:
        """Retrieve knowledge chunks with score.

        Args:
            query (str): 查询文本
            score_threshold (float): 分数阈值
            filters: 元数据过滤器
        Return:
            List[Chunk]: 带有分数的片段列表
        """
        # 使用 Elasticsearch 查询语法构造查询对象
        es_query = {"query": {"match": {"content": query}}}
        # 发送查询请求到 Elasticsearch，获取查询结果
        res = self._es_client.search(index=self._index_name, body=es_query)

        chunks_with_scores = []
        # 遍历查询结果中的每个文档
        for r in res["hits"]["hits"]:
            # 检查文档的分数是否大于等于指定的分数阈值
            if r["_score"] >= score_threshold:
                # 如果符合条件，则创建 Chunk 对象并添加到结果列表中
                chunks_with_scores.append(
                    Chunk(
                        chunk_id=r["_id"],
                        content=r["_source"]["content"],
                        metadata=json.loads(r["_source"]["metadata"]),
                        score=r["_score"],
                    )
                )
        # 如果指定了分数阈值且未检索到符合条件的文档，记录警告日志
        if score_threshold is not None and len(chunks_with_scores) == 0:
            logger.warning(
                "No relevant docs were retrieved using the relevance score"
                f" threshold {score_threshold}"
            )
        # 返回按照分数排序后的前 self._top_k 个文档
        return chunks_with_scores[: self._top_k]
```