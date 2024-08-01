# `.\DB-GPT-src\dbgpt\rag\retriever\db_schema.py`

```py
"""DBSchema retriever."""

from functools import reduce  # 导入 reduce 函数
from typing import List, Optional, cast  # 导入类型提示

from dbgpt.core import Chunk  # 导入 Chunk 类
from dbgpt.datasource.base import BaseConnector  # 导入 BaseConnector 类
from dbgpt.rag.index.base import IndexStoreBase  # 导入 IndexStoreBase 类
from dbgpt.rag.retriever.base import BaseRetriever  # 导入 BaseRetriever 类
from dbgpt.rag.retriever.rerank import DefaultRanker, Ranker  # 导入 DefaultRanker 和 Ranker 类
from dbgpt.rag.summary.rdbms_db_summary import _parse_db_summary  # 导入 _parse_db_summary 函数
from dbgpt.storage.vector_store.filters import MetadataFilters  # 导入 MetadataFilters 类
from dbgpt.util.chat_util import run_async_tasks  # 导入 run_async_tasks 函数
from dbgpt.util.executor_utils import blocking_func_to_async_no_executor  # 导入 blocking_func_to_async_no_executor 函数
from dbgpt.util.tracer import root_tracer  # 导入 root_tracer 函数

class DBSchemaRetriever(BaseRetriever):
    """DBSchema retriever."""

    def __init__(
        self,
        index_store: IndexStoreBase,
        top_k: int = 4,
        connector: Optional[BaseConnector] = None,
        query_rewrite: bool = False,
        rerank: Optional[Ranker] = None,
        **kwargs
    ):
        """Initialize DBSchemaRetriever instance.

        Args:
            index_store (IndexStoreBase): Index store for retrieval.
            top_k (int, optional): Number of top results to retrieve. Defaults to 4.
            connector (Optional[BaseConnector], optional): Connector for data retrieval. Defaults to None.
            query_rewrite (bool, optional): Flag indicating query rewriting. Defaults to False.
            rerank (Optional[Ranker], optional): Reranker object for result reranking. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(index_store, top_k, connector, query_rewrite, rerank, **kwargs)

    def _retrieve(
        self, query: str, filters: Optional[MetadataFilters] = None
    ) -> List[Chunk]:
        """Retrieve knowledge chunks.

        Args:
            query (str): Query text.
            filters (Optional[MetadataFilters], optional): Metadata filters. Defaults to None.

        Returns:
            List[Chunk]: List of retrieved chunks.
        """
        if self._need_embeddings:
            queries = [query]  # 创建包含查询文本的列表
            candidates = [
                self._index_store.similar_search(query, self._top_k, filters)
                for query in queries
            ]  # 使用索引存储对象进行相似搜索，返回候选结果列表
            return cast(List[Chunk], reduce(lambda x, y: x + y, candidates))  # 合并候选结果并转换为 Chunk 类型列表
        else:
            if not self._connector:
                raise RuntimeError("RDBMSConnector connection is required.")  # 如果连接器不存在，则抛出运行时错误
            table_summaries = _parse_db_summary(self._connector)  # 解析数据库摘要信息
            return [Chunk(content=table_summary) for table_summary in table_summaries]  # 构造 Chunk 对象列表

    def _retrieve_with_score(
        self,
        query: str,
        score_threshold: float,
        filters: Optional[MetadataFilters] = None,
    ) -> List[Chunk]:
        """Retrieve knowledge chunks with score.

        Args:
            query (str): Query text.
            score_threshold (float): Score threshold.
            filters (Optional[MetadataFilters], optional): Metadata filters. Defaults to None.

        Returns:
            List[Chunk]: List of retrieved chunks.
        """
        return self._retrieve(query, filters)  # 调用 _retrieve 方法进行知识块检索

    async def _aretrieve(
        self, query: str, filters: Optional[MetadataFilters] = None
    ) -> List[Chunk]:
        """Asynchronously retrieve knowledge chunks.

        Args:
            query (str): Query text.
            filters (Optional[MetadataFilters], optional): Metadata filters. Defaults to None.

        Returns:
            List[Chunk]: List of retrieved chunks.
        """
        return await blocking_func_to_async_no_executor(
            self._retrieve, query, filters
        )  # 转换为异步调用并返回检索结果
    async def _retrieve(
        self,
        query: str,
        filters: Optional[MetadataFilters] = None,
    ) -> List[Chunk]:
        """Retrieve knowledge chunks.

        Args:
            query (str): query text
            filters: metadata filters.

        Returns:
            List[Chunk]: list of chunks
        """
        # 如果需要计算嵌入向量
        if self._need_embeddings:
            # 将查询文本放入列表中
            queries = [query]
            # 进行相似性搜索，返回候选结果列表
            candidates = [
                self._similarity_search(
                    query, filters, root_tracer.get_current_span_id()
                )
                for query in queries
            ]
            # 异步运行所有候选结果的搜索任务
            result_candidates = await run_async_tasks(
                tasks=candidates, concurrency_limit=1
            )
            # 将所有候选结果合并成一个列表返回
            return cast(List[Chunk], reduce(lambda x, y: x + y, result_candidates))
        else:
            # 导入数据库摘要解析模块
            from dbgpt.rag.summary.rdbms_db_summary import (
                _parse_db_summary,
            )

            # 异步运行数据库摘要解析任务，返回摘要列表
            table_summaries = await run_async_tasks(
                tasks=[self._aparse_db_summary(root_tracer.get_current_span_id())],
                concurrency_limit=1,
            )
            # 将摘要列表中的每个摘要内容转换为Chunk对象，并返回列表
            return [
                Chunk(content=table_summary) for table_summary in table_summaries[0]
            ]

    async def _retrieve_with_score(
        self,
        query: str,
        score_threshold: float,
        filters: Optional[MetadataFilters] = None,
    ) -> List[Chunk]:
        """Retrieve knowledge chunks with score.

        Args:
            query (str): query text
            score_threshold (float): score threshold
            filters: metadata filters.
        """
        # 调用_retrieve方法来获取知识块列表，并返回结果
        return await self._retrieve(query, filters)

    async def _similarity_search(
        self,
        query,
        filters: Optional[MetadataFilters] = None,
        parent_span_id: Optional[str] = None,
    ) -> List[Chunk]:
        """Similar search."""
        # 使用跟踪器启动一个新的跟踪span，进行相似性搜索操作
        with root_tracer.start_span(
            "dbgpt.rag.retriever.db_schema._similarity_search",
            parent_span_id,
            metadata={"query": query},
        ):
            # 调用相似搜索函数，并异步执行，返回搜索结果列表
            return await blocking_func_to_async_no_executor(
                self._index_store.similar_search, query, self._top_k, filters
            )

    async def _aparse_db_summary(
        self, parent_span_id: Optional[str] = None
    ) -> List[str]:
        """Similar search."""
        # 导入数据库摘要解析函数
        from dbgpt.rag.summary.rdbms_db_summary import _parse_db_summary

        # 如果连接器未设置，则引发运行时错误
        if not self._connector:
            raise RuntimeError("RDBMSConnector connection is required.")
        # 使用跟踪器启动一个新的跟踪span，进行数据库摘要解析操作
        with root_tracer.start_span(
            "dbgpt.rag.retriever.db_schema._aparse_db_summary",
            parent_span_id,
        ):
            # 调用数据库摘要解析函数，并异步执行，返回解析结果列表
            return await blocking_func_to_async_no_executor(
                _parse_db_summary, self._connector
            )
```