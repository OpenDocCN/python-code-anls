# `.\DB-GPT-src\dbgpt\rag\retriever\embedding.py`

```py
"""Embedding retriever."""

from functools import reduce
from typing import Any, Dict, List, Optional, cast

from dbgpt.core import Chunk
from dbgpt.rag.index.base import IndexStoreBase
from dbgpt.rag.retriever.base import BaseRetriever, RetrieverStrategy
from dbgpt.rag.retriever.rerank import DefaultRanker, Ranker
from dbgpt.rag.retriever.rewrite import QueryRewrite
from dbgpt.storage.vector_store.filters import MetadataFilters
from dbgpt.util.chat_util import run_async_tasks
from dbgpt.util.executor_utils import blocking_func_to_async_no_executor
from dbgpt.util.tracer import root_tracer


class EmbeddingRetriever(BaseRetriever):
    """Embedding retriever."""

    def __init__(
        self,
        index_store: IndexStoreBase,
        top_k: int = 4,
        query_rewrite: Optional[QueryRewrite] = None,
        rerank: Optional[Ranker] = None,
        retrieve_strategy: Optional[RetrieverStrategy] = RetrieverStrategy.EMBEDDING,
    ):
        """Create EmbeddingRetriever.

        Args:
            index_store(IndexStore): vector store connector
            top_k (int): top k
            query_rewrite (Optional[QueryRewrite]): query rewrite
            rerank (Ranker): rerank

        Examples:
            .. code-block:: python

                from dbgpt.storage.vector_store.connector import VectorStoreConnector
                from dbgpt.storage.vector_store.chroma_store import ChromaVectorConfig
                from dbgpt.rag.retriever.embedding import EmbeddingRetriever
                from dbgpt.rag.embedding.embedding_factory import (
                    DefaultEmbeddingFactory,
                )

                embedding_factory = DefaultEmbeddingFactory()
                from dbgpt.rag.retriever.embedding import EmbeddingRetriever
                from dbgpt.storage.vector_store.connector import VectorStoreConnector

                embedding_fn = embedding_factory.create(
                    model_name=EMBEDDING_MODEL_CONFIG[CFG.EMBEDDING_MODEL]
                )
                vector_name = "test"
                config = ChromaVectorConfig(name=vector_name, embedding_fn=embedding_fn)
                vector_store_connector = VectorStoreConnector(
                    vector_store_type="Chroma",
                    vector_store_config=config,
                )
                embedding_retriever = EmbeddingRetriever(
                    top_k=3, vector_store_connector=vector_store_connector
                )
                chunks = embedding_retriever.retrieve("your query text")
                print(
                    f"embedding retriever results:{[chunk.content for chunk in chunks]}"
                )
        """
        # 初始化 EmbeddingRetriever 类的实例
        self._top_k = top_k  # 设置检索结果的 top k 值
        self._query_rewrite = query_rewrite  # 设置查询重写器
        self._index_store = index_store  # 设置索引存储器
        self._rerank = rerank or DefaultRanker(self._top_k)  # 设置重新排序器，如果未提供，则使用默认的 DefaultRanker
        self._retrieve_strategy = retrieve_strategy  # 设置检索策略，默认为 EMBEDDING
    # 加载文档到向量数据库中
    def load_document(self, chunks: List[Chunk], **kwargs: Dict[str, Any]) -> List[str]:
        """Load document in vector database.

        Args:
            chunks (List[Chunk]): document chunks.
        Return:
            List[str]: chunk ids.
        """
        return self._index_store.load_document(chunks)

    # 检索知识片段
    def _retrieve(
        self, query: str, filters: Optional[MetadataFilters] = None
    ) -> List[Chunk]:
        """Retrieve knowledge chunks.

        Args:
            query (str): query text
            filters: metadata filters.
        Return:
            List[Chunk]: list of chunks
        """
        # 将查询文本封装成列表
        queries = [query]
        # 对每个查询执行相似搜索，返回候选的片段列表
        candidates = [
            self._index_store.similar_search(query, self._top_k, filters)
            for query in queries
        ]
        # 将多个候选列表合并为一个列表
        res_candidates = cast(List[Chunk], reduce(lambda x, y: x + y, candidates))
        return res_candidates

    # 带有分数的知识片段检索
    def _retrieve_with_score(
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
        Return:
            List[Chunk]: list of chunks with score
        """
        # 将查询文本封装成列表
        queries = [query]
        # 对每个查询执行带有分数的相似搜索，返回候选的片段列表及其分数
        candidates_with_score = [
            self._index_store.similar_search_with_scores(
                query, self._top_k, score_threshold, filters
            )
            for query in queries
        ]
        # 将多个候选列表合并为一个列表
        new_candidates_with_score = cast(
            List[Chunk], reduce(lambda x, y: x + y, candidates_with_score)
        )
        # 使用重新排序器对候选列表进行重新排序
        new_candidates_with_score = self._rerank.rank(new_candidates_with_score, query)
        return new_candidates_with_score

    # 异步知识片段检索
    async def _aretrieve(
        self, query: str, filters: Optional[MetadataFilters] = None
    ) -> List[Chunk]:
        """Asynchronously retrieve knowledge chunks.

        Args:
            query (str): query text
            filters: metadata filters.
        Return:
            List[Chunk]: list of chunks
        """
        # 将查询文本封装成列表
        queries = [query]
        # 对每个查询执行相似搜索，返回候选的片段列表
        candidates = [
            await self._index_store.asimilar_search(query, self._top_k, filters)
            for query in queries
        ]
        # 将多个候选列表合并为一个列表
        res_candidates = cast(List[Chunk], reduce(lambda x, y: x + y, candidates))
        return res_candidates
        """Retrieve knowledge chunks.

        Args:
            query (str): query text.
            filters: metadata filters.
        Return:
            List[Chunk]: list of chunks
        """
        # 将查询文本放入列表中
        queries = [query]
        
        # 如果启用了查询重写功能
        if self._query_rewrite:
            # 创建候选任务列表，每个任务调用相似性搜索方法
            candidates_tasks = [
                self._similarity_search(
                    query, filters, root_tracer.get_current_span_id()
                )
                for query in queries
            ]
            # 异步运行候选任务并获取结果块
            chunks = await self._run_async_tasks(candidates_tasks)
            # 将所有结果块的内容合并为一个上下文字符串
            context = "\n".join([chunk.content for chunk in chunks])
            # 使用查询重写对象重写原始查询，生成新的查询列表
            new_queries = await self._query_rewrite.rewrite(
                origin_query=query, context=context, nums=1
            )
            # 将新生成的查询列表加入原始查询列表中
            queries.extend(new_queries)
        
        # 创建候选搜索任务列表，每个任务调用相似性搜索方法
        candidates = [
            self._similarity_search(query, filters, root_tracer.get_current_span_id())
            for query in queries
        ]
        # 异步运行所有候选搜索任务并获取结果
        new_candidates = await run_async_tasks(tasks=candidates, concurrency_limit=1)
        # 返回最终的搜索结果列表
        return new_candidates
    ) -> List[Chunk]:
        """Retrieve knowledge chunks with score.

        Args:
            query (str): query text
            score_threshold (float): score threshold
            filters: metadata filters.
        Return:
            List[Chunk]: list of chunks with score
        """
        # 将查询文本放入列表中
        queries = [query]
        # 如果启用了查询重写功能
        if self._query_rewrite:
            # 开始一个新的跟踪 span，用于相似性搜索的查询重写过程
            with root_tracer.start_span(
                "dbgpt.rag.retriever.embeddings.query_rewrite.similarity_search",
                metadata={"query": query, "score_threshold": score_threshold},
            ):
                # 为每个查询启动相似性搜索任务
                candidates_tasks = [
                    self._similarity_search(
                        query, filters, root_tracer.get_current_span_id()
                    )
                    for query in queries
                ]
                # 异步运行所有相似性搜索任务并获取结果
                chunks = await self._run_async_tasks(candidates_tasks)
                # 将搜索到的文本片段内容连接成一个上下文字符串
                context = "\n".join([chunk.content for chunk in chunks])
            # 开始一个新的跟踪 span，用于查询重写过程
            with root_tracer.start_span(
                "dbgpt.rag.retriever.embeddings.query_rewrite.rewrite",
                metadata={"query": query, "context": context, "nums": 1},
            ):
                # 使用查询重写模块对原始查询进行重写，得到新的查询列表
                new_queries = await self._query_rewrite.rewrite(
                    origin_query=query, context=context, nums=1
                )
                # 将新的查询列表添加到原始查询列表中
                queries.extend(new_queries)

        # 开始一个新的跟踪 span，用于带有分数的相似性搜索
        with root_tracer.start_span(
            "dbgpt.rag.retriever.embeddings.similarity_search_with_score",
            metadata={"query": query, "score_threshold": score_threshold},
        ):
            # 为每个查询启动带分数的相似性搜索任务
            candidates_with_score = [
                self._similarity_search_with_score(
                    query, score_threshold, filters, root_tracer.get_current_span_id()
                )
                for query in queries
            ]
            # 异步运行所有带分数的相似性搜索任务并获取结果
            res_candidates_with_score = await run_async_tasks(
                tasks=candidates_with_score, concurrency_limit=1
            )
            # 将所有搜索到的文本片段合并成一个列表
            new_candidates_with_score = cast(
                List[Chunk], reduce(lambda x, y: x + y, res_candidates_with_score)
            )

        # 开始一个新的跟踪 span，用于重新排序候选文本片段
        with root_tracer.start_span(
            "dbgpt.rag.retriever.embeddings.rerank",
            metadata={
                "query": query,
                "score_threshold": score_threshold,
                "rerank_cls": self._rerank.__class__.__name__,
            },
        ):
            # 使用重新排序模块对带分数的文本片段列表进行重新排序
            new_candidates_with_score = await self._rerank.arank(
                new_candidates_with_score, query
            )
            # 返回重新排序后的带分数文本片段列表作为最终结果
            return new_candidates_with_score

    async def _similarity_search(
        self,
        query,
        filters: Optional[MetadataFilters] = None,
        parent_span_id: Optional[str] = None,
    async def _run_async_tasks(self, tasks) -> List[Chunk]:
        """Run async tasks."""
        # 使用给定的任务列表并发执行异步任务，限制并发数为1
        candidates = await run_async_tasks(tasks=tasks, concurrency_limit=1)
        # 将多个任务返回的候选结果列表合并为一个列表
        candidates = reduce(lambda x, y: x + y, candidates)
        return cast(List[Chunk], candidates)

    async def _similarity_search_with_score(
        self,
        query,
        score_threshold,
        filters: Optional[MetadataFilters] = None,
        parent_span_id: Optional[str] = None,
    ) -> List[Chunk]:
        """Similar search with score."""
        # 使用分布式追踪工具开始一个名为
        # "dbgpt.rag.retriever.embeddings._do_similarity_search_with_score"的跟踪span，
        # 设置查询(query)和得分阈值(score_threshold)为元数据
        with root_tracer.start_span(
            "dbgpt.rag.retriever.embeddings._do_similarity_search_with_score",
            parent_span_id,
            metadata={
                "query": query,
                "score_threshold": score_threshold,
            },
        ):
            # 调用索引存储对象的带得分的相似性搜索方法，
            # 返回符合条件的前k个结果的列表
            return await self._index_store.asimilar_search_with_scores(
                query, self._top_k, score_threshold, filters
            )
```