# `.\DB-GPT-src\dbgpt\rag\retriever\time_weighted.py`

```py
"""Time weighted retriever."""

import datetime
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from dbgpt.core import Chunk  # 导入 Chunk 类
from dbgpt.rag.retriever.rerank import Ranker  # 导入 Ranker 类
from dbgpt.rag.retriever.rewrite import QueryRewrite  # 导入 QueryRewrite 类
from dbgpt.storage.vector_store.filters import MetadataFilters  # 导入 MetadataFilters 类

from ..index.base import IndexStoreBase  # 导入 IndexStoreBase 类
from .embedding import EmbeddingRetriever  # 导入 EmbeddingRetriever 类


def _get_hours_passed(time: datetime.datetime, ref_time: datetime.datetime) -> float:
    """Get the hours passed between two datetime objects."""
    return (time - ref_time).total_seconds() / 3600  # 计算两个时间对象之间的小时数


class TimeWeightedEmbeddingRetriever(EmbeddingRetriever):
    """Time weighted embedding retriever."""

    def __init__(
        self,
        index_store: IndexStoreBase,
        top_k: int = 100,
        query_rewrite: Optional[QueryRewrite] = None,
        rerank: Optional[Ranker] = None,
        decay_rate: float = 0.01,
    ):
        """Initialize TimeWeightedEmbeddingRetriever.

        Args:
            index_store (IndexStoreBase): vector store connector
            top_k (int): top k
            query_rewrite (Optional[QueryRewrite]): query rewrite
            rerank (Ranker): rerank
            decay_rate (float): decay rate for time weighting
        """
        super().__init__(
            index_store=index_store,
            top_k=top_k,
            query_rewrite=query_rewrite,
            rerank=rerank,
        )
        self.memory_stream: List[Chunk] = []  # 初始化内存流列表，用于存储 Chunk 对象
        self.other_score_keys: List[str] = []  # 初始化其他分数关键字列表
        self.decay_rate: float = decay_rate  # 初始化时间衰减率
        self.default_salience: Optional[float] = None  # 初始化默认显著性值
        self._top_k = top_k  # 设置 top_k 属性
        self._k = 4  # 设置私有属性 _k 为 4

    def load_document(self, chunks: List[Chunk], **kwargs: Dict[str, Any]) -> List[str]:
        """Load document in vector database.

        Args:
            chunks: document chunks.
            kwargs: additional keyword arguments.
        Return chunk ids.
        """
        current_time: Optional[datetime.datetime] = kwargs.get("current_time")  # 获取当前时间，如果未提供则使用当前时间
        if current_time is None:
            current_time = datetime.datetime.now()
        # 避免修改输入文档，创建文档副本列表
        dup_docs = [deepcopy(d) for d in chunks]
        for i, doc in enumerate(dup_docs):
            if doc.metadata.get("last_accessed_at") is None:
                doc.metadata["last_accessed_at"] = current_time  # 设置文档的最后访问时间为当前时间
            if "created_at" not in doc.metadata:
                doc.metadata["created_at"] = current_time  # 如果未设置创建时间，则设置为当前时间
            doc.metadata["buffer_idx"] = len(self.memory_stream) + i  # 设置文档在内存流中的索引位置
        self.memory_stream.extend(dup_docs)  # 将副本文档列表添加到内存流中
        return self._index_store.load_document(dup_docs)  # 调用 IndexStoreBase 对象的 load_document 方法加载文档

    def _retrieve(
        self, query: str, filters: Optional[MetadataFilters] = None
        """Retrieve documents matching the query with optional metadata filters."""
    ):
        # 方法未完整定义，此处应该继续编写方法的实现
    ) -> List[Chunk]:
        """Retrieve knowledge chunks.

        Args:
            query (str): query text
            filters: metadata filters.
        Return:
            List[Chunk]: list of chunks
        """
        # 获取当前时间
        current_time = datetime.datetime.now()
        
        # 从内存流中选择最近的 k 个文档，并为每个文档设置默认的重要性分数
        docs_and_scores = {
            doc.metadata["buffer_idx"]: (doc, self.default_salience)
            for doc in self.memory_stream[-self._k :]
        }
        
        # 更新文档的重要性分数，如果文档被认为是显著的
        docs_and_scores.update(self.get_salient_docs(query))
        
        # 对文档进行重新评分，结合重要性和时间因素
        rescored_docs = [
            (doc, self._get_combined_score(doc, relevance, current_time))
            for doc, relevance in docs_and_scores.values()
        ]
        
        # 根据综合分数对文档进行降序排序
        rescored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # 存储结果的列表
        result = []
        
        # 确保频繁访问的记忆不会被遗忘
        for doc, _ in rescored_docs[: self._k]:
            # 更新文档的最后访问时间
            buffered_doc = self.memory_stream[doc.metadata["buffer_idx"]]
            buffered_doc.metadata["last_accessed_at"] = current_time
            result.append(buffered_doc)
        
        # 返回结果列表
        return result

    def _get_combined_score(
        self,
        chunk: Chunk,
        vector_relevance: Optional[float],
        current_time: datetime.datetime,
    ) -> float:
        """Return the combined score for a document."""
        # 计算距离上次访问的时间（小时）
        hours_passed = _get_hours_passed(
            current_time,
            chunk.metadata["last_accessed_at"],
        )
        
        # 计算文档的综合分数，考虑时间衰减率和其他元数据因素
        score = (1.0 - self.decay_rate) ** hours_passed
        
        # 添加其他元数据因素的分数
        for key in self.other_score_keys:
            if key in chunk.metadata:
                score += chunk.metadata[key]
        
        # 如果有向量相关性分数，则加上该分数
        if vector_relevance is not None:
            score += vector_relevance
        
        # 返回文档的综合分数
        return score

    def get_salient_docs(self, query: str) -> Dict[int, Tuple[Chunk, float]]:
        """Return documents that are salient to the query."""
        # 使用索引存储执行与分数的相似搜索
        docs_and_scores: List[Chunk]
        docs_and_scores = self._index_store.similar_search_with_scores(
            query, topk=self._top_k, score_threshold=0
        )
        
        # 构建结果字典，包含文档缓冲区索引到文档和其分数的元组
        results = {}
        for ck in docs_and_scores:
            if "buffer_idx" in ck.metadata:
                buffer_idx = ck.metadata["buffer_idx"]
                doc = self.memory_stream[buffer_idx]
                results[buffer_idx] = (doc, ck.score)
        
        # 返回与查询相关的显著文档的结果字典
        return results
```