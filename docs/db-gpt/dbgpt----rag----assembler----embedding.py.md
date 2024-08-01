# `.\DB-GPT-src\dbgpt\rag\assembler\embedding.py`

```py
"""Embedding Assembler."""
# 导入必要的库和模块
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional

from dbgpt.core import Chunk, Embeddings  # 导入核心功能模块

from ...util.executor_utils import blocking_func_to_async  # 导入异步执行工具
from ..assembler.base import BaseAssembler  # 导入基础汇编器类
from ..chunk_manager import ChunkParameters  # 导入块管理器参数类
from ..index.base import IndexStoreBase  # 导入基础索引存储类
from ..knowledge.base import Knowledge  # 导入知识基础类
from ..retriever import BaseRetriever, RetrieverStrategy  # 导入检索器相关类
from ..retriever.embedding import EmbeddingRetriever  # 导入嵌入式检索器类


class EmbeddingAssembler(BaseAssembler):
    """Embedding Assembler.

    Example:
    .. code-block:: python

        from dbgpt.rag.assembler import EmbeddingAssembler

        pdf_path = "path/to/document.pdf"
        knowledge = KnowledgeFactory.from_file_path(pdf_path)
        assembler = EmbeddingAssembler.load_from_knowledge(
            knowledge=knowledge,
            embedding_model="text2vec",
        )
    """

    def __init__(
        self,
        knowledge: Knowledge,
        index_store: IndexStoreBase,
        chunk_parameters: Optional[ChunkParameters] = None,
        retrieve_strategy: Optional[RetrieverStrategy] = RetrieverStrategy.EMBEDDING,
        **kwargs: Any,
    ) -> None:
        """Initialize with Embedding Assembler arguments.

        Args:
            knowledge: (Knowledge) Knowledge datasource.
            index_store: (IndexStoreBase) IndexStoreBase to use.
            chunk_parameters: (Optional[ChunkParameters]) ChunkManager to use for
                chunking.
            keyword_store: (Optional[IndexStoreBase]) IndexStoreBase to use.
            embedding_model: (Optional[str]) Embedding model to use.
            embeddings: (Optional[Embeddings]) Embeddings to use.
        """
        if knowledge is None:
            raise ValueError("knowledge datasource must be provided.")
        self._index_store = index_store  # 设置索引存储
        self._retrieve_strategy = retrieve_strategy  # 设置检索策略

        super().__init__(
            knowledge=knowledge,
            chunk_parameters=chunk_parameters,
            **kwargs,
        )

    @classmethod
    def load_from_knowledge(
        cls,
        knowledge: Knowledge,
        index_store: IndexStoreBase,
        chunk_parameters: Optional[ChunkParameters] = None,
        embedding_model: Optional[str] = None,
        embeddings: Optional[Embeddings] = None,
        retrieve_strategy: Optional[RetrieverStrategy] = RetrieverStrategy.EMBEDDING,
        **kwargs: Any,
    ) -> 'EmbeddingAssembler':
        """Load an instance of EmbeddingAssembler from knowledge and related parameters.

        Args:
            knowledge: (Knowledge) Knowledge datasource.
            index_store: (IndexStoreBase) IndexStoreBase to use.
            chunk_parameters: (Optional[ChunkParameters]) ChunkManager to use for chunking.
            embedding_model: (Optional[str]) Embedding model to use.
            embeddings: (Optional[Embeddings]) Embeddings to use.
            retrieve_strategy: (Optional[RetrieverStrategy]) Retrieval strategy to use.

        Returns:
            EmbeddingAssembler: Instance of EmbeddingAssembler initialized with provided parameters.
        """
        # 创建并返回一个从知识和相关参数加载的EmbeddingAssembler实例
        pass  # 这里是占位符，实际实现时需要填充逻辑
    ) -> "EmbeddingAssembler":
        """
        返回一个 EmbeddingAssembler 类的实例。

        Args:
            knowledge: (Knowledge) 知识数据源。
            index_store: (IndexStoreBase) 要使用的索引存储。
            chunk_parameters: (Optional[ChunkParameters]) 用于分块的 ChunkManager。
            embedding_model: (Optional[str]) 要使用的嵌入模型。
            embeddings: (Optional[Embeddings]) 要使用的嵌入。
            retrieve_strategy: (Optional[RetrieverStrategy]) 检索策略。

        Returns:
             EmbeddingAssembler 类的实例
        """
        return cls(
            knowledge=knowledge,
            index_store=index_store,
            chunk_parameters=chunk_parameters,
            embedding_model=embedding_model,
            embeddings=embeddings,
            retrieve_strategy=retrieve_strategy,
        )

    @classmethod
    async def aload_from_knowledge(
        cls,
        knowledge: Knowledge,
        index_store: IndexStoreBase,
        chunk_parameters: Optional[ChunkParameters] = None,
        executor: Optional[ThreadPoolExecutor] = None,
        retrieve_strategy: Optional[RetrieverStrategy] = RetrieverStrategy.EMBEDDING,
    ) -> "EmbeddingAssembler":
        """
        从知识数据源加载文档嵌入到向量存储中。

        Args:
            knowledge: (Knowledge) 知识数据源。
            index_store: (IndexStoreBase) 要使用的索引存储。
            chunk_parameters: (Optional[ChunkParameters]) 用于分块的 ChunkManager。
            executor: (Optional[ThreadPoolExecutor) 要使用的 ThreadPoolExecutor。
            retrieve_strategy: (Optional[RetrieverStrategy]) 检索策略。

        Returns:
             EmbeddingAssembler 类的实例，通过异步加载返回。
        """
        executor = executor or ThreadPoolExecutor()
        return await blocking_func_to_async(
            executor,
            cls,
            knowledge,
            index_store,
            chunk_parameters,
            retrieve_strategy,
        )

    def persist(self, **kwargs) -> List[str]:
        """
        将块持久化到存储中。

        Returns:
            List[str]: 块 ID 的列表。
        """
        return self._index_store.load_document(self._chunks)

    async def apersist(self, **kwargs) -> List[str]:
        """
        将块异步持久化到存储中。

        Returns:
            List[str]: 块 ID 的列表。
        """
        # 将块持久化到向量存储中
        return await self._index_store.aload_document(self._chunks)

    def _extract_info(self, chunks) -> List[Chunk]:
        """
        从块中提取信息。

        Args:
            chunks: 块列表。

        Returns:
            List[Chunk]: 从块中提取的信息列表。
        """
        return []
    # 定义一个方法用于生成检索器对象，返回一个嵌入式检索器
    def as_retriever(self, top_k: int = 4, **kwargs) -> BaseRetriever:
        """Create a retriever.

        Args:
            top_k(int): default 4.  # 设置默认的检索结果返回数目为4

        Returns:
            EmbeddingRetriever  # 返回一个嵌入式检索器对象
        """
        # 创建并返回一个嵌入式检索器对象，设置参数包括检索的结果数目、索引存储和检索策略
        return EmbeddingRetriever(
            top_k=top_k,
            index_store=self._index_store,
            retrieve_strategy=self._retrieve_strategy,
        )
```