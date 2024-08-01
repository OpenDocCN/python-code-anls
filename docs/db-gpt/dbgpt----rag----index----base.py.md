# `.\DB-GPT-src\dbgpt\rag\index\base.py`

```py
"""Index store base class."""
# 导入日志模块
import logging
# 导入时间模块
import time
# 导入抽象基类模块
from abc import ABC, abstractmethod
# 导入并发执行器相关模块
from concurrent.futures import Executor, ThreadPoolExecutor
# 导入类型提示相关模块
from typing import Any, Dict, List, Optional

# 导入Pydantic相关模块
from dbgpt._private.pydantic import BaseModel, ConfigDict, Field, model_to_dict
# 导入核心模块中的Chunk和Embeddings类
from dbgpt.core import Chunk, Embeddings
# 导入向量存储中的MetadataFilters类
from dbgpt.storage.vector_store.filters import MetadataFilters
# 导入执行器工具函数
from dbgpt.util.executor_utils import (
    blocking_func_to_async,
    blocking_func_to_async_no_executor,
)

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


class IndexStoreConfig(BaseModel):
    """Index store config."""

    # 模型配置字典
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    # 索引存储的名称，默认为"dbgpt_collection"
    name: str = Field(
        default="dbgpt_collection",
        description="The name of index store, if not set, will use the default name.",
    )
    # 向量存储的嵌入函数，默认为None
    embedding_fn: Optional[Embeddings] = Field(
        default=None,
        description="The embedding function of vector store, if not set, will use the "
        "default embedding function.",
    )
    # 一次加载的最大文档块数，默认为10
    max_chunks_once_load: int = Field(
        default=10,
        description="The max number of chunks to load at once. If your document is "
        "large, you can set this value to a larger number to speed up the loading "
        "process. Default is 10.",
    )
    # 最大线程数，默认为1
    max_threads: int = Field(
        default=1,
        description="The max number of threads to use. Default is 1. If you set this "
        "bigger than 1, please make sure your vector store is thread-safe.",
    )

    def to_dict(self, **kwargs) -> Dict[str, Any]:
        """Convert to dict."""
        # 将配置对象转换为字典形式
        return model_to_dict(self, **kwargs)


class IndexStoreBase(ABC):
    """Index store base class."""

    def __init__(self, executor: Optional[Executor] = None):
        """Init index store."""
        # 初始化索引存储，设置执行器，默认使用线程池执行器
        self._executor = executor or ThreadPoolExecutor()

    @abstractmethod
    def load_document(self, chunks: List[Chunk]) -> List[str]:
        """Load document in index database.

        Args:
            chunks(List[Chunk]): document chunks.

        Return:
            List[str]: chunk ids.
        """
        # 抽象方法，加载文档到索引数据库，返回加载的文档块的ID列表

    @abstractmethod
    async def aload_document(self, chunks: List[Chunk]) -> List[str]:
        """Load document in index database asynchronously.

        Args:
            chunks(List[Chunk]): document chunks.

        Return:
            List[str]: chunk ids.
        """
        # 抽象方法，异步加载文档到索引数据库，返回加载的文档块的ID列表

    @abstractmethod
    def similar_search_with_scores(
        self,
        text,
        topk,
        score_threshold: float,
        filters: Optional[MetadataFilters] = None,
        ```
    ) -> List[Chunk]:
        """Similar search with scores in index database.

        Args:
            text(str): The query text.
            topk(int): The number of similar documents to return.
            score_threshold(int): score_threshold: Optional, a floating point value
                between 0 to 1
            filters(Optional[MetadataFilters]): metadata filters.
        Return:
            List[Chunk]: The similar documents.
        """
        # 定义一个抽象方法，用于在索引数据库中执行类似的搜索操作，返回相似文档列表

    @abstractmethod
    def delete_by_ids(self, ids: str) -> List[str]:
        """Delete docs.

        Args:
            ids(str): The vector ids to delete, separated by comma.
        """
        # 定义一个抽象方法，用于根据文档 ID 删除文档

    @abstractmethod
    def delete_vector_name(self, index_name: str):
        """Delete index by name.

        Args:
            index_name(str): The name of index to delete.
        """
        # 定义一个抽象方法，用于根据索引名称删除索引

    def vector_name_exists(self) -> bool:
        """Whether name exists."""
        # 检查索引名称是否存在
        return True

    def load_document_with_limit(
        self, chunks: List[Chunk], max_chunks_once_load: int = 10, max_threads: int = 1
    ) -> List[str]:
        """Load document in index database with specified limit.

        Args:
            chunks(List[Chunk]): Document chunks.
            max_chunks_once_load(int): Max number of chunks to load at once.
            max_threads(int): Max number of threads to use.

        Return:
            List[str]: Chunk ids.
        """
        # 将文档分组成指定大小的块
        chunk_groups = [
            chunks[i : i + max_chunks_once_load]
            for i in range(0, len(chunks), max_chunks_once_load)
        ]
        # 记录日志，显示加载的文档块数量、组数和使用的线程数
        logger.info(
            f"Loading {len(chunks)} chunks in {len(chunk_groups)} groups with "
            f"{max_threads} threads."
        )
        ids = []
        loaded_cnt = 0
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            tasks = []
            for chunk_group in chunk_groups:
                tasks.append(executor.submit(self.load_document, chunk_group))
            for future in tasks:
                success_ids = future.result()
                ids.extend(success_ids)
                loaded_cnt += len(success_ids)
                logger.info(f"Loaded {loaded_cnt} chunks, total {len(chunks)} chunks.")
        # 记录加载完成的总时间
        logger.info(
            f"Loaded {len(chunks)} chunks in {time.time() - start_time} seconds"
        )
        return ids

    async def aload_document_with_limit(
        self, chunks: List[Chunk], max_chunks_once_load: int = 10, max_threads: int = 1
    ):
        """Asynchronously load document in index database with specified limit.

        Args:
            chunks(List[Chunk]): Document chunks.
            max_chunks_once_load(int): Max number of chunks to load at once.
            max_threads(int): Max number of threads to use.
        """
        # 异步加载文档到索引数据库，带有指定的加载限制
    ) -> List[str]:
        """
        Load document in index database with specified limit.

        Args:
            chunks(List[Chunk]): Document chunks.
            max_chunks_once_load(int): Max number of chunks to load at once.
            max_threads(int): Max number of threads to use.

        Returns:
            List[str]: Chunk ids.
        """
        # 使用异步函数将阻塞函数转换为异步调用，传入执行器和加载文档限制函数及其参数
        return await blocking_func_to_async(
            self._executor,
            self.load_document_with_limit,
            chunks,
            max_chunks_once_load,
            max_threads,
        )

    def similar_search(
        self, text: str, topk: int, filters: Optional[MetadataFilters] = None
    ) -> List[Chunk]:
        """
        Similar search in index database.

        Args:
            text(str): The query text.
            topk(int): The number of similar documents to return.
            filters(Optional[MetadataFilters]): metadata filters.

        Returns:
            List[Chunk]: The similar documents.
        """
        # 调用带分数的相似搜索函数，设置默认相似度阈值为1.0
        return self.similar_search_with_scores(text, topk, 1.0, filters)

    async def asimilar_search_with_scores(
        self,
        doc: str,
        topk: int,
        score_threshold: float,
        filters: Optional[MetadataFilters] = None,
    ) -> List[Chunk]:
        """
        Async similar_search_with_score in vector database.

        Args:
            doc(str): The query document.
            topk(int): The number of similar documents to return.
            score_threshold(float): The similarity score threshold.
            filters(Optional[MetadataFilters]): metadata filters.

        Returns:
            List[Chunk]: The similar documents with scores.
        """
        # 使用无执行器的阻塞函数转换为异步调用，传入相似搜索函数及其参数
        return await blocking_func_to_async_no_executor(
            self.similar_search_with_scores, doc, topk, score_threshold, filters
        )
```