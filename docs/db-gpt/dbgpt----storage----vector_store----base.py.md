# `.\DB-GPT-src\dbgpt\storage\vector_store\base.py`

```py
"""Vector store base class."""
# 导入日志模块
import logging
# 导入数学模块
import math
# 导入抽象基类模块
from abc import ABC, abstractmethod
# 导入线程池执行器
from concurrent.futures import ThreadPoolExecutor
# 导入类型提示模块
from typing import Any, List, Optional

# 导入配置字典和字段模块
from dbgpt._private.pydantic import ConfigDict, Field
# 导入核心模块：分块、嵌入
from dbgpt.core import Chunk, Embeddings
# 导入AWEL流模块中的参数类
from dbgpt.core.awel.flow import Parameter
# 导入RAG索引基类模块和配置模块
from dbgpt.rag.index.base import IndexStoreBase, IndexStoreConfig
# 导入向量存储过滤器模块
from dbgpt.storage.vector_store.filters import MetadataFilters
# 导入执行器工具模块中的阻塞函数转异步函数工具
from dbgpt.util.executor_utils import blocking_func_to_async
# 导入国际化工具模块
from dbgpt.util.i18n_utils import _

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# 定义常见参数列表
_COMMON_PARAMETERS = [
    # 构建参数对象，描述集合名称
    Parameter.build_from(
        _("Collection Name"),
        "name",
        str,
        description=_(
            "The name of vector store, if not set, will use the default name."
        ),
        optional=True,
        default="dbgpt_collection",
    ),
    # 构建参数对象，描述用户
    Parameter.build_from(
        _("User"),
        "user",
        str,
        description=_(
            "The user of vector store, if not set, will use the default user."
        ),
        optional=True,
        default=None,
    ),
    # 构建参数对象，描述密码
    Parameter.build_from(
        _("Password"),
        "password",
        str,
        description=_(
            "The password of vector store, if not set, will use the "
            "default password."
        ),
        optional=True,
        default=None,
    ),
    # 构建参数对象，描述嵌入函数
    Parameter.build_from(
        _("Embedding Function"),
        "embedding_fn",
        Embeddings,
        description=_(
            "The embedding function of vector store, if not set, will use "
            "the default embedding function."
        ),
        optional=True,
        default=None,
    ),
    # 构建参数对象，描述一次加载的最大块数
    Parameter.build_from(
        _("Max Chunks Once Load"),
        "max_chunks_once_load",
        int,
        description=_(
            "The max number of chunks to load at once. If your document is "
            "large, you can set this value to a larger number to speed up the loading "
            "process. Default is 10."
        ),
        optional=True,
        default=10,
    ),
    # 构建参数对象，描述最大线程数
    Parameter.build_from(
        _("Max Threads"),
        "max_threads",
        int,
        description=_(
            "The max number of threads to use. Default is 1. If you set "
            "this bigger than 1, please make sure your vector store is thread-safe."
        ),
        optional=True,
        default=1,
    ),
]


class VectorStoreConfig(IndexStoreConfig):
    """Vector store config."""

    # 模型配置为任意类型的配置字典，允许额外字段
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    # 用户名，如果未设置，则使用默认用户
    user: Optional[str] = Field(
        default=None,
        description="The user of vector store, if not set, will use the default user.",
    )
    # 密码，如果未设置，则使用默认密码
    password: Optional[str] = Field(
        default=None,
        description=(
            "The password of vector store, if not set, will use the default password."
        ),
    )


class VectorStoreBase(IndexStoreBase, ABC):
    """Vector store base class."""
    def __init__(self, executor: Optional[ThreadPoolExecutor] = None):
        """
        Initialize vector store.

        Args:
            executor (Optional[ThreadPoolExecutor]): Optional thread pool executor.
        """
        super().__init__(executor)

    def filter_by_score_threshold(
        self, chunks: List[Chunk], score_threshold: float
    ) -> List[Chunk]:
        """
        Filter chunks by score threshold.

        Args:
            chunks (List[Chunk]): The chunks to filter.
            score_threshold (float): The score threshold.

        Returns:
            List[Chunk]: The filtered chunks.
        """
        candidates_chunks = chunks
        if score_threshold is not None:
            candidates_chunks = [
                Chunk(
                    metadata=chunk.metadata,
                    content=chunk.content,
                    score=chunk.score,
                    chunk_id=str(id),
                )
                for chunk in chunks
                if chunk.score >= score_threshold
            ]
            if len(candidates_chunks) == 0:
                logger.warning(
                    "No relevant docs were retrieved using the relevance score"
                    f" threshold {score_threshold}"
                )
        return candidates_chunks

    @abstractmethod
    def vector_name_exists(self) -> bool:
        """
        Whether vector name exists.

        Returns:
            bool: True if vector name exists, False otherwise.
        """
        return False

    def convert_metadata_filters(self, filters: MetadataFilters) -> Any:
        """
        Convert metadata filters to vector store filters.

        Args:
            filters (Optional[MetadataFilters]): Metadata filters.
        """
        raise NotImplementedError

    def _normalization_vectors(self, vectors):
        """
        Return L2-normalization vectors to scale[0,1].

        Args:
            vectors: Vectors to normalize.

        Returns:
            Normalized vectors scaled between [0,1].
        """
        import numpy as np

        norm = np.linalg.norm(vectors)
        return vectors / norm

    def _default_relevance_score_fn(self, distance: float) -> float:
        """
        Return a similarity score on a scale [0, 1].

        Args:
            distance (float): Distance value.

        Returns:
            float: Similarity score between 0 and 1.
        """
        return 1.0 - distance / math.sqrt(2)

    async def aload_document(self, chunks: List[Chunk]) -> List[str]:  # type: ignore
        """
        Async load document in index database.

        Args:
            chunks (List[Chunk]): Document chunks.

        Returns:
            List[str]: Chunk ids.
        """
        return await blocking_func_to_async(self._executor, self.load_document, chunks)
```