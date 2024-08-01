# `.\DB-GPT-src\dbgpt\rag\assembler\base.py`

```py
"""Base Assembler."""

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from dbgpt.core import Chunk  # 导入 Chunk 类
from dbgpt.util.tracer import root_tracer  # 导入 root_tracer 方法

from ..chunk_manager import ChunkManager, ChunkParameters  # 导入 ChunkManager 和 ChunkParameters 类
from ..extractor.base import Extractor  # 导入 Extractor 类
from ..knowledge.base import Knowledge  # 导入 Knowledge 类
from ..retriever.base import BaseRetriever  # 导入 BaseRetriever 类


class BaseAssembler(ABC):
    """Base Assembler."""

    def __init__(
        self,
        knowledge: Knowledge,
        chunk_parameters: Optional[ChunkParameters] = None,
        extractor: Optional[Extractor] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with Assembler arguments.

        Args:
            knowledge(Knowledge): Knowledge datasource.
            chunk_parameters: (Optional[ChunkParameters]) ChunkManager to use for
                chunking.
            extractor(Optional[Extractor]):  Extractor to use for summarization.
        """
        self._knowledge = knowledge  # 存储传入的 knowledge 参数
        self._chunk_parameters = chunk_parameters or ChunkParameters()  # 如果未提供 chunk_parameters，则使用默认值
        self._extractor = extractor  # 存储传入的 extractor 参数
        self._chunk_manager = ChunkManager(
            knowledge=self._knowledge, chunk_parameter=self._chunk_parameters
        )  # 创建 ChunkManager 对象，用于管理 chunk
        self._chunks: List[Chunk] = []  # 初始化 _chunks 为空列表
        metadata = {
            "knowledge_cls": (
                self._knowledge.__class__.__name__ if self._knowledge else None
            ),  # 获取 knowledge 类的名称
            "knowledge_type": self._knowledge.type().value if self._knowledge else None,  # 获取 knowledge 的类型值
            "path": (
                self._knowledge._path
                if self._knowledge and hasattr(self._knowledge, "_path")
                else None
            ),  # 获取 knowledge 的路径
            "chunk_parameters": self._chunk_parameters.dict(),  # 获取 chunk_parameters 的字典表示
        }
        with root_tracer.start_span("BaseAssembler.load_knowledge", metadata=metadata):
            self.load_knowledge(self._knowledge)  # 调用 load_knowledge 方法加载 knowledge 数据

    def load_knowledge(self, knowledge: Knowledge) -> None:
        """Load knowledge Pipeline."""
        if not knowledge:
            raise ValueError("knowledge must be provided.")  # 如果 knowledge 未提供则抛出 ValueError 异常
        with root_tracer.start_span("BaseAssembler.knowledge.load"):
            documents = knowledge.load()  # 调用 knowledge 对象的 load 方法加载文档
        with root_tracer.start_span("BaseAssembler.chunk_manager.split"):
            self._chunks = self._chunk_manager.split(documents)  # 使用 chunk_manager 对文档进行分割，生成 chunks

    @abstractmethod
    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        """Return a retriever."""
        # 抽象方法，子类需实现该方法以返回 BaseRetriever 对象

    @abstractmethod
    def persist(self, **kwargs: Any) -> List[str]:
        """Persist chunks.

        Returns:
            List[str]: List of persisted chunk ids.
        """
        # 抽象方法，子类需实现该方法以持久化 chunks，并返回持久化后的 chunk ids

    async def apersist(self, **kwargs: Any) -> List[str]:
        """Persist chunks.

        Returns:
            List[str]: List of persisted chunk ids.
        """
        raise NotImplementedError  # async 抽象方法，子类需实现该方法以异步方式持久化 chunks

    def get_chunks(self) -> List[Chunk]:
        """Return chunks."""
        return self._chunks  # 返回当前 _chunks 列表
```