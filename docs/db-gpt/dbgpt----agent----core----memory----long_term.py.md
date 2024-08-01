# `.\DB-GPT-src\dbgpt\agent\core\memory\long_term.py`

```py
"""Long-term memory module."""

from concurrent.futures import Executor  # 导入并发执行器
from datetime import datetime  # 导入日期时间模块
from typing import Generic, List, Optional  # 引入类型提示

from dbgpt.core import Chunk  # 导入核心模块中的Chunk类
from dbgpt.rag.retriever.time_weighted import TimeWeightedEmbeddingRetriever  # 导入时间加权嵌入检索器
from dbgpt.storage.vector_store.base import VectorStoreBase  # 导入向量存储基类
from dbgpt.storage.vector_store.filters import MetadataFilters  # 导入元数据过滤器
from dbgpt.util.annotations import immutable, mutable  # 导入不可变和可变注解
from dbgpt.util.executor_utils import blocking_func_to_async  # 导入阻塞函数转异步函数的工具函数

from .base import DiscardedMemoryFragments, Memory, T, WriteOperation  # 导入本地基类和类型

_FORGET_PLACEHOLDER = "[FORGET]"  # 定义忘记占位符常量
_MERGE_PLACEHOLDER = "[MERGE]"  # 定义合并占位符常量
_METADATA_BUFFER_IDX = "buffer_idx"  # 定义缓冲区索引元数据常量
_METADATA_LAST_ACCESSED_AT = "last_accessed_at"  # 定义最后访问时间元数据常量
_METADAT_IMPORTANCE = "importance"  # 定义重要性元数据常量


class LongTermRetriever(TimeWeightedEmbeddingRetriever):
    """Long-term retriever."""

    def __init__(self, now: datetime, **kwargs):
        """Create a long-term retriever."""
        self.now = now  # 设置当前时间属性
        super().__init__(**kwargs)  # 调用父类初始化方法

    @mutable
    def _retrieve(
        self, query: str, filters: Optional[MetadataFilters] = None
    ) -> List[Chunk]:
        """Retrieve memories."""
        current_time = self.now  # 获取当前时间
        # 构建文档及其得分的字典
        docs_and_scores = {
            doc.metadata[_METADATA_BUFFER_IDX]: (doc, self.default_salience)
            for doc in self.memory_stream  # 遍历记忆流中的所有文档
        }
        # 更新具有显著性的文档及其分数
        docs_and_scores.update(self.get_salient_docs(query))
        # 重新评分的文档列表
        rescored_docs = [
            (doc, self._get_combined_score(doc, relevance, current_time))
            for doc, relevance in docs_and_scores.values()
        ]
        # 根据综合分数降序排序文档
        rescored_docs.sort(key=lambda x: x[1], reverse=True)
        result = []
        # 确保经常访问的记忆不会被遗忘
        retrieved_num = 0
        for doc, _ in rescored_docs:
            if (
                retrieved_num < self._k
                and doc.content.find(_FORGET_PLACEHOLDER) == -1
                and doc.content.find(_MERGE_PLACEHOLDER) == -1
            ):
                retrieved_num += 1
                # 更新缓冲文档的最后访问时间
                buffered_doc = self.memory_stream[doc.metadata[_METADATA_BUFFER_IDX]]
                buffered_doc.metadata[_METADATA_LAST_ACCESSED_AT] = current_time
                result.append(buffered_doc)  # 将文档添加到结果列表中
        return result


class LongTermMemory(Memory, Generic[T]):
    """Long-term memory."""

    importance_weight: float = 0.15  # 设置重要性权重默认值

    def __init__(
        self,
        executor: Executor,  # 并发执行器
        vector_store: VectorStoreBase,  # 向量存储器基类
        now: Optional[datetime] = None,  # 可选的当前时间
        reflection_threshold: Optional[float] = None,  # 可选的反射阈值
        _default_importance: Optional[float] = None,  # 可选的默认重要性
        ...
        ):
        """
        Initialize LongTermMemory.

        Args:
            executor (Executor): Concurrent executor.
            vector_store (VectorStoreBase): Base vector store.
            now (Optional[datetime], optional): Current datetime. Defaults to None.
            reflection_threshold (Optional[float], optional): Reflection threshold. Defaults to None.
            _default_importance (Optional[float], optional): Default importance. Defaults to None.
        """
        super().__init__(executor, vector_store)  # 调用父类初始化方法
        self.importance_weight = _default_importance if _default_importance is not None else self.importance_weight
        # 设置重要性权重
        self.now = now  # 设置当前时间属性
        self.reflection_threshold = reflection_threshold  # 设置反射阈值属性
    ):
        """Create a long-term memory."""
        # 初始化长期记忆的对象
        self.now = now or datetime.now()  # 设置当前时间，如果未提供则使用当前时间
        self.executor = executor  # 设置执行器对象
        self.reflecting: bool = False  # 设置反射状态为False
        self.forgetting: bool = False  # 设置遗忘状态为False
        self.reflection_threshold: Optional[float] = reflection_threshold  # 设置反思阈值（可选）
        self.aggregate_importance: float = 0.0  # 初始化累积重要性为0.0
        self._vector_store = vector_store  # 设置向量存储对象
        self.memory_retriever = LongTermRetriever(
            now=self.now, index_store=vector_store
        )  # 初始化长期记忆检索器对象，使用当前时间和向量存储对象
        self._default_importance = _default_importance  # 设置默认重要性

    @immutable
    def structure_clone(
        self: "LongTermMemory[T]", now: Optional[datetime] = None
    ) -> "LongTermMemory[T]":
        """Create a structure clone of the long-term memory."""
        # 创建长期记忆的结构克隆
        new_name = self.name
        if not new_name:
            raise ValueError("name is required.")  # 如果没有名称则抛出数值错误
        m: LongTermMemory[T] = LongTermMemory(
            now=now,
            executor=self.executor,
            vector_store=self._vector_store,
            reflection_threshold=self.reflection_threshold,
            _default_importance=self._default_importance,
        )  # 创建新的长期记忆对象，复制当前对象的参数
        m._copy_from(self)  # 从当前对象复制数据到新对象
        return m  # 返回新的长期记忆对象

    @mutable
    async def write(
        self,
        memory_fragment: T,
        now: Optional[datetime] = None,
        op: WriteOperation = WriteOperation.ADD,
    ) -> Optional[DiscardedMemoryFragments[T]]:
        """Write a memory fragment to the memory."""
        # 将记忆片段写入长期记忆
        importance = memory_fragment.importance  # 获取记忆片段的重要性
        if importance is None:
            importance = self._default_importance  # 如果未提供重要性，则使用默认重要性
        last_accessed_time = memory_fragment.last_accessed_time  # 获取记忆片段的最后访问时间
        if importance is None:
            raise ValueError("importance is required.")  # 如果重要性未提供则抛出数值错误
        if not self.reflecting:
            self.aggregate_importance += importance  # 如果非反思状态，则累加重要性

        memory_idx = len(self.memory_retriever.memory_stream)  # 计算记忆流的长度作为记忆索引
        document = Chunk(
            page_content="[{}] ".format(memory_idx)
            + str(memory_fragment.raw_observation),
            metadata={
                _METADAT_IMPORTANCE: importance,
                _METADATA_LAST_ACCESSED_AT: last_accessed_time,
            },
        )  # 创建一个新的文档块，包含记忆片段的内容和元数据
        await blocking_func_to_async(
            self.executor,
            self.memory_retriever.load_document,
            [document],
            current_time=now,
        )  # 异步将文档加载到长期记忆中

        return None

    @mutable
    async def write_batch(
        self, memory_fragments: List[T], now: Optional[datetime] = None
    ):
        """Write a batch of memory fragments to the memory."""
        # 将一批记忆片段写入长期记忆
        # 遍历每个记忆片段
        for memory_fragment in memory_fragments:
            await self.write(memory_fragment, now=now)  # 调用单个写入函数来写入每个记忆片段
    ) -> Optional[DiscardedMemoryFragments[T]]:
        """Write a batch of memory fragments to the memory."""
        current_datetime = self.now
        if not now:
            raise ValueError("Now time is required.")
        for short_term_memory in memory_fragments:
            # Update the accessed time of each memory fragment to the current datetime
            short_term_memory.update_accessed_time(now=now)
            # Asynchronously write each memory fragment with the current datetime
            await self.write(short_term_memory, now=current_datetime)
        # TODO(fangyinc): Reflect on the memories and get high-level insights.
        # TODO(fangyinc): Forget memories that are not important.
        # Return None as the operation is complete
        return None

    @immutable
    async def read(
        self,
        observation: str,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
    ) -> List[T]:
        """Read memory fragments related to the observation."""
        # Fetch and return memories associated with the provided observation at the current time
        return await self.fetch_memories(observation=observation, now=self.now)

    @immutable
    async def fetch_memories(
        self, observation: str, now: Optional[datetime] = None
    ) -> List[T]:
        """Fetch memories related to the observation."""
        # TODO: Mock now?
        retrieved_memories = []
        # Convert blocking function to asynchronous operation to retrieve memories
        retrieved_list = await blocking_func_to_async(
            self.executor,
            self.memory_retriever.retrieve,
            observation,
        )
        # Process each retrieved chunk to build memory fragments
        for retrieved_chunk in retrieved_list:
            retrieved_memories.append(
                self.real_memory_fragment_class.build_from(
                    observation=retrieved_chunk.content,
                    importance=retrieved_chunk.metadata[_METADAT_IMPORTANCE],
                )
            )
        # Return the list of retrieved memories
        return retrieved_memories

    @mutable
    async def clear(self) -> List[T]:
        """Clear the memory.

        TODO: Implement this method.
        """
        # Return an empty list to indicate that memory has been cleared
        return []
```