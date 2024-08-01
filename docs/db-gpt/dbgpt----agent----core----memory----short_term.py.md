# `.\DB-GPT-src\dbgpt\agent\core\memory\short_term.py`

```py
"""Short term memory module."""

import random
from concurrent.futures import Executor
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from dbgpt.core import Embeddings
from dbgpt.util.annotations import immutable, mutable
from dbgpt.util.executor_utils import blocking_func_to_async
from dbgpt.util.similarity_util import cosine_similarity, sigmoid_function

from .base import (
    DiscardedMemoryFragments,
    InsightMemoryFragment,
    ShortTermMemory,
    T,
    WriteOperation,
)


class EnhancedShortTermMemory(ShortTermMemory[T]):
    """Enhanced short term memory."""

    def __init__(
        self,
        embeddings: Embeddings,
        executor: Executor,
        buffer_size: int = 2,
        enhance_similarity_threshold: float = 0.7,
        enhance_threshold: int = 3,
    ):
        """Initialize enhanced short term memory."""
        # 调用父类初始化方法，设置缓冲区大小
        super().__init__(buffer_size=buffer_size)
        # 设置执行器和嵌入模型
        self._executor = executor
        self._embeddings = embeddings
        # 初始化增强记忆相关的数据结构
        self.short_embeddings: List[List[float]] = []
        self.enhance_cnt: List[int] = [0 for _ in range(self._buffer_size)]
        self.enhance_memories: List[List[T]] = [[] for _ in range(self._buffer_size)]
        self.enhance_similarity_threshold = enhance_similarity_threshold
        self.enhance_threshold = enhance_threshold

    @immutable
    def structure_clone(
        self: "EnhancedShortTermMemory[T]", now: Optional[datetime] = None
    ) -> "EnhancedShortTermMemory[T]":
        """Return a structure clone of the memory."""
        # 创建当前对象的结构克隆
        m: EnhancedShortTermMemory[T] = EnhancedShortTermMemory(
            embeddings=self._embeddings,
            executor=self._executor,
            buffer_size=self._buffer_size,
            enhance_similarity_threshold=self.enhance_similarity_threshold,
            enhance_threshold=self.enhance_threshold,
        )
        # 复制当前对象的数据到克隆对象中
        m._copy_from(self)
        return m

    @mutable
    async def write(
        self,
        memory_fragment: T,
        now: Optional[datetime] = None,
        op: WriteOperation = WriteOperation.ADD,
    ) -> None:
        """Write operation to add a memory fragment asynchronously."""
        # 将内存片段异步写入增强短期记忆中
        pass  # 实际实现略去
    ) -> Optional[DiscardedMemoryFragments[T]]:
        """Write memory fragment to short term memory.

        Reference: https://github.com/RUC-GSAI/YuLan-Rec/blob/main/agents/recagent_memory.py#L336 # noqa
        """
        # 计算当前内存片段的嵌入
        memory_fragment_embeddings = await blocking_func_to_async(
            self._executor,
            memory_fragment.calculate_current_embeddings,
            self._embeddings.embed_documents,
        )
        # 更新内存片段的嵌入
        memory_fragment.update_embeddings(memory_fragment_embeddings)
        # 遍历短期嵌入中的每个索引和对应的嵌入向量
        for idx, memory_embedding in enumerate(self.short_embeddings):
            # 计算内存嵌入向量与当前嵌入的余弦相似度
            similarity = await blocking_func_to_async(
                self._executor,
                cosine_similarity,
                memory_embedding,
                memory_fragment_embeddings,
            )
            # 使用 sigmoid 函数将相似度转换为概率值，范围为 [0, 1]
            sigmoid_prob: float = await blocking_func_to_async(
                self._executor, sigmoid_function, similarity
            )
            # 如果概率值大于等于增强相似度阈值，并且随机数小于概率值，则执行以下操作
            if (
                sigmoid_prob >= self.enhance_similarity_threshold
                and random.random() < sigmoid_prob
            ):
                # 增加增强计数器
                self.enhance_cnt[idx] += 1
                # 将当前内存片段添加到增强记忆中
                self.enhance_memories[idx].append(memory_fragment)
        # 将内存片段转移到长期记忆中并获取丢弃的记忆片段
        discard_memories = await self.transfer_to_long_term(memory_fragment)
        # 如果写操作为 ADD，则将内存片段和其嵌入向量添加到内部片段和短期嵌入中，并处理溢出
        if op == WriteOperation.ADD:
            self._fragments.append(memory_fragment)
            self.short_embeddings.append(memory_fragment_embeddings)
            await self.handle_overflow(self._fragments)
        # 返回丢弃的记忆片段
        return discard_memories
    ) -> Optional[DiscardedMemoryFragments[T]]:
        """Transfer memory fragment to long term memory."""
        transfer_flag = False
        existing_memory = [True for _ in range(len(self.short_term_memories))]

        enhance_memories: List[T] = []
        to_get_insight_memories: List[T] = []
        for idx, memory in enumerate(self.short_term_memories):
            # if exceed the enhancement threshold
            if (
                self.enhance_cnt[idx] >= self.enhance_threshold
                and existing_memory[idx] is True
            ):
                existing_memory[idx] = False
                transfer_flag = True
                #
                # short-term memories
                content = [memory]
                # do not repeatedly add observation memory to summary, so use [:-1].
                for enhance_memory in self.enhance_memories[idx][:-1]:
                    content.append(enhance_memory)
                # Append the current observation memory
                content.append(memory_fragment)
                # Merge the enhanced memories to single memory
                merged_enhance_memory: T = memory.reduce(
                    content, importance=memory.importance
                )
                to_get_insight_memories.append(merged_enhance_memory)
                enhance_memories.append(merged_enhance_memory)
        # Get insights for the every enhanced memory
        enhance_insights: List[InsightMemoryFragment] = await self.get_insights(
            to_get_insight_memories
        )

        if transfer_flag:
            # re-construct the indexes of short-term memories after removing summarized
            # memories
            new_memories: List[T] = []
            new_embeddings: List[List[float]] = []
            new_enhance_memories: List[List[T]] = [[] for _ in range(self._buffer_size)]
            new_enhance_cnt: List[int] = [0 for _ in range(self._buffer_size)]
            for idx, memory in enumerate(self.short_term_memories):
                if existing_memory[idx]:
                    # Remove not enhanced memories to new memories
                    new_enhance_memories[len(new_memories)] = self.enhance_memories[idx]
                    new_enhance_cnt[len(new_memories)] = self.enhance_cnt[idx]
                    new_memories.append(memory)
                    new_embeddings.append(self.short_embeddings[idx])
            self._fragments = new_memories
            self.short_embeddings = new_embeddings
            self.enhance_memories = new_enhance_memories
            self.enhance_cnt = new_enhance_cnt
        return DiscardedMemoryFragments(enhance_memories, enhance_insights)
    ) -> Tuple[List[T], List[T]]:
        """Handle overflow of short term memory.

        Discard the least important memory fragment if the buffer size exceeds.
        """
        # 如果短期记忆溢出，处理溢出情况
        if len(self.short_term_memories) > self._buffer_size:
            # 初始化存储各记忆片段信息的字典
            id2fragments: Dict[int, Dict] = {}
            # 遍历除最后一个外的所有短期记忆
            for idx in range(len(self.short_term_memories) - 1):
                # 获取当前记忆片段
                memory = self.short_term_memories[idx]
                # 将记忆片段的增强计数和重要性保存到字典中
                id2fragments[idx] = {
                    "enhance_count": self.enhance_cnt[idx],
                    "importance": memory.importance,
                }
            
            # 根据重要性和增强计数对记忆片段进行排序，优先丢弃最不重要的
            sorted_ids = sorted(
                id2fragments.keys(),
                key=lambda x: (
                    id2fragments[x]["importance"],
                    id2fragments[x]["enhance_count"],
                ),
            )
            # 获取要丢弃的记忆片段的索引
            pop_id = sorted_ids[0]
            # 获取要丢弃的原始观察数据
            pop_raw_observation = self.short_term_memories[pop_id].raw_observation
            
            # 更新增强计数列表，将要丢弃的记忆片段从中删除，并补充新的增强计数
            self.enhance_cnt.pop(pop_id)
            self.enhance_cnt.append(0)
            # 删除要丢弃的增强记忆
            self.enhance_memories.pop(pop_id)
            self.enhance_memories.append([])

            # 删除要丢弃的记忆片段及其对应的短期嵌入
            discard_memory = self._fragments.pop(pop_id)
            self.short_embeddings.pop(pop_id)

            # 从其他短期记忆的增强记忆列表中移除要丢弃的记忆片段
            for idx in range(len(self.short_term_memories)):
                current_enhance_memories: List[T] = self.enhance_memories[idx]
                to_remove_idx = []
                for i, ehf in enumerate(current_enhance_memories):
                    if ehf.raw_observation == pop_raw_observation:
                        to_remove_idx.append(i)
                for i in to_remove_idx:
                    current_enhance_memories.pop(i)
                # 更新相应记忆片段的增强计数
                self.enhance_cnt[idx] -= len(to_remove_idx)

            # 返回更新后的记忆片段列表和要丢弃的记忆片段列表
            return memory_fragments, [discard_memory]
        
        # 如果未溢出，直接返回当前的记忆片段列表和空列表
        return memory_fragments, []
```