# `.\DB-GPT-src\dbgpt\agent\core\memory\agent_memory.py`

```py
"""
Agent memory module.
"""

# 导入所需模块和类
from datetime import datetime
from typing import Callable, List, Optional, Type, cast

# 导入自定义模块和类
from dbgpt.core import LLMClient
from dbgpt.util.annotations import immutable, mutable
from dbgpt.util.id_generator import new_id

# 导入基类和其他相关类
from .base import (
    DiscardedMemoryFragments,
    ImportanceScorer,
    InsightExtractor,
    Memory,
    MemoryFragment,
    ShortTermMemory,
    WriteOperation,
)
from .gpts import GptsMemory, GptsMessageMemory, GptsPlansMemory


class AgentMemoryFragment(MemoryFragment):
    """Default memory fragment for agent memory."""

    def __init__(
        self,
        observation: str,
        embeddings: Optional[List[float]] = None,
        memory_id: Optional[int] = None,
        importance: Optional[float] = None,
        last_accessed_time: Optional[datetime] = None,
        is_insight: bool = False,
    ):
        """Create a memory fragment."""
        # 如果没有指定memory_id，则使用snowflake id生成器生成一个新的memory_id
        if not memory_id:
            memory_id = new_id()
        self.observation = observation
        self._embeddings = embeddings  # 存储嵌入向量
        self.memory_id: int = cast(int, memory_id)  # 存储内存片段的唯一标识符
        self._importance: Optional[float] = importance  # 存储内存片段的重要性
        self._last_accessed_time: Optional[datetime] = last_accessed_time  # 存储最后访问时间
        self._is_insight = is_insight  # 表示内存片段是否为洞察

    @property
    def id(self) -> int:
        """Return the memory id."""
        return self.memory_id  # 返回内存片段的唯一标识符

    @property
    def raw_observation(self) -> str:
        """Return the raw observation."""
        return self.observation  # 返回原始观察结果

    @property
    def embeddings(self) -> Optional[List[float]]:
        """Return the embeddings of the memory fragment."""
        return self._embeddings  # 返回嵌入向量

    def update_embeddings(self, embeddings: List[float]) -> None:
        """Update the embeddings of the memory fragment.

        Args:
            embeddings(List[float]): embeddings
        """
        self._embeddings = embeddings  # 更新嵌入向量

    def calculate_current_embeddings(
        self, embedding_func: Callable[[List[str]], List[List[float]]]
    ) -> List[float]:
        """Calculate the embeddings of the memory fragment.

        Args:
            embedding_func(Callable[[List[str]], List[List[float]]]): Function to
                compute embeddings

        Returns:
            List[float]: Embeddings of the memory fragment
        """
        embeddings = embedding_func([self.observation])  # 使用给定的函数计算嵌入向量
        return embeddings[0]  # 返回计算得到的嵌入向量

    @property
    def is_insight(self) -> bool:
        """Return whether the memory fragment is an insight.

        Returns:
            bool: Whether the memory fragment is an insight
        """
        return self._is_insight  # 返回内存片段是否为洞察的布尔值

    @property
    def importance(self) -> Optional[float]:
        """Return the importance of the memory fragment.

        Returns:
            Optional[float]: Importance of the memory fragment
        """
        return self._importance  # 返回内存片段的重要性
    # 更新内存片段的重要性。
    # 参数:
    #     importance(float): 内存片段的重要性
    # 返回:
    #     Optional[float]: 旧的重要性
    def update_importance(self, importance: float) -> Optional[float]:
        old_importance = self._importance  # 保存旧的重要性值
        self._importance = importance  # 更新重要性值
        return old_importance  # 返回旧的重要性值

    @property
    # 返回内存片段的最后访问时间。
    # 用于确定最近最少使用的内存片段。
    # 返回:
    #     Optional[datetime]: 最后访问时间
    def last_accessed_time(self) -> Optional[datetime]:
        return self._last_accessed_time

    # 更新内存片段的最后访问时间。
    # 参数:
    #     now(datetime): 当前时间
    # 返回:
    #     Optional[datetime]: 旧的最后访问时间
    def update_accessed_time(self, now: datetime) -> Optional[datetime]:
        old_time = self._last_accessed_time  # 保存旧的最后访问时间
        self._last_accessed_time = now  # 更新最后访问时间为当前时间
        return old_time  # 返回旧的最后访问时间

    @classmethod
    # 从给定参数构建一个内存片段。
    # 参数:
    #     observation(str): 观察结果
    #     embeddings(Optional[List[float]]): 嵌入向量（可选）
    #     memory_id(Optional[int]): 内存 ID（可选）
    #     importance(Optional[float]): 重要性（可选）
    #     is_insight(bool): 是否为洞察结果（默认为 False）
    #     last_accessed_time(Optional[datetime]): 最后访问时间（可选）
    # 返回:
    #     "AgentMemoryFragment": 构建的内存片段
    def build_from(
        cls: Type["AgentMemoryFragment"],
        observation: str,
        embeddings: Optional[List[float]] = None,
        memory_id: Optional[int] = None,
        importance: Optional[float] = None,
        is_insight: bool = False,
        last_accessed_time: Optional[datetime] = None,
        **kwargs
    ) -> "AgentMemoryFragment":
        return cls(
            observation=observation,
            embeddings=embeddings,
            memory_id=memory_id,
            importance=importance,
            last_accessed_time=last_accessed_time,
            is_insight=is_insight,
        )

    # 返回内存片段的副本。
    # 返回:
    #     "AgentMemoryFragment": 内存片段的副本
    def copy(self: "AgentMemoryFragment") -> "AgentMemoryFragment":
        return AgentMemoryFragment.build_from(
            observation=self.observation,
            embeddings=self._embeddings,
            memory_id=self.memory_id,
            importance=self.importance,
            last_accessed_time=self.last_accessed_time,
            is_insight=self.is_insight,
        )
class AgentMemory(Memory[AgentMemoryFragment]):
    """Agent memory."""

    def __init__(
        self,
        memory: Optional[Memory[AgentMemoryFragment]] = None,
        importance_scorer: Optional[ImportanceScorer[AgentMemoryFragment]] = None,
        insight_extractor: Optional[InsightExtractor[AgentMemoryFragment]] = None,
        gpts_memory: Optional[GptsMemory] = None,
    ):
        """Create an agent memory.

        Args:
            memory(Memory[AgentMemoryFragment]): Memory to store fragments
            importance_scorer(ImportanceScorer[AgentMemoryFragment]): Scorer to
                calculate the importance of memory fragments
            insight_extractor(InsightExtractor[AgentMemoryFragment]): Extractor to
                extract insights from memory fragments
            gpts_memory(GptsMemory): Memory to store GPTs related information
        """
        # 如果未提供memory参数，使用默认的ShortTermMemory来创建一个内存对象
        if not memory:
            memory = ShortTermMemory(buffer_size=5)
        # 如果未提供gpts_memory参数，创建一个新的GptsMemory对象
        if not gpts_memory:
            gpts_memory = GptsMemory()
        # 将传入的参数分别赋值给对象的属性
        self.memory: Memory[AgentMemoryFragment] = cast(
            Memory[AgentMemoryFragment], memory
        )
        self.importance_scorer = importance_scorer
        self.insight_extractor = insight_extractor
        self.gpts_memory = gpts_memory

    @immutable
    def structure_clone(
        self: "AgentMemory", now: Optional[datetime] = None
    ) -> "AgentMemory":
        """Return a structure clone of the memory.

        The gpst_memory is not cloned, it will be shared in whole agent memory.
        """
        # 创建一个AgentMemory对象的结构克隆，并传入当前时间参数
        m = AgentMemory(
            memory=self.memory.structure_clone(now),
            importance_scorer=self.importance_scorer,
            insight_extractor=self.insight_extractor,
            gpts_memory=self.gpts_memory,
        )
        # 从当前对象复制其他属性到新创建的结构克隆对象中
        m._copy_from(self)
        return m

    @mutable
    def initialize(
        self,
        name: Optional[str] = None,
        llm_client: Optional[LLMClient] = None,
        importance_scorer: Optional[ImportanceScorer[AgentMemoryFragment]] = None,
        insight_extractor: Optional[InsightExtractor[AgentMemoryFragment]] = None,
        real_memory_fragment_class: Optional[Type[AgentMemoryFragment]] = None,
    ) -> None:
        """Initialize the memory."""
        # 调用内存对象的初始化方法，传入相应参数初始化内存
        self.memory.initialize(
            name=name,
            llm_client=llm_client,
            importance_scorer=importance_scorer or self.importance_scorer,
            insight_extractor=insight_extractor or self.insight_extractor,
            real_memory_fragment_class=real_memory_fragment_class
            or AgentMemoryFragment,
        )

    @mutable
    async def write(
        self,
        memory_fragment: AgentMemoryFragment,
        now: Optional[datetime] = None,
        op: WriteOperation = WriteOperation.ADD,
    ) -> Optional[DiscardedMemoryFragments[AgentMemoryFragment]]:
        """Write a memory fragment to the memory."""
        # 将内存片段写入内存，返回可能被丢弃的内存片段（如果有的话）
        return await self.memory.write(memory_fragment, now)

    @immutable
    # 异步方法：读取与给定观察相关的记忆片段列表
    async def read(
        self,
        observation: str,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
    ) -> List[AgentMemoryFragment]:
        """Read memory fragments related to the observation.

        Args:
            observation(str): Observation
            alpha(float): Importance weight
            beta(float): Time weight
            gamma(float): Randomness weight

        Returns:
            List[AgentMemoryFragment]: List of memory fragments
        """
        # 调用内部 memory 对象的 read 方法，传递给定的参数，并异步等待返回结果
        return await self.memory.read(observation, alpha, beta, gamma)

    # 可变修饰器方法：清空内存
    @mutable
    async def clear(self) -> List[AgentMemoryFragment]:
        """Clear the memory."""
        # 调用内部 memory 对象的 clear 方法，异步等待返回结果
        return await self.memory.clear()

    # 属性方法：返回计划记忆
    @property
    def plans_memory(self) -> GptsPlansMemory:
        """Return the plan memory."""
        # 返回内部 gpts_memory 对象的 plans_memory 属性
        return self.gpts_memory.plans_memory

    # 属性方法：返回消息记忆
    @property
    def message_memory(self) -> GptsMessageMemory:
        """Return the message memory."""
        # 返回内部 gpts_memory 对象的 message_memory 属性
        return self.gpts_memory.message_memory
```