# `.\DB-GPT-src\dbgpt\agent\core\memory\hybrid.py`

```py
# 导入必要的模块和类
import os.path
from concurrent.futures import Executor, ThreadPoolExecutor
from datetime import datetime
from typing import Generic, List, Optional, Tuple, Type

# 导入自定义模块和类
from dbgpt.core import Embeddings, LLMClient
from dbgpt.storage.vector_store.base import VectorStoreBase
from dbgpt.util.annotations import immutable, mutable

# 导入内部的模块和类
from .base import (
    DiscardedMemoryFragments,
    ImportanceScorer,
    InsightExtractor,
    Memory,
    SensoryMemory,
    ShortTermMemory,
    T,
    WriteOperation,
)
from .long_term import LongTermMemory
from .short_term import EnhancedShortTermMemory

# 定义混合记忆类，继承自 Memory，支持泛型 T
class HybridMemory(Memory, Generic[T]):
    """Hybrid memory for the agent."""

    # 重要性权重，默认为 0.9
    importance_weight: float = 0.9

    def __init__(
        self,
        now: datetime,
        sensory_memory: SensoryMemory[T],
        short_term_memory: ShortTermMemory[T],
        long_term_memory: LongTermMemory[T],
        default_insight_extractor: Optional[InsightExtractor] = None,
        default_importance_scorer: Optional[ImportanceScorer] = None,
    ):
        """Create a hybrid memory."""
        # 初始化方法，接受当前时间、感官记忆、短期记忆、长期记忆等参数
        self.now = now
        self._sensory_memory = sensory_memory  # 设置感官记忆
        self._short_term_memory = short_term_memory  # 设置短期记忆
        self._long_term_memory = long_term_memory  # 设置长期记忆
        self._default_insight_extractor = default_insight_extractor  # 设置默认洞察提取器
        self._default_importance_scorer = default_importance_scorer  # 设置默认重要性评分器

    def structure_clone(
        self: "HybridMemory[T]", now: Optional[datetime] = None
    ) -> "HybridMemory[T]":
        """Return a structure clone of the memory."""
        # 结构克隆方法，返回一个记忆结构的克隆体
        now = now or self.now
        m = HybridMemory(
            now=now,
            sensory_memory=self._sensory_memory.structure_clone(now),
            short_term_memory=self._short_term_memory.structure_clone(now),
            long_term_memory=self._long_term_memory.structure_clone(now),
            default_insight_extractor=self._default_insight_extractor,
            default_importance_scorer=self._default_importance_scorer,
        )
        m._copy_from(self)  # 调用内部方法复制当前对象的状态
        return m

    @classmethod
    def from_chroma(
        cls,
        vstore_name: Optional[str] = "_chroma_agent_memory_",
        vstore_path: Optional[str] = None,
        embeddings: Optional[Embeddings] = None,
        executor: Optional[Executor] = None,
        now: Optional[datetime] = None,
        sensory_memory: Optional[SensoryMemory[T]] = None,
        short_term_memory: Optional[ShortTermMemory[T]] = None,
        long_term_memory: Optional[LongTermMemory[T]] = None,
        **kwargs
    ):
    ):
        """
        从 Chroma 向量存储创建混合内存。
        """
        from dbgpt.configs.model_config import DATA_DIR
        from dbgpt.storage.vector_store.chroma_store import (
            ChromaStore,
            ChromaVectorConfig,
        )

        if not embeddings:
            from dbgpt.rag.embedding import DefaultEmbeddingFactory

            embeddings = DefaultEmbeddingFactory.openai()

        vstore_path = vstore_path or os.path.join(DATA_DIR, "agent_memory")

        # 创建 ChromaStore 对象，配置为指定的名称、持久化路径和嵌入函数
        vector_store = ChromaStore(
            ChromaVectorConfig(
                name=vstore_name,
                persist_path=vstore_path,
                embedding_fn=embeddings,
            )
        )
        # 调用类方法 from_vstore 创建类的实例并返回
        return cls.from_vstore(
            vector_store=vector_store,
            embeddings=embeddings,
            executor=executor,
            now=now,
            sensory_memory=sensory_memory,
            short_term_memory=short_term_memory,
            long_term_memory=long_term_memory,
            **kwargs
        )

    @classmethod
    def from_vstore(
        cls,
        vector_store: "VectorStoreBase",
        embeddings: Optional[Embeddings] = None,
        executor: Optional[Executor] = None,
        now: Optional[datetime] = None,
        sensory_memory: Optional[SensoryMemory[T]] = None,
        short_term_memory: Optional[ShortTermMemory[T]] = None,
        long_term_memory: Optional[LongTermMemory[T]] = None,
        **kwargs
    ):
        """
        从向量存储创建混合内存。
        """
        if not embeddings:
            raise ValueError("embeddings is required.")
        if not executor:
            executor = ThreadPoolExecutor()
        if not now:
            now = datetime.now()

        # 如果 sensory_memory 为 None，则初始化为 SensoryMemory 对象
        if not sensory_memory:
            sensory_memory = SensoryMemory()
        # 如果 short_term_memory 为 None，则使用 embeddings 创建 EnhancedShortTermMemory 对象
        if not short_term_memory:
            if not embeddings:
                raise ValueError("embeddings is required.")
            short_term_memory = EnhancedShortTermMemory(embeddings, executor)
        # 如果 long_term_memory 为 None，则使用给定的 executor、vector_store 和当前时间创建 LongTermMemory 对象
        if not long_term_memory:
            long_term_memory = LongTermMemory(
                executor,
                vector_store,
                now=now,
            )
        # 创建并返回类的实例，初始化内存配置及其他参数
        return cls(now, sensory_memory, short_term_memory, long_term_memory, **kwargs)

    def initialize(
        self,
        name: Optional[str] = None,
        llm_client: Optional[LLMClient] = None,
        importance_scorer: Optional[ImportanceScorer[T]] = None,
        insight_extractor: Optional[InsightExtractor[T]] = None,
        real_memory_fragment_class: Optional[Type[T]] = None,
    ) -> None:
        """Initialize the memory.

        It will initialize all the memories.
        """
        # 定义初始化方法，设置所有的内存空间
        memories = [
            self._sensory_memory,  # 感知记忆空间
            self._short_term_memory,  # 短期记忆空间
            self._long_term_memory,  # 长期记忆空间
        ]
        kwargs = {
            "name": name,  # 内存名称
            "llm_client": llm_client,  # LLM客户端
            "importance_scorer": importance_scorer,  # 重要性评分器
            "insight_extractor": insight_extractor,  # 洞察提取器
            "real_memory_fragment_class": real_memory_fragment_class,  # 真实内存片段类
        }
        for memory in memories:
            memory.initialize(**kwargs)  # 初始化各个内存空间
        super().initialize(**kwargs)  # 调用父类的初始化方法

    @mutable
    async def write(
        self,
        memory_fragment: T,
        now: Optional[datetime] = None,
        op: WriteOperation = WriteOperation.ADD,
    ) -> Optional[DiscardedMemoryFragments[T]]:
        """Write a memory fragment to the memory."""
        # 首先写入感知记忆空间
        sen_discarded_memories = await self._sensory_memory.write(memory_fragment)
        if not sen_discarded_memories:
            return None
        short_term_discarded_memories = []
        discarded_memory_fragments = []
        discarded_insights = []
        for sen_memory in sen_discarded_memories.discarded_memory_fragments:
            # 写入短期记忆空间
            short_discarded_memory = await self._short_term_memory.write(sen_memory)
            if short_discarded_memory:
                short_term_discarded_memories.append(short_discarded_memory)
                discarded_memory_fragments.extend(
                    short_discarded_memory.discarded_memory_fragments
                )
                for insight in short_discarded_memory.discarded_insights:
                    # 仅保留第一个洞察
                    discarded_insights.append(insight.insights[0])
        # 获取洞察的重要性评分
        insight_scores = await self.score_memory_importance(discarded_insights)
        # 更新洞察的重要性
        for i, ins in enumerate(discarded_insights):
            ins.update_importance(insight_scores[i])
        all_memories = discarded_memory_fragments + discarded_insights
        if self._long_term_memory:
            # 写入长期记忆空间
            await self._long_term_memory.write_batch(all_memories, self.now)
        return None

    @immutable
    async def read(
        self,
        observation: str,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
    ) -> List[T]:
        """Read memories from the memory."""
        (
            retrieved_long_term_memories,
            short_term_discarded_memories,
        ) = await self.fetch_memories(observation, self._short_term_memory)

        await self.save_memories_after_retrieval(short_term_discarded_memories)
        return retrieved_long_term_memories

    @immutable
    async def fetch_memories(
        self,
        observation: str,
        short_term_memory: Optional[ShortTermMemory[T]] = None,
    ) -> Tuple[List[T], List[DiscardedMemoryFragments[T]]]:
        """Fetch memories from long term memory.

        If short_term_memory is provided, write the fetched memories to the short term
        memory.
        """
        # 从长期记忆中获取与给定观察相关的记忆片段
        retrieved_long_term_memories = await self._long_term_memory.fetch_memories(
            observation
        )
        # 如果没有提供 short_term_memory，则直接返回长期记忆中获取的记忆片段和空列表
        if not short_term_memory:
            return retrieved_long_term_memories, []
        # 初始化存储丢弃的短期记忆片段和丢弃的记忆片段列表
        short_term_discarded_memories: List[DiscardedMemoryFragments[T]] = []
        discarded_memory_fragments: List[T] = []
        # 遍历长期记忆中获取的每个记忆片段
        for ltm in retrieved_long_term_memories:
            # 将每个长期记忆片段写入短期记忆中，操作为检索
            short_discarded_memory = await short_term_memory.write(
                ltm, op=WriteOperation.RETRIEVAL
            )
            # 如果有丢弃的短期记忆片段，则添加到列表中
            if short_discarded_memory:
                short_term_discarded_memories.append(short_discarded_memory)
                # 将丢弃的记忆片段添加到总体丢弃的记忆片段列表中
                discarded_memory_fragments.extend(
                    short_discarded_memory.discarded_memory_fragments
                )
        # 遍历短期记忆中的每个短期记忆
        for stm in short_term_memory.short_term_memories:
            # 根据当前类构建新的记忆片段，基于原始观察和重要性
            retrieved_long_term_memories.append(
                stm.current_class.build_from(
                    observation=stm.raw_observation,
                    importance=stm.importance,
                )
            )
        # 返回所有获取的长期记忆片段和丢弃的短期记忆片段列表
        return retrieved_long_term_memories, short_term_discarded_memories

    async def save_memories_after_retrieval(
        self, fragments: List[DiscardedMemoryFragments[T]]
    ):
        """Save memories after retrieval."""
        # 初始化丢弃的记忆片段和丢弃的记忆洞察力列表
        discarded_memory_fragments = []
        discarded_memory_insights: List[T] = []
        # 遍历传入的记忆片段列表
        for f in fragments:
            # 将每个记忆片段的丢弃记忆片段添加到总体丢弃的记忆片段列表中
            discarded_memory_fragments.extend(f.discarded_memory_fragments)
            # 遍历每个记忆片段的丢弃洞察力
            for fi in f.discarded_insights:
                # 将每个洞察力的第一个洞察添加到丢弃的记忆洞察力列表中
                discarded_memory_insights.append(fi.insights[0])
        # 对丢弃的记忆洞察力进行重要性评分
        insights_importance = await self.score_memory_importance(
            discarded_memory_insights
        )
        # 更新每个洞察力的重要性
        for i, ins in enumerate(discarded_memory_insights):
            ins.update_importance(insights_importance[i])
        # 组合所有丢弃的记忆片段和洞察力为一个列表
        all_memories = discarded_memory_fragments + discarded_memory_insights
        # 将所有记忆批量写入长期记忆，并使用当前时间戳
        await self._long_term_memory.write_batch(all_memories, self.now)

    async def clear(self) -> List[T]:
        """Clear the memory.

        # TODO
        """
        # 返回一个空列表，表示内存已被清除
        return []
```