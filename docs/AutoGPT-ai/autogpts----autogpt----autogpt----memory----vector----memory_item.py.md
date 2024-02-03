# `.\AutoGPT\autogpts\autogpt\autogpt\memory\vector\memory_item.py`

```py
# 导入必要的模块和库
from __future__ import annotations
import json
import logging
from typing import Literal
import ftfy
import numpy as np
from pydantic import BaseModel
from autogpt.config import Config
from autogpt.core.resource.model_providers import (
    ChatMessage,
    ChatModelProvider,
    EmbeddingModelProvider,
)
from autogpt.processing.text import chunk_content, split_text, summarize_text
from .utils import Embedding, get_embedding

# 设置日志记录器
logger = logging.getLogger(__name__)

# 定义一个类型别名，表示记忆文档的类型
MemoryDocType = Literal["webpage", "text_file", "code_file", "agent_history"]

# 定义一个记忆项类，包含原始内容、摘要、内容块、内容块摘要、嵌入向量等属性
class MemoryItem(BaseModel, arbitrary_types_allowed=True):
    """Memory object containing raw content as well as embeddings"""

    raw_content: str
    summary: str
    chunks: list[str]
    chunk_summaries: list[str]
    e_summary: Embedding
    e_chunks: list[Embedding]
    metadata: dict

    # 计算查询与记忆项的相关性
    def relevance_for(self, query: str, e_query: Embedding | None = None):
        return MemoryItemRelevance.of(self, query, e_query)

    # 将记忆项转换为字符串形式，包括内容块数量、元数据、摘要和原始内容
    def dump(self, calculate_length=False) -> str:
        n_chunks = len(self.e_chunks)
        return f"""
=============== MemoryItem ===============
Size: {n_chunks} chunks
Metadata: {json.dumps(self.metadata, indent=2)}
---------------- SUMMARY -----------------
{self.summary}
------------------ RAW -------------------
{self.raw_content}
==========================================
"""
    # 定义对象的相等比较方法，接受另一个 MemoryItem 对象作为参数
    def __eq__(self, other: MemoryItem):
        # 检查原始内容是否相等
        return (
            self.raw_content == other.raw_content
            # 检查块是否相等
            and self.chunks == other.chunks
            # 检查块摘要是否相等
            and self.chunk_summaries == other.chunk_summaries
            # 检查嵌入是否相等，嵌入可以是 list[float] 或 np.ndarray[float32]，必须是相同类型才能比较
            and np.array_equal(
                self.e_summary
                if isinstance(self.e_summary, np.ndarray)
                else np.array(self.e_summary, dtype=np.float32),
                other.e_summary
                if isinstance(other.e_summary, np.ndarray)
                else np.array(other.e_summary, dtype=np.float32),
            )
            # 检查块嵌入是否相等，块嵌入可以是 list[np.ndarray] 或 np.ndarray[np.float32]，必须是相同类型才能比较
            and np.array_equal(
                self.e_chunks
                if isinstance(self.e_chunks[0], np.ndarray)
                else [np.array(c, dtype=np.float32) for c in self.e_chunks],
                other.e_chunks
                if isinstance(other.e_chunks[0], np.ndarray)
                else [np.array(c, dtype=np.float32) for c in other.e_chunks],
            )
        )
# 定义一个内存项工厂类
class MemoryItemFactory:
    # 初始化方法，接受语言模型提供者和嵌入模型提供者作为参数
    def __init__(
        self,
        llm_provider: ChatModelProvider,
        embedding_provider: EmbeddingModelProvider,
    ):
        # 将语言模型提供者和嵌入模型提供者保存到实例变量中
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider

    # 从文本创建内存项
    async def from_text(
        self,
        text: str,
        source_type: MemoryDocType,
        config: Config,
        metadata: dict = {},
        how_to_summarize: str | None = None,
        question_for_summary: str | None = None,
    ):
    
    # 从文本文件创建内存项
    def from_text_file(self, content: str, path: str, config: Config):
        # 调用from_text方法，传入内容、文档类型、配置和元数据
        return self.from_text(content, "text_file", config, {"location": path})

    # 从代码文件创建内存项
    def from_code_file(self, content: str, path: str):
        # TODO: 实现定制的代码内存项
        # 调用from_text方法，传入内容、文档类型和元数据
        return self.from_text(content, "code_file", {"location": path})
    def from_ai_action(self, ai_message: ChatMessage, result_message: ChatMessage):
        # 从 AI 消息中提取信息，结果消息包含用户反馈或者 AI 指定命令的结果

        if ai_message.role != "assistant":
            # 如果 AI 消息的角色不是助手，则引发数值错误
            raise ValueError(f"Invalid role on 'ai_message': {ai_message.role}")

        # 提取结果消息中的内容，如果以"Command"开头，则为结果，否则为"None"
        result = (
            result_message.content
            if result_message.content.startswith("Command")
            else "None"
        )
        # 提取结果消息中的内容，如果以"Human feedback"开头，则为用户输入，否则为"None"
        user_input = (
            result_message.content
            if result_message.content.startswith("Human feedback")
            else "None"
        )
        # 组合助手回复、结果和用户反馈为记忆内容
        memory_content = (
            f"Assistant Reply: {ai_message.content}"
            "\n\n"
            f"Result: {result}"
            "\n\n"
            f"Human Feedback: {user_input}"
        )

        # 返回文本形式的记忆内容，指定来源类型为"agent_history"，并提供摘要方式
        return self.from_text(
            text=memory_content,
            source_type="agent_history",
            how_to_summarize=(
                "if possible, also make clear the link between the command in the"
                " assistant's response and the command result. "
                "Do not mention the human feedback if there is none.",
            ),
        )

    def from_webpage(
        self, content: str, url: str, config: Config, question: str | None = None
    ):
        # 从网页内容中创建记忆，指定来源类型为"webpage"，提供配置和元数据
        return self.from_text(
            text=content,
            source_type="webpage",
            config=config,
            metadata={"location": url},
            question_for_summary=question,
        )
class MemoryItemRelevance(BaseModel):
    """
    Class that encapsulates memory relevance search functionality and data.
    Instances contain a MemoryItem and its relevance scores for a given query.
    """

    # 定义一个类，封装了内存相关性搜索功能和数据，实例包含一个MemoryItem及其对于给定查询的相关性分数
    memory_item: MemoryItem
    for_query: str
    summary_relevance_score: float
    chunk_relevance_scores: list[float]

    @staticmethod
    def of(
        memory_item: MemoryItem, for_query: str, e_query: Embedding | None = None
    ) -> MemoryItemRelevance:
        # 如果给定的e_query为空，则获取查询的嵌入
        e_query = e_query if e_query is not None else get_embedding(for_query)
        # 计算MemoryItem和e_query之间的相关性分数
        _, srs, crs = MemoryItemRelevance.calculate_scores(memory_item, e_query)
        # 返回MemoryItemRelevance对象
        return MemoryItemRelevance(
            for_query=for_query,
            memory_item=memory_item,
            summary_relevance_score=srs,
            chunk_relevance_scores=crs,
        )

    @staticmethod
    def calculate_scores(
        memory: MemoryItem, compare_to: Embedding
    ) -> tuple[float, float, list[float]]:
        """
        Calculates similarity between given embedding and all embeddings of the memory

        Returns:
            float: the aggregate (max) relevance score of the memory
            float: the relevance score of the memory summary
            list: the relevance scores of the memory chunks
        """
        # 计算给定嵌入和内存中所有嵌入之间的相似性
        summary_relevance_score = np.dot(memory.e_summary, compare_to)
        chunk_relevance_scores = np.dot(memory.e_chunks, compare_to).tolist()
        logger.debug(f"Relevance of summary: {summary_relevance_score}")
        logger.debug(f"Relevance of chunks: {chunk_relevance_scores}")

        relevance_scores = [summary_relevance_score, *chunk_relevance_scores]
        logger.debug(f"Relevance scores: {relevance_scores}")
        # 返回内存的最大相关性分数、内存摘要的相关性分数和内存块的相关性分数列表
        return max(relevance_scores), summary_relevance_score, chunk_relevance_scores

    @property
    # 返回记忆项对于给定查询的综合相关性分数，取摘要相关性分数和所有块相关性分数的最大值
    def score(self) -> float:
        """The aggregate relevance score of the memory item for the given query"""
        return max([self.summary_relevance_score, *self.chunk_relevance_scores])

    # 返回记忆项中最相关的块及其相关性分数的元组
    @property
    def most_relevant_chunk(self) -> tuple[str, float]:
        """The most relevant chunk of the memory item + its score for the given query"""
        # 找到块相关性分数中最大值的索引
        i_relmax = np.argmax(self.chunk_relevance_scores)
        # 返回最相关块的内容和相关性分数
        return self.memory_item.chunks[i_relmax], self.chunk_relevance_scores[i_relmax]

    # 返回记忆项的字符串表示
    def __str__(self):
        return (
            # 格式化输出记忆项的摘要和相关性分数，以及所有块的相关性分数
            f"{self.memory_item.summary} ({self.summary_relevance_score}) "
            f"{self.chunk_relevance_scores}"
        )
```