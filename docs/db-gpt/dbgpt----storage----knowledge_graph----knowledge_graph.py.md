# `.\DB-GPT-src\dbgpt\storage\knowledge_graph\knowledge_graph.py`

```py
"""
Knowledge graph class.
"""
# 导入必要的库
import asyncio  # 异步IO库，用于异步处理任务
import logging  # 日志记录库
import os  # 系统操作库
from typing import List, Optional  # 引入类型提示

from dbgpt._private.pydantic import ConfigDict, Field  # 引入配置相关模块
from dbgpt.core import Chunk, LLMClient  # 引入核心模块和客户端
from dbgpt.rag.transformer.keyword_extractor import KeywordExtractor  # 引入关键词提取器
from dbgpt.rag.transformer.triplet_extractor import TripletExtractor  # 引入三元组提取器
from dbgpt.storage.graph_store.base import GraphStoreBase, GraphStoreConfig  # 引入图存储相关模块
from dbgpt.storage.graph_store.factory import GraphStoreFactory  # 引入图存储工厂
from dbgpt.storage.graph_store.graph import Graph  # 引入图相关模块
from dbgpt.storage.knowledge_graph.base import KnowledgeGraphBase, KnowledgeGraphConfig  # 引入知识图谱相关模块
from dbgpt.storage.vector_store.filters import MetadataFilters  # 引入向量存储相关模块

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class BuiltinKnowledgeGraphConfig(KnowledgeGraphConfig):
    """Builtin knowledge graph config."""

    model_config = ConfigDict(arbitrary_types_allowed=True)  # 模型配置允许任意类型

    llm_client: LLMClient = Field(default=None, description="The default llm client.")  # LLM客户端字段

    model_name: str = Field(default=None, description="The name of llm model.")  # 模型名称字段

    graph_store_type: str = Field(
        default="TuGraph", description="The type of graph store."
    )  # 图存储类型字段


class BuiltinKnowledgeGraph(KnowledgeGraphBase):
    """Builtin knowledge graph class."""

    def __init__(self, config: BuiltinKnowledgeGraphConfig):
        """Create builtin knowledge graph instance."""
        self._config = config  # 设置配置属性
        super().__init__()  # 调用父类构造函数
        self._llm_client = config.llm_client  # 设置LLM客户端属性
        if not self._llm_client:
            raise ValueError("No llm client provided.")  # 如果未提供LLM客户端，则引发值错误异常

        self._model_name = config.model_name  # 设置模型名称属性
        self._triplet_extractor = TripletExtractor(self._llm_client, self._model_name)  # 初始化三元组提取器
        self._keyword_extractor = KeywordExtractor(self._llm_client, self._model_name)  # 初始化关键词提取器
        self._graph_store_type = (
            os.getenv("GRAPH_STORE_TYPE", "TuGraph") or config.graph_store_type
        )  # 获取图存储类型，优先使用环境变量中的类型，否则使用配置中的类型

        def configure(cfg: GraphStoreConfig):
            cfg.name = self._config.name  # 设置图存储配置的名称属性
            cfg.embedding_fn = self._config.embedding_fn  # 设置图存储配置的嵌入函数属性

        self._graph_store: GraphStoreBase = GraphStoreFactory.create(
            self._graph_store_type, configure
        )  # 创建图存储对象

    def load_document(self, chunks: List[Chunk]) -> List[str]:
        """Extract and persist triplets to graph store."""

        async def process_chunk(chunk):
            triplets = await self._triplet_extractor.extract(chunk.content)  # 提取文档块中的三元组
            for triplet in triplets:
                self._graph_store.insert_triplet(*triplet)  # 将三元组插入图存储
            logger.info(f"load {len(triplets)} triplets from chunk {chunk.chunk_id}")  # 记录日志信息
            return chunk.chunk_id

        # 等待异步任务完成
        tasks = [process_chunk(chunk) for chunk in chunks]  # 创建任务列表
        loop = asyncio.new_event_loop()  # 创建新的事件循环
        asyncio.set_event_loop(loop)  # 设置事件循环
        result = loop.run_until_complete(asyncio.gather(*tasks))  # 执行异步任务
        loop.close()  # 关闭事件循环
        return result  # 返回结果列表
    async def aload_document(self, chunks: List[Chunk]) -> List[str]:  # type: ignore
        """Extract and persist triplets to graph store.

        Args:
            chunks: List[Chunk]: document chunks.
        Return:
            List[str]: chunk ids.
        """
        # 遍历文档的每个片段
        for chunk in chunks:
            # 提取当前片段的三元组
            triplets = await self._triplet_extractor.extract(chunk.content)
            # 将三元组插入到图数据库中
            for triplet in triplets:
                self._graph_store.insert_triplet(*triplet)
            # 记录日志，指示从当前片段加载了多少个三元组
            logger.info(f"load {len(triplets)} triplets from chunk {chunk.chunk_id}")
        # 返回所有片段的 chunk_id 组成的列表
        return [chunk.chunk_id for chunk in chunks]

    def similar_search_with_scores(
        self,
        text,
        topk,
        score_threshold: float,
        filters: Optional[MetadataFilters] = None,
    ) -> List[Chunk]:
        """Search neighbours on knowledge graph."""
        # 同步的相似搜索不支持，抛出异常
        raise Exception("Sync similar_search_with_scores not supported")

    async def asimilar_search_with_scores(
        self,
        text,
        topk,
        score_threshold: float,
        filters: Optional[MetadataFilters] = None,
    ) -> List[Chunk]:
        """Search neighbours on knowledge graph."""
        # 如果没有定义过滤器，记录日志表明知识图谱上的过滤器暂不支持
        if not filters:
            logger.info("Filters on knowledge graph not supported yet")

        # 提取关键词并探索图数据库
        keywords = await self._keyword_extractor.extract(text)
        subgraph = self._graph_store.explore(keywords, limit=topk)
        # 记录日志，指示从关键词中搜索出的子图的数据量
        logger.info(f"Search subgraph from {len(keywords)} keywords")

        # 构建内容，描述从知识图谱检索的子图数据
        content = (
            "The following vertices and edges data after [Subgraph Data] "
            "are retrieved from the knowledge graph based on the keywords:\n"
            f"Keywords:\n{','.join(keywords)}\n"
            "---------------------\n"
            "You can refer to the sample vertices and edges to understand "
            "the real knowledge graph data provided by [Subgraph Data].\n"
            "Sample vertices:\n"
            "(alice)\n"
            "(bob:{age:28})\n"
            '(carry:{age:18;role:"teacher"})\n\n'
            "Sample edges:\n"
            "(alice)-[reward]->(alice)\n"
            '(alice)-[notify:{method:"email"}]->'
            '(carry:{age:18;role:"teacher"})\n'
            '(bob:{age:28})-[teach:{course:"math";hour:180}]->(alice)\n'
            "---------------------\n"
            f"Subgraph Data:\n{subgraph.format()}\n"
        )
        # 返回包含内容和元数据的 Chunk 对象列表
        return [Chunk(content=content, metadata=subgraph.schema())]

    def query_graph(self, limit: Optional[int] = None) -> Graph:
        """Query graph."""
        # 查询图数据库，返回完整的图数据
        return self._graph_store.get_full_graph(limit)

    def delete_vector_name(self, index_name: str):
        """Delete vector name."""
        # 记录日志，指示正在删除的图索引名称
        logger.info(f"Remove graph index {index_name}")
        # 调用图数据库的删除方法
        self._graph_store.drop()
```