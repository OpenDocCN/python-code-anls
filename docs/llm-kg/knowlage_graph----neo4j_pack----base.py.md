# `.\knowlage_graph\neo4j_pack\base.py`

```
"""Neo4j Query Engine Pack."""

# 导入所需模块
import os
from typing import Any, Dict, List, Optional
from enum import Enum

# 导入自定义模块
from llama_index.llama_pack.base import BaseLlamaPack
from llama_index.schema import Document
from llama_index.graph_stores import Neo4jGraphStore
from llama_index.llms import OpenAI, AzureOpenAI
from llama_index.embeddings import AzureOpenAIEmbedding
from llama_index import (
    StorageContext,
    ServiceContext,
    KnowledgeGraphIndex,
)
from llama_index import get_response_synthesizer, VectorStoreIndex
from llama_index.text_splitter import SentenceSplitter
from llama_index.retrievers import VectorIndexRetriever, KGTableRetriever


class Neo4jQueryEngineType(str, Enum):
    """Neo4j query engine type"""

    # 定义 Neo4j 查询引擎的类型枚举
    KG_KEYWORD = "keyword"
    KG_HYBRID = "hybrid"
    RAW_VECTOR = "vector"
    RAW_VECTOR_KG_COMBO = "vector_kg"
    KG_QE = "KnowledgeGraphQueryEngine"
    KG_RAG_RETRIEVER = "KnowledgeGraphRAGRetriever"


class Neo4jQueryEnginePack(BaseLlamaPack):
    """Neo4j Query Engine pack."""

    def __init__(
        self,
        username: str,
        password: str,
        url: str,
        database: str,
        docs: List[Document],
        query_engine_type: Optional[Neo4jQueryEngineType] = None,
        **kwargs: Any,
    ):
        """Initialize Neo4j Query Engine Pack."""
        
        # 调用父类的构造函数，初始化参数
        super().__init__(**kwargs)
        
        # 初始化 Neo4j 查询引擎包
        self.username = username
        self.password = password
        self.url = url
        self.database = database
        self.docs = docs
        self.query_engine_type = query_engine_type

        # 初始化连接 Neo4j 图数据库的对象
        self.graph_store = Neo4jGraphStore(
            self.username, self.password, self.url, self.database
        )

        # 初始化 LLM 对象
        self.llm = OpenAI()
        
        # 初始化服务上下文对象
        self.service_context = ServiceContext(
            graph_store=self.graph_store,
            vector_index=None,
            knowledge_graph_index=None,
            response_synthesizer=get_response_synthesizer(),
            sentence_splitter=SentenceSplitter(),
        )
        
        # 初始化查询引擎对象
        self.query_engine = KnowledgeGraphIndex(
            service_context=self.service_context,
            retriever=VectorIndexRetriever(
                vector_index=None, sentence_splitter=SentenceSplitter()
            ),
            rag_retriever=None,
            retriever_cache=None,
        )

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        
        # 返回 Neo4j 查询引擎包中的模块字典
        return {
            "llm": self.llm,
            "service_context": self.service_context,
            "query_engine": self.query_engine,
        }

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline."""
        
        # 运行查询引擎的查询方法
        return self.query_engine.query(*args, **kwargs)


# 导入所需模块
from llama_index import QueryBundle
from llama_index.schema import NodeWithScore
from llama_index.retrievers import BaseRetriever, VectorIndexRetriever, KGTableRetriever


class CustomRetriever(BaseRetriever):
    """Custom retriever that performs both Vector search and Knowledge Graph search"""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        kg_retriever: KGTableRetriever,
        mode: str = "OR",
    ) -> None:
        """Initialize CustomRetriever."""
        
        # 初始化自定义检索器
        self._vector_retriever = vector_retriever
        self._kg_retriever = kg_retriever
        # 检查模式是否有效
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode
    # 定义一个方法 `_retrieve`，接收一个类型为 `QueryBundle` 的参数，并返回一个 `List[NodeWithScore]` 类型的列表
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""
        # 使用向量检索器检索查询包中的节点，返回结果为一个节点和分数的列表
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        # 使用知识图谱检索器检索查询包中的节点，返回结果为一个节点和分数的列表
        kg_nodes = self._kg_retriever.retrieve(query_bundle)

        # 从向量节点列表中提取节点 ID，存储在集合 `vector_ids` 中
        vector_ids = {n.node.node_id for n in vector_nodes}
        # 从知识图谱节点列表中提取节点 ID，存储在集合 `kg_ids` 中
        kg_ids = {n.node.node_id for n in kg_nodes}

        # 创建一个字典 `combined_dict`，将向量检索器和知识图谱检索器返回的节点列表中的节点 ID 映射到节点对象上
        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in kg_nodes})

        # 根据检索模式 `_mode` 进行节点 ID 的组合操作
        if self._mode == "AND":
            # 如果模式为 "AND"，则取向量节点 ID 和知识图谱节点 ID 的交集作为需要检索的节点 ID 集合
            retrieve_ids = vector_ids.intersection(kg_ids)
        else:
            # 如果模式不是 "AND"，则取向量节点 ID 和知识图谱节点 ID 的并集作为需要检索的节点 ID 集合
            retrieve_ids = vector_ids.union(kg_ids)

        # 根据 `combined_dict` 中的节点 ID 提取需要检索的节点，并存储在列表 `retrieve_nodes` 中
        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        # 返回需要检索的节点列表
        return retrieve_nodes
```