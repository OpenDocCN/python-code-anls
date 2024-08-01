# `.\DB-GPT-src\dbgpt\storage\vector_store\__init__.py`

```py
"""Vector Store Module."""
# 导入类型提示工具
from typing import Tuple, Type


# 定义私有函数，用于导入PGVector相关类
def _import_pgvector() -> Tuple[Type, Type]:
    # 从dbgpt.storage.vector_store.pgvector_store模块中导入PGVectorConfig和PGVectorStore类
    from dbgpt.storage.vector_store.pgvector_store import PGVectorConfig, PGVectorStore

    return PGVectorStore, PGVectorConfig


# 定义私有函数，用于导入Milvus相关类
def _import_milvus() -> Tuple[Type, Type]:
    # 从dbgpt.storage.vector_store.milvus_store模块中导入MilvusStore和MilvusVectorConfig类
    from dbgpt.storage.vector_store.milvus_store import MilvusStore, MilvusVectorConfig

    return MilvusStore, MilvusVectorConfig


# 定义私有函数，用于导入Chroma相关类
def _import_chroma() -> Tuple[Type, Type]:
    # 从dbgpt.storage.vector_store.chroma_store模块中导入ChromaStore和ChromaVectorConfig类
    from dbgpt.storage.vector_store.chroma_store import ChromaStore, ChromaVectorConfig

    return ChromaStore, ChromaVectorConfig


# 定义私有函数，用于导入Weaviate相关类
def _import_weaviate() -> Tuple[Type, Type]:
    # 从dbgpt.storage.vector_store.weaviate_store模块中导入WeaviateStore和WeaviateVectorConfig类
    from dbgpt.storage.vector_store.weaviate_store import (
        WeaviateStore,
        WeaviateVectorConfig,
    )

    return WeaviateStore, WeaviateVectorConfig


# 定义私有函数，用于导入OceanBase相关类
def _import_oceanbase() -> Tuple[Type, Type]:
    # 从dbgpt.storage.vector_store.oceanbase_store模块中导入OceanBaseConfig和OceanBaseStore类
    from dbgpt.storage.vector_store.oceanbase_store import (
        OceanBaseConfig,
        OceanBaseStore,
    )

    return OceanBaseStore, OceanBaseConfig


# 定义私有函数，用于导入ElasticSearch相关类
def _import_elastic() -> Tuple[Type, Type]:
    # 从dbgpt.storage.vector_store.elastic_store模块中导入ElasticsearchVectorConfig和ElasticStore类
    from dbgpt.storage.vector_store.elastic_store import (
        ElasticsearchVectorConfig,
        ElasticStore,
    )

    return ElasticStore, ElasticsearchVectorConfig


# 定义私有函数，用于导入内置知识图谱相关类
def _import_builtin_knowledge_graph() -> Tuple[Type, Type]:
    # 从dbgpt.storage.knowledge_graph.knowledge_graph模块中导入BuiltinKnowledgeGraph和BuiltinKnowledgeGraphConfig类
    from dbgpt.storage.knowledge_graph.knowledge_graph import (
        BuiltinKnowledgeGraph,
        BuiltinKnowledgeGraphConfig,
    )

    return BuiltinKnowledgeGraph, BuiltinKnowledgeGraphConfig


# 定义私有函数，用于导入OpenSPG相关类
def _import_openspg() -> Tuple[Type, Type]:
    # 从dbgpt.storage.knowledge_graph.open_spg模块中导入OpenSPG和OpenSPGConfig类
    from dbgpt.storage.knowledge_graph.open_spg import OpenSPG, OpenSPGConfig

    return OpenSPG, OpenSPGConfig


# 定义私有函数，用于导入全文文档存储相关类
def _import_full_text() -> Tuple[Type, Type]:
    # 从dbgpt.storage.full_text.elasticsearch模块中导入ElasticDocumentConfig和ElasticDocumentStore类
    from dbgpt.storage.full_text.elasticsearch import (
        ElasticDocumentConfig,
        ElasticDocumentStore,
    )

    return ElasticDocumentStore, ElasticDocumentConfig


# 定义特殊函数__getattr__，根据名称动态返回对应的类
def __getattr__(name: str) -> Tuple[Type, Type]:
    if name == "Chroma":
        return _import_chroma()
    elif name == "Milvus":
        return _import_milvus()
    elif name == "Weaviate":
        return _import_weaviate()
    elif name == "PGVector":
        return _import_pgvector()
    elif name == "OceanBase":
        return _import_oceanbase()
    elif name == "ElasticSearch":
        return _import_elastic()
    elif name == "KnowledgeGraph":
        return _import_builtin_knowledge_graph()
    elif name == "OpenSPG":
        return _import_openspg()
    elif name == "FullText":
        return _import_full_text()
    else:
        # 如果名称不匹配任何已知的模块，抛出异常
        raise AttributeError(f"Could not find: {name}")


# 定义向量存储相关模块列表
__vector_store__ = [
    "Chroma",
    "Milvus",
    "Weaviate",
    "OceanBase",
    "PGVector",
    "ElasticSearch",
]

# 定义知识图谱相关模块列表
__knowledge_graph__ = ["KnowledgeGraph", "OpenSPG"]

# 定义文档存储相关模块列表
__document_store__ = ["FullText"]

# 定义__all__变量，包含所有向量存储、知识图谱和文档存储相关模块
__all__ = __vector_store__ + __knowledge_graph__ + __document_store__
```