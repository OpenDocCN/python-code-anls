# `.\DB-GPT-src\dbgpt\storage\graph_store\__init__.py`

```py
# Graph Store Module.
"""Graph Store Module."""
from typing import Tuple, Type


def _import_memgraph() -> Tuple[Type, Type]:
    # 导入内存图存储模块
    from dbgpt.storage.graph_store.memgraph_store import (
        MemoryGraphStore,
        MemoryGraphStoreConfig,
    )

    return MemoryGraphStore, MemoryGraphStoreConfig


def _import_tugraph() -> Tuple[Type, Type]:
    # 导入TuGraph存储模块
    from dbgpt.storage.graph_store.tugraph_store import TuGraphStore, TuGraphStoreConfig

    return TuGraphStore, TuGraphStoreConfig


def _import_neo4j() -> Tuple[Type, Type]:
    # 导入Neo4j存储模块
    from dbgpt.storage.graph_store.neo4j_store import Neo4jStore, Neo4jStoreConfig

    return Neo4jStore, Neo4jStoreConfig


def __getattr__(name: str) -> Tuple[Type, Type]:
    # 根据属性名动态获取对应的存储模块类型和配置类型
    if name == "Memory":
        return _import_memgraph()
    elif name == "TuGraph":
        return _import_tugraph()
    elif name == "Neo4j":
        return _import_neo4j()
    else:
        raise AttributeError(f"Could not find: {name}")


__all__ = ["Memory", "TuGraph", "Neo4j"]
```