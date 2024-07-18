# `.\graphrag\graphrag\vector_stores\__init__.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A package containing vector-storage implementations."""

# 导入 AzureAISearch 类，实现与 Azure AI 搜索相关的功能
from .azure_ai_search import AzureAISearch
# 导入基础向量存储相关的类和接口
from .base import BaseVectorStore, VectorStoreDocument, VectorStoreSearchResult
# 导入 LanceDBVectorStore 类，提供与 LanceDB 数据库相关的向量存储功能
from .lancedb import LanceDBVectorStore
# 导入向量存储工厂接口和向量存储类型接口
from .typing import VectorStoreFactory, VectorStoreType

# __all__ 列表定义了在 import * 时导出的符号列表
__all__ = [
    "AzureAISearch",
    "BaseVectorStore",
    "LanceDBVectorStore",
    "VectorStoreDocument",
    "VectorStoreFactory",
    "VectorStoreSearchResult",
    "VectorStoreType",
]
```