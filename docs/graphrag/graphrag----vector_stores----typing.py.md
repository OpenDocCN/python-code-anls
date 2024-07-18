# `.\graphrag\graphrag\vector_stores\typing.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A package containing the supported vector store types."""

from enum import Enum  # 导入枚举类型的支持
from typing import ClassVar  # 导入类型提示中的ClassVar

from .azure_ai_search import AzureAISearch  # 导入Azure AI Search类
from .lancedb import LanceDBVectorStore  # 导入LanceDBVectorStore类


class VectorStoreType(str, Enum):
    """The supported vector store types."""
    
    LanceDB = "lancedb"  # 定义枚举成员 LanceDB，对应字符串 "lancedb"
    AzureAISearch = "azure_ai_search"  # 定义枚举成员 AzureAISearch，对应字符串 "azure_ai_search"


class VectorStoreFactory:
    """A factory class for creating vector stores."""
    
    vector_store_types: ClassVar[dict[str, type]] = {}  # 类变量，用于存储不同类型的向量存储类

    @classmethod
    def register(cls, vector_store_type: str, vector_store: type):
        """Register a vector store type."""
        cls.vector_store_types[vector_store_type] = vector_store  # 将给定类型的向量存储类注册到类变量中

    @classmethod
    def get_vector_store(
        cls, vector_store_type: VectorStoreType | str, kwargs: dict
    ) -> LanceDBVectorStore | AzureAISearch:
        """Get the vector store type from a string."""
        match vector_store_type:  # 使用match表达式匹配向量存储类型
            case VectorStoreType.LanceDB:  # 如果是 LanceDB 类型
                return LanceDBVectorStore(**kwargs)  # 创建并返回 LanceDBVectorStore 对象
            case VectorStoreType.AzureAISearch:  # 如果是 AzureAISearch 类型
                return AzureAISearch(**kwargs)  # 创建并返回 AzureAISearch 对象
            case _:  # 对于其他情况
                if vector_store_type in cls.vector_store_types:  # 如果类型存在于注册的向量存储类中
                    return cls.vector_store_types[vector_store_type](**kwargs)  # 创建并返回对应的向量存储对象
                msg = f"Unknown vector store type: {vector_store_type}"  # 构造错误消息
                raise ValueError(msg)  # 抛出值错误异常，指示未知的向量存储类型
```