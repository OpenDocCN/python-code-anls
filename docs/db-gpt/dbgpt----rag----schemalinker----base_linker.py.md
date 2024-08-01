# `.\DB-GPT-src\dbgpt\rag\schemalinker\base_linker.py`

```py
"""Base Linker."""

from abc import ABC, abstractmethod
from typing import List


class BaseSchemaLinker(ABC):
    """定义一个抽象基类 BaseSchemaLinker，继承自 ABC（Abstract Base Class）。"""

    def schema_linking(self, query: str) -> List:
        """调用内部方法 _schema_linking 处理查询，返回数据库 schema 的列表。

        Args:
            query (str): 查询文本
        Returns:
            List: schema 列表
        """
        return self._schema_linking(query)

    def schema_linking_with_vector_db(self, query: str) -> List:
        """调用内部方法 _schema_linking_with_vector_db 处理查询，返回带有向量数据库的 schema 列表。

        Args:
            query (str): 查询文本
        Returns:
            List: schema 列表
        """
        return self._schema_linking_with_vector_db(query)

    async def schema_linking_with_llm(self, query: str) -> List:
        """异步调用内部方法 _schema_linking_with_llm 处理查询，返回带有 LLN 的 schema 列表。

        Args:
            query(str): 查询文本
        Returns:
            List: schema 列表
        """
        return await self._schema_linking_with_llm(query)

    @abstractmethod
    def _schema_linking(self, query: str) -> List:
        """抽象方法：获取数据库 schema 信息。

        Args:
            query (str): 查询文本
        Returns:
            List: schema 列表
        """

    @abstractmethod
    def _schema_linking_with_vector_db(self, query: str) -> List:
        """抽象方法：使用向量数据库查询 schema 信息。

        Args:
            query (str): 查询文本
        Returns:
            List: schema 列表
        """

    @abstractmethod
    async def _schema_linking_with_llm(self, query: str) -> List:
        """抽象方法：使用 LLN 查询 schema 信息。

        Args:
            query (str): 查询文本
        Returns:
            List: schema 列表
        """
```