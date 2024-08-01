# `.\DB-GPT-src\dbgpt\rag\operators\schema_linking.py`

```py
"""Simple schema linking operator.

Warning: This operator is in development and is not yet ready for production use.
"""

# 引入需要的库和模块
from typing import Any, Optional

from dbgpt.core import LLMClient  # 导入LLMClient类
from dbgpt.core.awel import MapOperator  # 导入MapOperator类
from dbgpt.datasource.base import BaseConnector  # 导入BaseConnector类
from dbgpt.rag.index.base import IndexStoreBase  # 导入IndexStoreBase类
from dbgpt.rag.schemalinker.schema_linking import SchemaLinking  # 导入SchemaLinking类


class SchemaLinkingOperator(MapOperator[Any, Any]):
    """The Schema Linking Operator."""

    def __init__(
        self,
        connector: BaseConnector,
        model_name: str,
        llm: LLMClient,
        top_k: int = 5,
        index_store: Optional[IndexStoreBase] = None,
        **kwargs
    ):
        """Create the schema linking operator.

        Args:
            connector (BaseConnector): The connection.
            model_name (str): Name of the model used for schema linking.
            llm (LLMClient): Instance of LLMClient for language model operations.
            top_k (int, optional): Number of top results to retrieve. Defaults to 5.
            index_store (Optional[IndexStoreBase], optional): Index store for schema linking. Defaults to None.
            **kwargs: Additional keyword arguments for MapOperator.
        """
        super().__init__(**kwargs)

        # 初始化SchemaLinking对象，用于表模式链接
        self._schema_linking = SchemaLinking(
            top_k=top_k,
            connector=connector,
            llm=llm,
            model_name=model_name,
            index_store=index_store,
        )

    async def map(self, query: str) -> str:
        """Retrieve the table schemas with llm.

        Args:
            query (str): Query string for schema linking.

        Returns:
            str: Schema information retrieved.
        """
        # 调用SchemaLinking对象的schema_linking_with_llm方法，获取与查询相关的表模式信息
        return str(await self._schema_linking.schema_linking_with_llm(query))
```