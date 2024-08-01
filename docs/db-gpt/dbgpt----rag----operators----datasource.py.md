# `.\DB-GPT-src\dbgpt\rag\operators\datasource.py`

```py
"""Datasource operator for RDBMS database."""

# 引入必要的类型声明
from typing import Any, List

# 从dbgpt.core.interface.operators.retriever模块中导入RetrieverOperator类
from dbgpt.core.interface.operators.retriever import RetrieverOperator

# 从dbgpt.datasource.base模块中导入BaseConnector类
from dbgpt.datasource.base import BaseConnector

# 从dbgpt.rag.summary.rdbms_db_summary模块中导入_parse_db_summary函数
from dbgpt.rag.summary.rdbms_db_summary import _parse_db_summary


class DatasourceRetrieverOperator(RetrieverOperator[Any, List[str]]):
    """The Datasource Retriever Operator."""

    def __init__(self, connector: BaseConnector, **kwargs):
        """Create a new DatasourceRetrieverOperator."""
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 初始化连接器属性
        self._connector = connector

    def retrieve(self, input_value: Any) -> List[str]:
        """Retrieve the database summary."""
        # 调用_parse_db_summary函数，传入连接器对象，获取数据库摘要信息
        summary = _parse_db_summary(self._connector)
        # 返回数据库摘要信息（列表形式）
        return summary
```