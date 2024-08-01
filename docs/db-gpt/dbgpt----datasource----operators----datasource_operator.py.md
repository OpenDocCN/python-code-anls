# `.\DB-GPT-src\dbgpt\datasource\operators\datasource_operator.py`

```py
"""DatasourceOperator class.

Warning: This operator is in development and is not yet ready for production use.
"""
# 导入所需模块和类
from typing import Any
from dbgpt.core.awel import MapOperator
from ..base import BaseConnector

# 定义数据源操作类，继承自MapOperator类，键类型为str，值类型为Any
class DatasourceOperator(MapOperator[str, Any]):
    """The Datasource Operator."""

    def __init__(self, connector: BaseConnector, **kwargs):
        """Create the datasource operator."""
        # 调用父类构造函数初始化操作符
        super().__init__(**kwargs)
        # 设置数据连接器
        self._connector = connector

    async def map(self, input_value: str) -> Any:
        """Execute the query."""
        # 将阻塞函数转换为异步执行，执行查询操作
        return await self.blocking_func_to_async(self.query, input_value)

    def query(self, input_value: str) -> Any:
        """Execute the query."""
        # 调用连接器对象的方法执行查询，并返回结果
        return self._connector.run_to_df(input_value)
```