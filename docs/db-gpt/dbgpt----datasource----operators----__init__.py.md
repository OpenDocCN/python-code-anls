# `.\DB-GPT-src\dbgpt\datasource\operators\__init__.py`

```py
"""Datasource operators."""
# 导入 datasource_operator 模块中的 DatasourceOperator 类，忽略 F401 类型的导入警告
from .datasource_operator import DatasourceOperator  # noqa: F401

# 模块全局变量，指定导出的符号，只有 DatasourceOperator 类被导出
__ALL__ = ["DatasourceOperator"]
```