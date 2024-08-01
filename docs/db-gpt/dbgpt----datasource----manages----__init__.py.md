# `.\DB-GPT-src\dbgpt\datasource\manages\__init__.py`

```py
"""This module is responsible for managing the connectors."""

# 导入 ConnectorManager 类从 connector_manager 模块，忽略 F401 类型的 lint 错误
from .connector_manager import ConnectorManager  # noqa: F401

# 将 ConnectorManager 加入 __ALL__ 列表，以便在 from module import * 时被导入
__ALL__ = ["ConnectorManager"]
```