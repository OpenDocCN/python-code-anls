# `.\DB-GPT-src\dbgpt\client\__init__.py`

```py
"""This module is the client of the dbgpt package."""

# 导入 dbgpt 包中的 Client 和 ClientException 类，忽略 F401 错误（未使用警告）
from .client import Client, ClientException  # noqa: F401

# 设置 __ALL__ 变量，指定当使用 from module import * 时应该导入的符号列表
__ALL__ = ["Client", "ClientException"]
```