# `.\DB-GPT-src\dbgpt\agent\util\__init__.py`

```py
"""Utils for the agent module."""
# 引入cmp模块中的cmp_string_equal函数作为工具模块的一部分
from .cmp import cmp_string_equal  # noqa: F401

# 声明__ALL__列表，指定在使用from module import *时应该导入的符号列表
__ALL__ = ["cmp_string_equal"]
```