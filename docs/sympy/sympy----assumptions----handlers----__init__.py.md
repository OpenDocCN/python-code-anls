# `D:\src\scipysrc\sympy\sympy\assumptions\handlers\__init__.py`

```
"""
Multipledispatch handlers for ``Predicate`` are implemented here.
Handlers in this module are not directly imported to other modules in
order to avoid circular import problem.
"""

# 从当前包的common模块中导入AskHandler、CommonHandler和test_closed_group
from .common import (AskHandler, CommonHandler,
    test_closed_group)

# 声明这些名称在模块外可见的列表
__all__ = [
    'AskHandler', 'CommonHandler',
    'test_closed_group'
]
```