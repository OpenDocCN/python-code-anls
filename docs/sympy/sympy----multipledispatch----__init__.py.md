# `D:\src\scipysrc\sympy\sympy\multipledispatch\__init__.py`

```
# 从当前包中导入模块 core 中的 dispatch 函数
# 以及从 dispatcher 模块导入 Dispatcher、halt_ordering、restart_ordering
# 和 MDNotImplementedError 这些对象
from .core import dispatch
from .dispatcher import (Dispatcher, halt_ordering, restart_ordering,
    MDNotImplementedError)

# 设置当前模块的版本号为字符串 '0.4.9'
__version__ = '0.4.9'

# 声明当前模块中可以被导出的对象列表
__all__ = [
    'dispatch',  # 导出 dispatch 函数

    'Dispatcher',  # 导出 Dispatcher 类
    'halt_ordering',  # 导出 halt_ordering 函数
    'restart_ordering',  # 导出 restart_ordering 函数
    'MDNotImplementedError',  # 导出 MDNotImplementedError 类
]
```