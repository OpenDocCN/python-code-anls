# `D:\src\scipysrc\sympy\sympy\strategies\branch\__init__.py`

```
# 导入当前包下的 traverse 模块
from . import traverse
# 从 core 模块中导入以下函数和类
from .core import (
    condition, debug, multiplex, exhaust, notempty,
    chain, onaction, sfilter, yieldify, do_one, identity)
# 从 tools 模块中导入 canon 函数
from .tools import canon

# 定义了当前模块中可以被外部访问的符号列表
__all__ = [
    'traverse',  # 可导出 traverse 模块
    'condition', 'debug', 'multiplex', 'exhaust', 'notempty', 'chain',
    'onaction', 'sfilter', 'yieldify', 'do_one', 'identity',  # 可导出 core 模块中的这些函数和类
    'canon',  # 可导出 tools 模块中的 canon 函数
]
```