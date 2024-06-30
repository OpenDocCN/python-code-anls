# `D:\src\scipysrc\sympy\sympy\physics\__init__.py`

```
"""
A module that helps solving problems in physics.
"""

# 导入本地模块中的单位
from . import units
# 从本地模块中导入矩阵相关的函数和变量
from .matrices import mgamma, msigma, minkowski_tensor, mdft

# __all__ 列表定义了在使用 from module import * 时需要导入的符号
__all__ = [
    'units',                 # 导出 units 模块
    'mgamma', 'msigma',      # 导出矩阵相关的几个函数和变量
    'minkowski_tensor',
    'mdft',
]
```