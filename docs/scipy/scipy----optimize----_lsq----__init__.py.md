# `D:\src\scipysrc\scipy\scipy\optimize\_lsq\__init__.py`

```
"""This module contains least-squares algorithms."""
# 导入最小二乘算法模块，用于处理线性最小二乘问题
from .least_squares import least_squares
# 导入线性最小二乘问题的求解模块
from .lsq_linear import lsq_linear

# 将模块中公开的函数名列出，方便外部使用
__all__ = ['least_squares', 'lsq_linear']
```