# `D:\src\scipysrc\sympy\sympy\calculus\__init__.py`

```
"""Calculus-related methods."""

# 导入欧拉方程相关函数
from .euler import euler_equations
# 导入奇点相关函数和增长性质判定函数
from .singularities import (singularities, is_increasing,
                            is_strictly_increasing, is_decreasing,
                            is_strictly_decreasing, is_monotonic)
# 导入有限差分权重计算函数和有限差分相关函数
from .finite_diff import finite_diff_weights, apply_finite_diff, differentiate_finite
# 导入周期性判定、非空性判定、凸性判定、静止点、最小值和最大值计算函数
from .util import (periodicity, not_empty_in, is_convex,
                   stationary_points, minimum, maximum)
# 导入区间累积边界类
from .accumulationbounds import AccumBounds

# 导出的符号列表，指定外部可访问的函数和类
__all__ = [
    'euler_equations',  # 欧拉方程函数

    'singularities', 'is_increasing',  # 奇点及增长性质判定函数
    'is_strictly_increasing', 'is_decreasing',
    'is_strictly_decreasing', 'is_monotonic',

    'finite_diff_weights', 'apply_finite_diff', 'differentiate_finite',  # 有限差分函数

    'periodicity', 'not_empty_in', 'is_convex', 'stationary_points',  # 辅助函数
    'minimum', 'maximum',

    'AccumBounds'  # 区间累积边界类
]
```