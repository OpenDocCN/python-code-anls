# `D:\src\scipysrc\sympy\sympy\physics\vector\__init__.py`

```
# 定义 __all__ 列表，用于声明模块中公开的符号（变量、函数等）

__all__ = [
    'CoordinateSym', 'ReferenceFrame',  # 符号坐标系和参考框架类

    'Dyadic',  # 双线性形式类

    'Vector',  # 向量类

    'Point',  # 点类

    'cross', 'dot', 'express', 'time_derivative', 'outer',  # 向量运算函数：叉乘、点乘、表达式、时间导数、外积
    'kinematic_equations', 'get_motion_params', 'partial_velocity',  # 运动学方程、运动参数获取、偏速度

    'dynamicsymbols',  # 动态符号

    'vprint', 'vsstrrepr', 'vsprint', 'vpprint', 'vlatex', 'init_vprinting',  # 打印函数：详细打印、字符串表示、简要打印、漂亮打印、LaTeX打印、初始化打印设置

    'curl', 'divergence', 'gradient', 'is_conservative', 'is_solenoidal',  # 场函数：旋度、散度、梯度、保守性、无源性
    'scalar_potential', 'scalar_potential_difference',  # 标量势函数、标量势差函数
]
# 导入各个模块的具体类和函数
from .frame import CoordinateSym, ReferenceFrame
from .dyadic import Dyadic
from .vector import Vector
from .point import Point
from .functions import (cross, dot, express, time_derivative, outer,
        kinematic_equations, get_motion_params, partial_velocity,
        dynamicsymbols)
from .printing import (vprint, vsstrrepr, vsprint, vpprint, vlatex,
        init_vprinting)
from .fieldfunctions import (curl, divergence, gradient, is_conservative,
        is_solenoidal, scalar_potential, scalar_potential_difference)
```