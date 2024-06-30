# `D:\src\scipysrc\sympy\sympy\utilities\__init__.py`

```
# 导入本模块中所需的功能和类
"""This module contains some general purpose utilities that are used across
SymPy.
"""
# 从 iterables 模块导入多个函数和类
from .iterables import (flatten, group, take, subsets,
    variations, numbered_symbols, cartes, capture, dict_merge,
    prefixes, postfixes, sift, topological_sort, unflatten,
    has_dups, has_variety, reshape, rotations)

# 从 misc 模块导入 filldedent 函数
from .misc import filldedent

# 从 lambdify 模块导入 lambdify 函数
from .lambdify import lambdify

# 从 decorator 模块导入 threaded, xthreaded, public, memoize_property 函数或装饰器
from .decorator import threaded, xthreaded, public, memoize_property

# 从 timeutils 模块导入 timed 函数
from .timeutils import timed

# 定义 __all__ 列表，包含了所有应该被导出的函数和类名
__all__ = [
    'flatten', 'group', 'take', 'subsets', 'variations', 'numbered_symbols',
    'cartes', 'capture', 'dict_merge', 'prefixes', 'postfixes', 'sift',
    'topological_sort', 'unflatten', 'has_dups', 'has_variety', 'reshape',
    'rotations',

    'filldedent',  # 导出 filldedent 函数

    'lambdify',  # 导出 lambdify 函数

    'threaded', 'xthreaded', 'public', 'memoize_property',  # 导出装饰器和函数

    'timed',  # 导出 timed 函数
]
```