# `D:\src\scipysrc\sympy\sympy\assumptions\__init__.py`

```
"""
A module to implement logical predicates and assumption system.
"""

# 从当前目录中导入以下模块和类
from .assume import (
    AppliedPredicate, Predicate, AssumptionsContext, assuming,
    global_assumptions
)
# 从当前目录中导入 ask、Q、register_handler 和 remove_handler 函数
from .ask import Q, ask, register_handler, remove_handler
# 从当前目录中导入 refine 函数
from .refine import refine
# 从当前目录中导入 BinaryRelation 和 AppliedBinaryRelation 类
from .relation import BinaryRelation, AppliedBinaryRelation

# 模块中对外暴露的变量和类列表
__all__ = [
    'AppliedPredicate', 'Predicate', 'AssumptionsContext', 'assuming',
    'global_assumptions', 'Q', 'ask', 'register_handler', 'remove_handler',
    'refine',
    'BinaryRelation', 'AppliedBinaryRelation'
]
```