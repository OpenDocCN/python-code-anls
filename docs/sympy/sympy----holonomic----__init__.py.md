# `D:\src\scipysrc\sympy\sympy\holonomic\__init__.py`

```
"""
The :py:mod:`~sympy.holonomic` module is intended to deal with holonomic functions along
with various operations on them like addition, multiplication, composition,
integration and differentiation. The module also implements various kinds of
conversions such as converting holonomic functions to a different form and the
other way around.
"""

# 导入 holonomic 模块中的特定对象和函数
from .holonomic import (DifferentialOperator, HolonomicFunction, DifferentialOperators,
    from_hyper, from_meijerg, expr_to_holonomic)
# 导入 recurrence 模块中的对象
from .recurrence import RecurrenceOperators, RecurrenceOperator, HolonomicSequence

# 将以下对象和函数添加到模块的公开接口中
__all__ = [
    'DifferentialOperator', 'HolonomicFunction', 'DifferentialOperators',
    'from_hyper', 'from_meijerg', 'expr_to_holonomic',

    'RecurrenceOperators', 'RecurrenceOperator', 'HolonomicSequence',
]
```