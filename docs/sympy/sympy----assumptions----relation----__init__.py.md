# `D:\src\scipysrc\sympy\sympy\assumptions\relation\__init__.py`

```
"""
A module to implement finitary relations [1] as predicate.

References
==========

.. [1] https://en.wikipedia.org/wiki/Finitary_relation

"""

# 定义模块级别的文档字符串，介绍该模块实现有限关系的谓词表示

# 导出的模块级别变量，指定了该模块中可以被导入的公开接口
__all__ = ['BinaryRelation', 'AppliedBinaryRelation']

# 导入所需的类 `BinaryRelation` 和 `AppliedBinaryRelation`，从模块 `.binrel` 中
from .binrel import BinaryRelation, AppliedBinaryRelation
```