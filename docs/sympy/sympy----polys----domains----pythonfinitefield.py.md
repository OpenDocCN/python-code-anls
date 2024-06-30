# `D:\src\scipysrc\sympy\sympy\polys\domains\pythonfinitefield.py`

```
"""
Implementation of :class:`PythonFiniteField` class.
"""

# 导入所需模块和类
from sympy.polys.domains.finitefield import FiniteField
from sympy.polys.domains.pythonintegerring import PythonIntegerRing
from sympy.utilities import public

# 使用 @public 装饰器使得类可以被公开访问
@public
# 定义 PythonFiniteField 类，继承自 FiniteField 类
class PythonFiniteField(FiniteField):
    """Finite field based on Python's integers."""

    # 类别名
    alias = 'FF_python'

    # 初始化方法，接收 mod 和 symmetric 参数
    def __init__(self, mod, symmetric=True):
        # 调用父类的初始化方法
        super().__init__(mod, PythonIntegerRing(), symmetric)
```