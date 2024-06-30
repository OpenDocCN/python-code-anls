# `D:\src\scipysrc\sympy\sympy\polys\domains\gmpyfinitefield.py`

```
"""Implementation of :class:`GMPYFiniteField` class. """

# 导入所需模块和类
from sympy.polys.domains.finitefield import FiniteField
from sympy.polys.domains.gmpyintegerring import GMPYIntegerRing
from sympy.utilities import public

# 声明一个公共类 GMPYFiniteField，继承自 FiniteField 类
@public
class GMPYFiniteField(FiniteField):
    """Finite field based on GMPY integers. """

    # 类属性，定义别名为 'FF_gmpy'
    alias = 'FF_gmpy'

    # 初始化方法，接受 mod 和 symmetric 两个参数
    def __init__(self, mod, symmetric=True):
        # 调用父类的初始化方法，传递 mod 参数和 GMPYIntegerRing() 实例以及 symmetric 参数
        super().__init__(mod, GMPYIntegerRing(), symmetric)
```