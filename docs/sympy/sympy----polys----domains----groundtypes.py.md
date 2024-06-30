# `D:\src\scipysrc\sympy\sympy\polys\domains\groundtypes.py`

```
"""Ground types for various mathematical domains in SymPy. """

# 导入Python内建的模块
import builtins
# 导入SymPy外部的GMPY相关函数和常量
from sympy.external.gmpy import GROUND_TYPES, factorial, sqrt, is_square, sqrtrem

# 定义Python中的整数、实数和复数类型
PythonInteger = builtins.int
PythonReal = builtins.float
PythonComplex = builtins.complex

# 导入Python有理数类
from .pythonrational import PythonRational

# 导入SymPy核心整数函数
from sympy.core.intfunc import (
    igcdex as python_gcdex,
    igcd2 as python_gcd,
    ilcm as python_lcm,
)

# 导入SymPy核心数值类
from sympy.core.numbers import (Float as SymPyReal, Integer as SymPyInteger, Rational as SymPyRational)


# 定义GMPY整数类
class _GMPYInteger:
    def __init__(self, obj):
        pass

# 定义GMPY有理数类
class _GMPYRational:
    def __init__(self, obj):
        pass


# 根据GROUND_TYPES的值选择不同的数值类型库
if GROUND_TYPES == 'gmpy':
    # 导入GMPY2库中的相关函数和类型
    from gmpy2 import (
        mpz as GMPYInteger,
        mpq as GMPYRational,
        numer as gmpy_numer,
        denom as gmpy_denom,
        gcdext as gmpy_gcdex,
        gcd as gmpy_gcd,
        lcm as gmpy_lcm,
        qdiv as gmpy_qdiv,
    )
    # 赋值给全局变量
    gcdex = gmpy_gcdex
    gcd = gmpy_gcd
    lcm = gmpy_lcm

elif GROUND_TYPES == 'flint':
    # 导入FLINT库中的相关类型
    from flint import fmpz as _fmpz
    # 使用_GMPYInteger和_GMPYRational作为GMPY整数和有理数类
    GMPYInteger = _GMPYInteger
    GMPYRational = _GMPYRational
    # 清空其他GMPY相关变量
    gmpy_numer = None
    gmpy_denom = None
    gmpy_gcdex = None
    gmpy_gcd = None
    gmpy_lcm = None
    gmpy_qdiv = None

    # 自定义gcd、gcdex、lcm函数，用于FLINT库
    def gcd(a, b):
        return a.gcd(b)

    def gcdex(a, b):
        x, y, g = python_gcdex(a, b)
        return _fmpz(x), _fmpz(y), _fmpz(g)

    def lcm(a, b):
        return a.lcm(b)

else:
    # 默认情况下使用Python本身的整数和有理数类
    GMPYInteger = _GMPYInteger
    GMPYRational = _GMPYRational
    gmpy_numer = None
    gmpy_denom = None
    gmpy_gcdex = None
    gmpy_gcd = None
    gmpy_lcm = None
    gmpy_qdiv = None
    gcdex = python_gcdex
    gcd = python_gcd
    lcm = python_lcm


# 模块中公开的符号列表
__all__ = [
    'PythonInteger', 'PythonReal', 'PythonComplex',

    'PythonRational',

    'python_gcdex', 'python_gcd', 'python_lcm',

    'SymPyReal', 'SymPyInteger', 'SymPyRational',

    'GMPYInteger', 'GMPYRational', 'gmpy_numer',
    'gmpy_denom', 'gmpy_gcdex', 'gmpy_gcd', 'gmpy_lcm',
    'gmpy_qdiv',

    'factorial', 'sqrt', 'is_square', 'sqrtrem',

    'GMPYInteger', 'GMPYRational',
]
```