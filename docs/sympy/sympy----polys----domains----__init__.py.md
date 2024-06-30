# `D:\src\scipysrc\sympy\sympy\polys\domains\__init__.py`

```
# 数学领域的数学域的实现。
"""Implementation of mathematical domains. """

# 所有公开的类和函数的列表
__all__ = [
    'Domain', 'FiniteField', 'IntegerRing', 'RationalField', 'RealField',
    'ComplexField', 'AlgebraicField', 'PolynomialRing', 'FractionField',
    'ExpressionDomain', 'PythonRational',

    'GF', 'FF', 'ZZ', 'QQ', 'ZZ_I', 'QQ_I', 'RR', 'CC', 'EX', 'EXRAW',
]

# 导入各个数学领域的相关模块和类
from .domain import Domain
from .finitefield import FiniteField, FF, GF
from .integerring import IntegerRing, ZZ
from .rationalfield import RationalField, QQ
from .algebraicfield import AlgebraicField
from .gaussiandomains import ZZ_I, QQ_I
from .realfield import RealField, RR
from .complexfield import ComplexField, CC
from .polynomialring import PolynomialRing
from .fractionfield import FractionField
from .expressiondomain import ExpressionDomain, EX
from .expressionrawdomain import EXRAW
from .pythonrational import PythonRational

# 以下部分是为了向后兼容而导入的模块，因为代码库的某些部分曾经从这里导入，
# 并且有可能下游代码也在使用:
from sympy.external.gmpy import GROUND_TYPES  # noqa: F401

#
# 下面的部分已经过时，只提供向后兼容性支持:
#

# 导入PythonFiniteField模块，提供有限域的Python实现
from .pythonfinitefield import PythonFiniteField
# 导入GMPYFiniteField模块，提供有限域的GMPY实现
from .gmpyfinitefield import GMPYFiniteField
# 导入PythonIntegerRing模块，提供整数环的Python实现
from .pythonintegerring import PythonIntegerRing
# 导入GMPYIntegerRing模块，提供整数环的GMPY实现
from .gmpyintegerring import GMPYIntegerRing
# 导入PythonRationalField模块，提供有理数域的Python实现
from .pythonrationalfield import PythonRationalField
# 导入GMPYRationalField模块，提供有理数域的GMPY实现
from .gmpyrationalfield import GMPYRationalField

# 为了方便访问和导入，将一些新的名称加入到__all__列表中
FF_python = PythonFiniteField
FF_gmpy = GMPYFiniteField

ZZ_python = PythonIntegerRing
ZZ_gmpy = GMPYIntegerRing

QQ_python = PythonRationalField
QQ_gmpy = GMPYRationalField

# 扩展__all__列表，添加新加入的类和函数名，以及向后兼容性导入的内容
__all__.extend((
    'PythonFiniteField', 'GMPYFiniteField', 'PythonIntegerRing',
    'GMPYIntegerRing', 'PythonRational', 'GMPYRationalField',

    'FF_python', 'FF_gmpy', 'ZZ_python', 'ZZ_gmpy', 'QQ_python', 'QQ_gmpy',
))
```