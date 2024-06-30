# `D:\src\scipysrc\sympy\sympy\polys\domains\pythonrational.py`

```
"""
Rational number type based on Python integers.

The PythonRational class from here has been moved to
sympy.external.pythonmpq

This module is just left here for backwards compatibility.
"""

# 从 sympy.core.numbers 导入 Rational 类，该类用于表示有理数
from sympy.core.numbers import Rational
# 从 sympy.core.sympify 导入 _sympy_converter 函数，用于处理符号计算表达式的转换
from sympy.core.sympify import _sympy_converter
# 从 sympy.utilities 导入 public 函数，用于将变量公开为外部接口
from sympy.utilities import public
# 从 sympy.external.pythonmpq 导入 PythonMPQ 类，该类实现了 Python 的有理数类型
from sympy.external.pythonmpq import PythonMPQ

# 将 PythonMPQ 类公开为 PythonRational 类，用于向后兼容
PythonRational = public(PythonMPQ)

# 定义函数 sympify_pythonrational，将 PythonRational 对象转换为 Rational 对象
def sympify_pythonrational(arg):
    return Rational(arg.numerator, arg.denominator)

# 将 PythonRational 对象注册到 _sympy_converter 字典中，以便进行符号计算表达式的转换
_sympy_converter[PythonRational] = sympify_pythonrational
```