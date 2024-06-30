# `D:\src\scipysrc\sympy\sympy\external\tests\test_pythonmpq.py`

```
"""
test_pythonmpq.py

Test the PythonMPQ class for consistency with gmpy2's mpq type. If gmpy2 is
installed run the same tests for both.
"""
# 导入所需的模块和类
from fractions import Fraction  # 导入分数计算模块中的分数类
from decimal import Decimal  # 导入十进制计算模块中的十进制类
import pickle  # 导入用于序列化和反序列化对象的 pickle 模块
from typing import Callable, List, Tuple, Type  # 导入用于类型提示的模块和类

# 导入用于测试的函数和类
from sympy.testing.pytest import raises  # 导入用于测试时的异常处理类

# 导入 PythonMPQ 类
from sympy.external.pythonmpq import PythonMPQ

# 如果 gmpy2 模块可用，则导入 mpq 和 mpz 类型
rational_types: List[Tuple[Callable, Type, Callable, Type]]
rational_types = [(PythonMPQ, PythonMPQ, int, int)]
try:
    from gmpy2 import mpq, mpz
    rational_types.append((mpq, type(mpq(1)), mpz, type(mpz(1))))
except ImportError:
    pass


def test_PythonMPQ():
    #
    # Test PythonMPQ and also mpq if gmpy/gmpy2 is installed.
    #
```