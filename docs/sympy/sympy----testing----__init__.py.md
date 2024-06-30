# `D:\src\scipysrc\sympy\sympy\testing\__init__.py`

```
"""This module contains code for running the tests in SymPy."""  # 模块级文档字符串，描述了该模块的作用和内容


from .runtests import doctest  # 导入 runtests 模块中的 doctest 符号
from .runtests_pytest import test  # 导入 runtests_pytest 模块中的 test 符号


__all__ = [  # 定义模块公开的符号列表，只包含 test 和 doctest
    'test', 'doctest',
]
```