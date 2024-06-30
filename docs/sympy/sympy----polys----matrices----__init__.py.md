# `D:\src\scipysrc\sympy\sympy\polys\matrices\__init__.py`

```
"""
sympy.polys.matrices package.

The main export from this package is the DomainMatrix class which is a
lower-level implementation of matrices based on the polys Domains. This
implementation is typically a lot faster than SymPy's standard Matrix class
but is a work in progress and is still experimental.

"""
# 导入DomainMatrix和DM类，从domainmatrix模块中
from .domainmatrix import DomainMatrix, DM
# 定义__all__变量，指定模块中公开的符号列表
__all__ = [
    'DomainMatrix', 'DM',
]
```