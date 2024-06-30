# `D:\src\scipysrc\scipy\scipy\_distributor_init.py`

```
""" Distributor init file

Distributors: you can replace the contents of this file with your own custom
code to support particular distributions of SciPy.

For example, this is a good place to put any checks for hardware requirements
or BLAS/LAPACK library initialization.

The SciPy standard source distribution will not put code in this file beyond
the try-except import of `_distributor_init_local` (which is not part of a
standard source distribution), so you can safely replace this file with your
own version.
"""

# 尝试导入本地的 `_distributor_init_local` 模块，忽略 F401 错误（未使用的导入）
try:
    from . import _distributor_init_local  # noqa: F401
# 如果导入失败，则忽略错误，继续执行
except ImportError:
    pass
```