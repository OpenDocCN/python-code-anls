# `.\numpy\numpy\_distributor_init.py`

```
"""
Distributor init file

Distributors: you can add custom code here to support particular distributions
of numpy.

For example, this is a good place to put any BLAS/LAPACK initialization code.

The numpy standard source distribution will not put code in this file, so you
can safely replace this file with your own version.
"""

# 尝试导入本地的 _distributor_init_local 模块
try:
    from . import _distributor_init_local
# 如果导入失败，则忽略该错误继续执行
except ImportError:
    pass
```