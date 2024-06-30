# `D:\src\scipysrc\scipy\scipy\linalg\basic.py`

```
# 以下代码文件不是为公共使用而设计的，并且将在 SciPy v2.0.0 中被移除。
# 使用 `scipy.linalg` 命名空间来导入下面列出的函数。

# 从 scipy._lib.deprecation 模块中导入 _sub_module_deprecation 函数
from scipy._lib.deprecation import _sub_module_deprecation

# 定义 __all__ 列表，其中包含了将被公开的函数名
__all__ = [
    'solve', 'solve_triangular', 'solveh_banded', 'solve_banded',
    'solve_toeplitz', 'solve_circulant', 'inv', 'det', 'lstsq',
    'pinv', 'pinvh', 'matrix_balance', 'matmul_toeplitz',
    'get_lapack_funcs', 'LinAlgError', 'LinAlgWarning',
]

# 定义 __dir__() 函数，返回 __all__ 列表，用于模块的自省
def __dir__():
    return __all__

# 定义 __getattr__(name) 函数，处理属性的获取
def __getattr__(name):
    # 调用 _sub_module_deprecation 函数，标记子模块 linalg.basic._basic 的使用已过时
    return _sub_module_deprecation(sub_package="linalg", module="basic",
                                   private_modules=["_basic"], all=__all__,
                                   attribute=name)
```