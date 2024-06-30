# `D:\src\scipysrc\scipy\scipy\sparse\linalg\dsolve.py`

```
# 导入 `_sub_module_deprecation` 函数，用于处理子模块的废弃警告
from scipy._lib.deprecation import _sub_module_deprecation

# 定义 `__all__` 列表，指定在 `from module import *` 时导入的符号列表
__all__ = [  # noqa: F822
    'MatrixRankWarning', 'SuperLU', 'factorized',
    'spilu', 'splu', 'spsolve',
    'spsolve_triangular', 'use_solver', 'test'
]


# 定义 `__dir__()` 函数，返回模块的 `__all__` 列表，用于 `dir(module)` 操作
def __dir__():
    return __all__


# 定义 `__getattr__(name)` 函数，用于在访问不存在的属性时执行操作
def __getattr__(name):
    # 调用 `_sub_module_deprecation` 函数，向用户发出废弃警告，建议使用 `sparse.linalg` 的 `dsolve` 子模块
    # private_modules 参数指定废弃的私有模块，all 参数传递 `__all__` 列表，attribute 参数传递请求的属性名
    return _sub_module_deprecation(sub_package="sparse.linalg", module="dsolve",
                                   private_modules=["_dsolve"], all=__all__,
                                   attribute=name)
```