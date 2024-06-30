# `D:\src\scipysrc\scipy\scipy\sparse\linalg\eigen.py`

```
# 导入 _sub_module_deprecation 函数，用于处理子模块的废弃警告
from scipy._lib.deprecation import _sub_module_deprecation

# 定义模块中公开的函数和类列表，用于限制公开的接口
__all__ = [  # noqa: F822
    'ArpackError', 'ArpackNoConvergence', 'ArpackError',  # 列出公开的异常和函数
    'eigs', 'eigsh', 'lobpcg', 'svds', 'test'  # 列出公开的函数
]


def __dir__():
    return __all__  # 返回模块中公开的函数和类列表


def __getattr__(name):
    # 当试图获取不存在的属性时，调用 _sub_module_deprecation 函数
    return _sub_module_deprecation(sub_package="sparse.linalg", module="eigen",
                                   private_modules=["_eigen"], all=__all__,
                                   attribute=name)
```