# `D:\src\scipysrc\scipy\scipy\special\specfun.py`

```
# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.special` namespace for importing the functions
# included below.
# 导入特定子模块的函数时使用 `scipy.special` 命名空间。

from scipy._lib.deprecation import _sub_module_deprecation

# ruff: noqa: F822
# 禁止使用 flake8 F822 错误检查

__all__ = [
    'clpmn',
    'lpmn',
    'lpn',
    'lqmn',
    'pbdv'
]

# 定义模块中公开的函数名列表

def __dir__():
    return __all__

# 定义一个特殊方法 `__dir__()`，返回模块中公开的所有函数名列表

def __getattr__(name):
    return _sub_module_deprecation(sub_package="special", module="specfun",
                                   private_modules=["_basic", "_specfun"], all=__all__,
                                   attribute=name)

# 定义一个特殊方法 `__getattr__()`，用于处理动态获取模块属性的操作，
# 当属性名在 `__all__` 列表中时，通过 `_sub_module_deprecation` 函数处理，
# 将警告或错误传递给 `_sub_module_deprecation` 处理。
```