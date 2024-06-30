# `D:\src\scipysrc\scipy\scipy\io\matlab\miobase.py`

```
# 导入 `_sub_module_deprecation` 函数，用于处理子模块的废弃警告
# 用于定义当前模块中公开的类和函数列表，供外部使用，包括异常和警告类
from scipy._lib.deprecation import _sub_module_deprecation

# 定义模块中公开的类和函数名称列表，用于 `__all__` 属性
__all__ = ["MatReadError", "MatReadWarning", "MatWriteError"]  # noqa: F822

# 定义 `__dir__()` 函数，返回当前模块中公开的所有类和函数名称列表
def __dir__():
    return __all__


# 定义 `__getattr__(name)` 函数，处理动态属性访问
def __getattr__(name):
    return _sub_module_deprecation(
        sub_package="io.matlab",  # 子模块的名称空间
        module="miobase",         # 当前模块的名称
        private_modules=["_miobase"],  # 私有模块列表
        all=__all__,              # 当前模块中公开的所有类和函数名称列表
        attribute=name            # 要访问的属性名称
    )
```