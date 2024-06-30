# `D:\src\scipysrc\scipy\scipy\io\matlab\byteordercodes.py`

```
# 这个文件不是公共使用的，将在 SciPy v2.0.0 中移除。
# 请使用 `scipy.io.matlab` 命名空间来导入以下列出的函数。

# 导入 `_sub_module_deprecation` 函数，用于处理子模块的弃用警告
from scipy._lib.deprecation import _sub_module_deprecation

# 初始化空列表 `__all__`，用于存储模块中公开的对象名称
__all__: list[str] = []


# 定义 `__dir__` 特殊函数，返回 `__all__` 列表，影响 `dir()` 函数的行为
def __dir__():
    return __all__


# 定义 `__getattr__` 特殊函数，处理动态属性访问时的情况
def __getattr__(name):
    return _sub_module_deprecation(sub_package="io.matlab", module="byteordercodes",
                                   private_modules=["_byteordercodes"], all=__all__,
                                   attribute=name)
```