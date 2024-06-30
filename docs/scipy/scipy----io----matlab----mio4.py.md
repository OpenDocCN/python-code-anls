# `D:\src\scipysrc\scipy\scipy\io\matlab\mio4.py`

```
# 导入需要的模块，此文件不适合公共使用，将在 SciPy v2.0.0 中移除。
# 使用 `scipy.io.matlab` 命名空间来导入下面列出的函数。

from scipy._lib.deprecation import _sub_module_deprecation

# 定义一个空的公开接口列表，用于存储模块公开的所有符号
__all__: list[str] = []

# 自定义特殊方法 `__dir__()`，返回模块内所有公开符号的列表
def __dir__():
    return __all__

# 自定义特殊方法 `__getattr__(name)`，用于处理属性访问，当属性未找到时会调用 `_sub_module_deprecation` 函数
def __getattr__(name):
    # 调用 `_sub_module_deprecation` 函数，指定子包名为 "io.matlab"，模块名为 "mio4"
    # 同时指定私有模块为 "_mio4"，将所有公开符号传递给 `__all__`，并传递访问的属性名 `name`
    return _sub_module_deprecation(sub_package="io.matlab", module="mio4",
                                   private_modules=["_mio4"], all=__all__,
                                   attribute=name)
```