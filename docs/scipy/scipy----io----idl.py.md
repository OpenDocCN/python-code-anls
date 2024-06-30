# `D:\src\scipysrc\scipy\scipy\io\idl.py`

```
# 导入模块 `_sub_module_deprecation` 中的 `_sub_module_deprecation` 函数
# 用于警告和提醒用户，该文件不适合公开使用，并将在 SciPy v2.0.0 中移除。
# 使用 `scipy.io` 命名空间导入下面列出的函数。

from scipy._lib.deprecation import _sub_module_deprecation

# 定义公开的函数和对象列表，只包含 "readsav" 一个函数，忽略 F822 类型的警告
__all__ = ["readsav"]  # noqa: F822


def __dir__():
    # 返回当前模块的公开函数和对象列表
    return __all__


def __getattr__(name):
    # 当请求的属性名不存在时，调用 `_sub_module_deprecation` 函数
    # 进行子模块警告，指定子包为 "io"，模块为 "idl"，私有模块为 "_idl"，
    # 全部公开模块为 __all__ 中指定的列表，请求的属性为 name。
    return _sub_module_deprecation(sub_package="io", module="idl",
                                   private_modules=["_idl"], all=__all__,
                                   attribute=name)
```