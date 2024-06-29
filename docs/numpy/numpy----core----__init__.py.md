# `.\numpy\numpy\core\__init__.py`

```py
"""
The `numpy.core` submodule exists solely for backward compatibility
purposes. The original `core` was renamed to `_core` and made private.
`numpy.core` will be removed in the future.
"""
# 从 `numpy` 包中导入 `_core` 子模块，用于向后兼容性目的
from numpy import _core

# 从当前包的 `_utils` 模块中导入 `_raise_warning` 函数
from ._utils import _raise_warning


# We used to use `np.core._ufunc_reconstruct` to unpickle.
# This is unnecessary, but old pickles saved before 1.20 will be using it,
# and there is no reason to break loading them.
# 定义 `_ufunc_reconstruct` 函数，用于反序列化时重建 ufunc 对象
def _ufunc_reconstruct(module, name):
    # 导入指定模块 `module` 并返回其中指定的 `name` 对象
    # `fromlist` 参数确保当模块名嵌套时，`mod` 指向最内层模块而非父包
    mod = __import__(module, fromlist=[name])
    return getattr(mod, name)


# force lazy-loading of submodules to ensure a warning is printed

# 定义 `__all__` 列表，包含公开的子模块名称，用于 `from package import *` 语法
__all__ = ["arrayprint", "defchararray", "_dtype_ctypes", "_dtype",
           "einsumfunc", "fromnumeric", "function_base", "getlimits",
           "_internal", "multiarray", "_multiarray_umath", "numeric",
           "numerictypes", "overrides", "records", "shape_base", "umath"]

# 定义 `__getattr__` 函数，用于动态获取 `numpy.core` 中的属性
def __getattr__(attr_name):
    # 从 `numpy._core` 中获取指定属性 `attr_name`
    attr = getattr(_core, attr_name)
    # 调用 `_raise_warning` 函数，对获取的属性进行警告处理
    _raise_warning(attr_name)
    # 返回获取的属性
    return attr
```