# `D:\src\scipysrc\scipy\scipy\optimize\minpack2.py`

```
# 导入 `_sub_module_deprecation` 函数从 `scipy._lib.deprecation` 模块
# `__all__` 是一个空列表，用于定义当前模块的公开接口

from scipy._lib.deprecation import _sub_module_deprecation

# 定义空列表 `__all__`，用于标识当前模块的公开接口

__all__: list[str] = []


# 定义特殊方法 `__dir__()`，返回当前模块的公开接口列表 `__all__`

def __dir__():
    return __all__


# 定义特殊方法 `__getattr__(name)`，用于获取 `optimize.minpack2` 模块中指定属性的值
# 若属性被访问，会调用 `_sub_module_deprecation` 函数，标记 `minpack2` 模块作废，建议使用 `optimize` 命名空间
# 参数包括 `sub_package="optimize"`, `module="minpack2"`, `private_modules=["_minpack2"]`, `all=__all__`, `attribute=name`

def __getattr__(name):
    return _sub_module_deprecation(sub_package="optimize", module="minpack2",
                                   private_modules=["_minpack2"], all=__all__,
                                   attribute=name)
```