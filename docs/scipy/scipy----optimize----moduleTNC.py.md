# `D:\src\scipysrc\scipy\scipy\optimize\moduleTNC.py`

```
# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.optimize` namespace for importing the functions
# included below.
# 导入私有模块_deprecation中的_sub_module_deprecation函数

from scipy._lib.deprecation import _sub_module_deprecation

# 定义一个空的公共变量__all__，用于指定当前模块中可以导出的公共对象

__all__ = []

# 定义一个特殊方法__dir__()，当调用dir()函数时返回__all__变量的内容

def __dir__():
    return __all__

# 定义一个特殊方法__getattr__(name)，在试图访问当前模块中不存在的属性时调用
# 返回调用_sub_module_deprecation函数的结果，用于处理对optimize模块中的moduleTNC子模块的访问
# 使用了私有模块_deprecation中的_sub_module_deprecation函数，传递了sub_package、module、private_modules、all和attribute参数
```