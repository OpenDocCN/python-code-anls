# `D:\src\scipysrc\scipy\scipy\sparse\data.py`

```
# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.sparse` namespace for importing the functions
# included below.

# 从 scipy._lib.deprecation 模块导入 _sub_module_deprecation 函数
from scipy._lib.deprecation import _sub_module_deprecation

# 定义一个列表，包含了当前模块公开的所有函数和变量名
__all__ = [  # noqa: F822
    'isscalarlike',  # 可以被当做标量的对象
    'name',           # 名称
    'npfunc',         # NumPy 函数
    'validateaxis',   # 验证轴
]


# 定义 __dir__() 函数，返回当前模块公开的所有函数和变量名
def __dir__():
    return __all__


# 定义 __getattr__(name) 函数，用于处理对当前模块中未定义的属性的访问
def __getattr__(name):
    # 调用 _sub_module_deprecation 函数，返回对应属性的 deprecated 提示
    return _sub_module_deprecation(sub_package="sparse", module="data",
                                   private_modules=["_data"], all=__all__,
                                   attribute=name)
```