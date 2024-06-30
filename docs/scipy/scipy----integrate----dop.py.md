# `D:\src\scipysrc\scipy\scipy\integrate\dop.py`

```
# 引入 _sub_module_deprecation 函数来处理子模块弃用警告
from scipy._lib.deprecation import _sub_module_deprecation

# 定义一个空的字符串列表，用于存放模块的公共接口名称
__all__: list[str] = []

# 定义一个特殊方法 __dir__()，返回模块的公共接口列表 __all__
def __dir__():
    return __all__

# 定义一个特殊方法 __getattr__(name)，用于获取指定名称的属性
def __getattr__(name):
    # 调用 _sub_module_deprecation 函数，发出子模块弃用警告
    return _sub_module_deprecation(sub_package="integrate", module="dop",
                                   private_modules=["_dop"], all=__all__,
                                   attribute=name)
```