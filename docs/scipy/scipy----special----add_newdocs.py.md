# `D:\src\scipysrc\scipy\scipy\special\add_newdocs.py`

```
# This file is not meant for public use and will be removed in SciPy v2.0.0.

# 导入函数 _sub_module_deprecation 从 scipy._lib.deprecation 模块
from scipy._lib.deprecation import _sub_module_deprecation

# 定义一个空列表 __all__，用于存储模块中公开的所有名称
__all__: list[str] = []

# 定义一个特殊方法 __dir__()，返回 __all__，这样在使用 dir() 函数时会返回 __all__ 中的内容
def __dir__():
    return __all__

# 定义一个特殊方法 __getattr__(name)，当访问未定义的属性时会调用该方法
def __getattr__(name):
    # 调用 _sub_module_deprecation 函数，给出关于特定子模块和属性的弃用警告信息
    return _sub_module_deprecation(sub_package="special", module="add_newdocs",
                                   private_modules=["_add_newdocs"], all=__all__,
                                   attribute=name)
```