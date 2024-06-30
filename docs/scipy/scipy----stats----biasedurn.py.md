# `D:\src\scipysrc\scipy\scipy\stats\biasedurn.py`

```
# 导入 _sub_module_deprecation 函数，用于处理子模块的废弃警告
from scipy._lib.deprecation import _sub_module_deprecation

# 定义一个空列表，用于存放模块中公开的对象
__all__: list[str] = []

# 定义 __dir__() 函数，当使用 dir() 函数或对象.__dir__() 调用时返回 __all__ 列表
def __dir__():
    return __all__

# 定义 __getattr__(name) 函数，用于获取指定属性的值
def __getattr__(name):
    # 调用 _sub_module_deprecation 函数，发出有关子模块 "stats" 和模块 "biasedurn" 的废弃警告，
    # 并指定私有模块 "_biasedurn"，所有公开的属性存放在 __all__ 列表中，返回指定属性的值
    return _sub_module_deprecation(sub_package="stats", module="biasedurn",
                                   private_modules=["_biasedurn"], all=__all__,
                                   attribute=name)
```