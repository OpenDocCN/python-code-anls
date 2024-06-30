# `D:\src\scipysrc\scipy\scipy\sparse\sputils.py`

```
# 导入 _sub_module_deprecation 函数，用于向用户发出子模块弃用警告
# 该文件不适用于公共使用，并将在 SciPy v2.0.0 中删除
# 使用 `scipy.sparse` 命名空间来导入以下列出的函数

# 用于存储公开导出的名称列表
from scipy._lib.deprecation import _sub_module_deprecation

# 定义一个空列表，用于指示模块中公开的名称
__all__: list[str] = []

# 定义 __dir__() 函数，返回模块中公开的所有名称列表
def __dir__():
    return __all__

# 定义 __getattr__(name) 函数，用于当属性未找到时进行处理
def __getattr__(name):
    # 调用 _sub_module_deprecation 函数，发出关于子模块 "sparse" 中模块 "sputils" 的弃用警告
    # private_modules 参数指定需要弃用警告的私有模块列表
    # all 参数传入 __all__，表示该函数的所有导出名称
    return _sub_module_deprecation(sub_package="sparse", module="sputils",
                                   private_modules=["_sputils"], all=__all__,
                                   attribute=name)
```