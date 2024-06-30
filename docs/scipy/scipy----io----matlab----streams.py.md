# `D:\src\scipysrc\scipy\scipy\io\matlab\streams.py`

```
# 引入 _sub_module_deprecation 函数从 scipy._lib.deprecation 模块
# 该文件不适用于公共使用，并且在 SciPy v2.0.0 中将被移除
from scipy._lib.deprecation import _sub_module_deprecation

# 定义空的列表 __all__，用于模块的导出
__all__: list[str] = []

# 定义一个特殊方法 __dir__()，返回当前模块可用的全部属性列表
def __dir__():
    return __all__

# 定义一个特殊方法 __getattr__(name)，当访问当前模块中不存在的属性时调用
def __getattr__(name):
    # 调用 _sub_module_deprecation 函数，进行模块废弃警告处理
    return _sub_module_deprecation(sub_package="io.matlab", module="streams",
                                   private_modules=["_streams"], all=__all__,
                                   attribute=name)
```