# `D:\src\scipysrc\scipy\scipy\signal\fir_filter_design.py`

```
# 该文件不适用于公共使用，并将在 SciPy v2.0.0 中移除。
# 使用 `scipy.signal` 命名空间来导入以下包含的函数。

# 从 scipy._lib.deprecation 模块导入 _sub_module_deprecation 函数
from scipy._lib.deprecation import _sub_module_deprecation

# 定义一个全局变量 __all__，包含了一组函数名
__all__ = [
    'kaiser_beta', 'kaiser_atten', 'kaiserord',
    'firwin', 'firwin2', 'remez', 'firls', 'minimum_phase',
]


# 定义 __dir__() 函数，返回模块中的所有函数名列表
def __dir__():
    return __all__


# 定义 __getattr__(name) 函数，用于动态获取模块中的属性
def __getattr__(name):
    # 调用 _sub_module_deprecation 函数，指定子模块为 "signal"，模块为 "fir_filter_design"
    # 传递私有模块列表 ["_fir_filter_design"] 和全局变量 __all__
    return _sub_module_deprecation(sub_package="signal", module="fir_filter_design",
                                   private_modules=["_fir_filter_design"], all=__all__,
                                   attribute=name)
```