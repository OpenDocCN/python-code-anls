# `D:\src\scipysrc\scipy\scipy\signal\bsplines.py`

```
# 该文件不适合公共使用，并且将在 SciPy v2.0.0 版本中移除。
# 使用 `scipy.signal` 命名空间来导入下面列出的函数。

# 从 scipy._lib.deprecation 模块导入 _sub_module_deprecation 函数
from scipy._lib.deprecation import _sub_module_deprecation

# 定义导出的模块成员列表
__all__ = [
    'spline_filter', 'gauss_spline',
    'cspline1d', 'qspline1d', 'cspline1d_eval', 'qspline1d_eval',
    'cspline2d', 'sepfir2d'
]


# 定义 __dir__() 函数，返回模块的所有导出成员
def __dir__():
    return __all__


# 定义 __getattr__(name) 函数，用于获取指定名称的属性
def __getattr__(name):
    return _sub_module_deprecation(sub_package="signal", module="bsplines",
                                   private_modules=["_bsplines"], all=__all__,
                                   attribute=name)
```