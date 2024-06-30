# `D:\src\scipysrc\scipy\scipy\interpolate\interpolate.py`

```
# 导入需要的函数和类，这些函数和类将被废弃，不建议公开使用，预计在 SciPy v2.0.0 中删除。
# 建议使用 `scipy.interpolate` 命名空间来导入下面列出的函数。

from scipy._lib.deprecation import _sub_module_deprecation

# 指定导出的符号列表，用于 `__all__` 变量
__all__ = [  # noqa: F822
    'BPoly',
    'BSpline',
    'NdPPoly',
    'PPoly',
    'RectBivariateSpline',
    'RegularGridInterpolator',
    'interp1d',
    'interp2d',
    'interpn',
    'lagrange',
    'make_interp_spline',
]

# 定义 `__dir__` 函数，返回模块的导出符号列表
def __dir__():
    return __all__

# 定义 `__getattr__` 函数，当访问不存在的属性时调用，用于处理废弃警告和导入替代功能
def __getattr__(name):
    return _sub_module_deprecation(
        sub_package="interpolate",    # 废弃警告中的子包名称
        module="interpolate",         # 废弃警告中的模块名称
        private_modules=["_interpolate", "fitpack2", "_rgi"],  # 私有模块列表，可能是替代功能的实现
        all=__all__,                 # 可用的所有导出符号的列表
        attribute=name               # 被访问的属性名称
    )
```