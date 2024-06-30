# `D:\src\scipysrc\scipy\scipy\interpolate\polyint.py`

```
# 此文件不适用于公共使用，并将在 SciPy v2.0.0 中删除。
# 使用 `scipy.interpolate` 命名空间来导入以下包含的函数。

# 从 scipy._lib.deprecation 模块中导入 _sub_module_deprecation 函数
from scipy._lib.deprecation import _sub_module_deprecation

# 定义 __all__ 列表，包含需要在本模块中公开的符号名称
__all__ = [
    'BarycentricInterpolator',
    'KroghInterpolator',
    'approximate_taylor_polynomial',
    'barycentric_interpolate',
    'krogh_interpolate',
]


def __dir__():
    # 返回模块的 __all__ 列表，用于自动补全和显示
    return __all__


def __getattr__(name):
    # 当模块试图获取未定义的属性时，调用 _sub_module_deprecation 函数，
    # 向用户发出关于弃用的警告，并建议使用 scipy.interpolate 命名空间的正确方式
    return _sub_module_deprecation(sub_package="interpolate", module="polyint",
                                   private_modules=["_polyint"], all=__all__,
                                   attribute=name)
```