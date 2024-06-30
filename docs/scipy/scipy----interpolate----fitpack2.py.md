# `D:\src\scipysrc\scipy\scipy\interpolate\fitpack2.py`

```
# 这个文件不是用于公共使用的，将在 SciPy v2.0.0 中删除。
# 使用 `scipy.interpolate` 命名空间来导入以下所包含的函数。

# 从 scipy._lib.deprecation 模块中导入 _sub_module_deprecation 函数
from scipy._lib.deprecation import _sub_module_deprecation

# 定义一个包含需要导出的符号名列表
__all__ = [  # noqa: F822
    'BivariateSpline',                 # 双变量样条插值
    'InterpolatedUnivariateSpline',    # 插值的单变量样条插值
    'LSQBivariateSpline',              # 最小二乘双变量样条插值
    'LSQSphereBivariateSpline',        # 最小二乘球面双变量样条插值
    'LSQUnivariateSpline',             # 最小二乘单变量样条插值
    'RectBivariateSpline',             # 矩形双变量样条插值
    'RectSphereBivariateSpline',       # 矩形球面双变量样条插值
    'SmoothBivariateSpline',           # 平滑双变量样条插值
    'SmoothSphereBivariateSpline',     # 平滑球面双变量样条插值
    'UnivariateSpline',                # 单变量样条插值
]

# 定义 __dir__() 函数，返回模块中的所有导出符号名列表
def __dir__():
    return __all__

# 定义 __getattr__(name) 函数，用于获取模块中不存在的属性名时的行为
def __getattr__(name):
    return _sub_module_deprecation(sub_package="interpolate", module="fitpack2",
                                   private_modules=["_fitpack2"], all=__all__,
                                   attribute=name)
```