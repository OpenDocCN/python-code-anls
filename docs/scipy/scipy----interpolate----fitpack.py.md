# `D:\src\scipysrc\scipy\scipy\interpolate\fitpack.py`

```
# 此文件不适用于公共使用，并将在 SciPy v2.0.0 中删除。
# 请使用 `scipy.interpolate` 命名空间来导入以下列出的函数。

# 导入 `_sub_module_deprecation` 函数，用于处理子模块的弃用警告
from scipy._lib.deprecation import _sub_module_deprecation

# 定义一个列表，包含将在当前模块中公开的函数和类名
__all__ = [  # noqa: F822
    'BSpline',    # B样条曲线
    'bisplev',    # 二维样条插值的求值
    'bisplrep',   # 二维样条插值的平滑表示
    'insert',     # 在二维样条表示中插入数据点
    'spalde',     # B样条曲线的导数
    'splantider', # 在指定点处计算样条的任意导数
    'splder',     # 计算样条的导数表示
    'splev',      # 计算样条的值
    'splint',     # 样条曲线的积分
    'splprep',    # 样条平滑曲线的参数化
    'splrep',     # 样条曲线的平滑表示
    'sproot',     # 计算样条的根
]

# 定义 `__dir__()` 函数，返回当前模块中公开的所有函数和类名
def __dir__():
    return __all__

# 定义 `__getattr__(name)` 函数，处理对当前模块中未定义的属性的访问
def __getattr__(name):
    # 调用 `_sub_module_deprecation` 函数，发出子模块已弃用的警告
    return _sub_module_deprecation(sub_package="interpolate", module="fitpack",
                                   private_modules=["_fitpack_py"], all=__all__,
                                   attribute=name)
```