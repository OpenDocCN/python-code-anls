# `D:\src\scipysrc\scipy\scipy\stats\mstats_extras.py`

```
# 本文件不是公共使用的，将在 SciPy v2.0.0 中删除。
# 使用 `scipy.stats` 命名空间来导入下面列出的函数。

# 从 `scipy._lib.deprecation` 模块中导入 `_sub_module_deprecation` 函数
from scipy._lib.deprecation import _sub_module_deprecation

# 将下面列出的函数名组成的列表赋值给 `__all__` 变量，同时禁用 F822 警告
__all__ = [
    'compare_medians_ms',
    'hdquantiles', 'hdmedian', 'hdquantiles_sd',
    'idealfourths',
    'median_cihs','mjci','mquantiles_cimj',
    'rsh',
    'trimmed_mean_ci',
]

# 自定义函数 `__dir__()`，返回 `__all__` 变量的值
def __dir__():
    return __all__

# 自定义函数 `__getattr__(name)`，在模块 `mstats_extras` 中使用 `_sub_module_deprecation` 函数进行警告处理
# 对于私有模块 `_mstats_extras`，使用 `_sub_module_deprecation` 函数进行警告处理
# `all=__all__` 传递函数列表到 `_sub_module_deprecation` 函数
# `attribute=name` 传递属性名到 `_sub_module_deprecation` 函数
# `correct_module="mstats"` 提示正确的模块是 `mstats`
def __getattr__(name):
    return _sub_module_deprecation(sub_package="stats", module="mstats_extras",
                                   private_modules=["_mstats_extras"], all=__all__,
                                   attribute=name, correct_module="mstats")
```