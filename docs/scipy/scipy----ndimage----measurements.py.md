# `D:\src\scipysrc\scipy\scipy\ndimage\measurements.py`

```
# 导入来自 `scipy._lib.deprecation` 模块的 `_sub_module_deprecation` 函数，
# 用于处理子模块废弃警告。

from scipy._lib.deprecation import _sub_module_deprecation

# 定义 `__all__` 变量，包含了需要在当前命名空间中公开的函数名列表，
# 这些函数名将在导入时被 `from module import *` 形式使用。

__all__ = [  # noqa: F822
    'label', 'find_objects', 'labeled_comprehension',
    'sum', 'mean', 'variance', 'standard_deviation',
    'minimum', 'maximum', 'median', 'minimum_position',
    'maximum_position', 'extrema', 'center_of_mass',
    'histogram', 'watershed_ift', 'sum_labels'
]

# 定义 `__dir__()` 函数，返回当前模块中公开的函数和变量的列表，
# 这些列表由 `__all__` 变量定义。

def __dir__():
    return __all__

# 定义 `__getattr__(name)` 函数，当试图访问当前模块中未定义的属性时被调用，
# 它将委托给 `_sub_module_deprecation` 函数处理废弃警告，
# 传递了子模块名 'ndimage'，模块名 'measurements'，私有模块列表 '_measurements'，
# 全部公开函数列表 `__all__`，以及属性名 `name`。

def __getattr__(name):
    return _sub_module_deprecation(sub_package='ndimage', module='measurements',
                                   private_modules=['_measurements'], all=__all__,
                                   attribute=name)
```