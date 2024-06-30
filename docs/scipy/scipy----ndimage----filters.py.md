# `D:\src\scipysrc\scipy\scipy\ndimage\filters.py`

```
# 导入特定的函数和类别，从 scipy._lib.deprecation 模块中导入 _sub_module_deprecation 函数
from scipy._lib.deprecation import _sub_module_deprecation

# 定义一个列表，包含所有公开的函数和类别名称，用于模块导入
__all__ = [  # noqa: F822
    'correlate1d', 'convolve1d', 'gaussian_filter1d',
    'gaussian_filter', 'prewitt', 'sobel', 'generic_laplace',
    'laplace', 'gaussian_laplace', 'generic_gradient_magnitude',
    'gaussian_gradient_magnitude', 'correlate', 'convolve',
    'uniform_filter1d', 'uniform_filter', 'minimum_filter1d',
    'maximum_filter1d', 'minimum_filter', 'maximum_filter',
    'rank_filter', 'median_filter', 'percentile_filter',
    'generic_filter1d', 'generic_filter'
]

# 定义一个特殊的函数 __dir__()，返回模块的所有公开名称
def __dir__():
    return __all__

# 定义一个特殊的函数 __getattr__(name)，处理对不存在的属性的访问
def __getattr__(name):
    # 调用 _sub_module_deprecation 函数，标记子模块的使用过时，并提供相关的提示信息
    return _sub_module_deprecation(sub_package='ndimage', module='filters',
                                   private_modules=['_filters'], all=__all__,
                                   attribute=name)
```