# `D:\src\scipysrc\scipy\scipy\signal\windows\windows.py`

```
# 导入 _sub_module_deprecation 函数从 scipy._lib.deprecation 模块
# 这些函数不再推荐使用，并将在 SciPy v2.0.0 中移除，建议使用 scipy.signal.windows 命名空间进行导入

from scipy._lib.deprecation import _sub_module_deprecation

# 定义公开的函数和变量列表，用于指定在导入时会包含的函数名称
__all__ = [  # noqa: F822
    'boxcar', 'triang', 'parzen', 'bohman', 'blackman', 'nuttall',
    'blackmanharris', 'flattop', 'bartlett', 'barthann',
    'hamming', 'kaiser', 'gaussian', 'general_cosine',
    'general_gaussian', 'general_hamming', 'chebwin', 'cosine',
    'hann', 'exponential', 'tukey', 'taylor', 'dpss', 'get_window',
]

# 定义 __dir__ 函数，返回 __all__ 列表，用于支持 dir() 函数的调用
def __dir__():
    return __all__

# 定义 __getattr__ 函数，用于当访问模块中未定义的属性时触发
def __getattr__(name):
    # 调用 _sub_module_deprecation 函数，提示用户已弃用的子模块
    return _sub_module_deprecation(sub_package="signal.windows", module="windows",
                                   private_modules=["_windows"], all=__all__,
                                   attribute=name)
```