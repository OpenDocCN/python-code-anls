# `D:\src\scipysrc\scipy\scipy\signal\signaltools.py`

```
# 本文件不适用于公共使用，并且将在 SciPy v2.0.0 中删除。
# 使用 `scipy.signal` 命名空间来导入以下列出的函数。

# 从 scipy._lib.deprecation 模块中导入 _sub_module_deprecation 函数
from scipy._lib.deprecation import _sub_module_deprecation

# 定义 __all__ 列表，包含了所有公开的函数名，用于模块导入时的限定
__all__ = [
    'correlate', 'correlation_lags', 'correlate2d',
    'convolve', 'convolve2d', 'fftconvolve', 'oaconvolve',
    'order_filter', 'medfilt', 'medfilt2d', 'wiener', 'lfilter',
    'lfiltic', 'sosfilt', 'deconvolve', 'hilbert', 'hilbert2',
    'unique_roots', 'invres', 'invresz', 'residue',
    'residuez', 'resample', 'resample_poly', 'detrend',
    'lfilter_zi', 'sosfilt_zi', 'sosfiltfilt', 'choose_conv_method',
    'filtfilt', 'decimate', 'vectorstrength',
    'dlti', 'upfirdn', 'get_window', 'cheby1', 'firwin'
]

# 定义 __dir__() 函数，返回模块中公开的所有函数名
def __dir__():
    return __all__

# 定义 __getattr__(name) 函数，用于动态获取指定名称的属性
# 当模块中不存在指定的属性时，会调用 _sub_module_deprecation 函数进行处理
def __getattr__(name):
    return _sub_module_deprecation(sub_package="signal", module="signaltools",
                                   private_modules=["_signaltools"], all=__all__,
                                   attribute=name)
```