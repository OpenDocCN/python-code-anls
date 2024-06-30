# `D:\src\scipysrc\scipy\scipy\signal\spectral.py`

```
# 导入需要的警告模块，标记此文件不应对外公开，并将在 SciPy v2.0.0 中移除。
# 在此处使用 `scipy.signal` 命名空间导入下面列出的函数。

from scipy._lib.deprecation import _sub_module_deprecation
# 导入 `_sub_module_deprecation` 函数用于处理子模块过时警告

__all__ = [  # noqa: F822
    'periodogram', 'welch', 'lombscargle', 'csd', 'coherence',
    'spectrogram', 'stft', 'istft', 'check_COLA', 'check_NOLA',
    'get_window',
]
# 定义 `__all__` 列表，包含要导出到模块外的函数名

def __dir__():
    return __all__
# 定义 `__dir__()` 函数，返回模块的 `__all__` 列表，用于 `dir()` 函数调用时返回的属性列表

def __getattr__(name):
    return _sub_module_deprecation(sub_package="signal", module="spectral",
                                   private_modules=["_spectral_py"], all=__all__,
                                   attribute=name)
# 定义 `__getattr__()` 函数，用于处理对模块中未定义属性的访问。
# 返回 `_sub_module_deprecation()` 处理后的结果，提供过时警告和推荐替代信息。
```