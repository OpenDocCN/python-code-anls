# `D:\src\scipysrc\scipy\scipy\signal\waveforms.py`

```
# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.signal` namespace for importing the functions
# included below.

# 从 scipy._lib.deprecation 模块导入 _sub_module_deprecation 函数
from scipy._lib.deprecation import _sub_module_deprecation

# 定义 __all__ 列表，包含了将在本模块中公开的函数名称
__all__ = [  # noqa: F822
    'sawtooth', 'square', 'gausspulse', 'chirp', 'sweep_poly',
    'unit_impulse',
]


# 定义 __dir__() 函数，返回模块中公开的所有函数名列表
def __dir__():
    return __all__


# 定义 __getattr__(name) 函数，用于在引用未定义的属性时发出警告
def __getattr__(name):
    return _sub_module_deprecation(sub_package="signal", module="waveforms",
                                   private_modules=["_waveforms"], all=__all__,
                                   attribute=name)
```