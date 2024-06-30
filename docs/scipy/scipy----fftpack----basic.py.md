# `D:\src\scipysrc\scipy\scipy\fftpack\basic.py`

```
# 此文件不适用于公共使用，并将在 SciPy v2.0.0 中移除。
# 请使用 `scipy.fftpack` 命名空间来导入下面列出的函数。

# 从 scipy._lib.deprecation 模块导入 _sub_module_deprecation 函数
from scipy._lib.deprecation import _sub_module_deprecation

# 定义公开的函数和变量列表
__all__ = [  # noqa: F822
    'fft', 'ifft', 'fftn', 'ifftn', 'rfft', 'irfft',
    'fft2', 'ifft2'
]


# 定义特殊方法 __dir__()，返回当前模块的公开函数和变量列表
def __dir__():
    return __all__


# 定义特殊方法 __getattr__(name)，当访问不存在的属性时调用该方法
def __getattr__(name):
    # 使用 _sub_module_deprecation 函数生成关于废弃警告的消息
    return _sub_module_deprecation(sub_package="fftpack", module="basic",
                                   private_modules=["_basic"], all=__all__,
                                   attribute=name)
```