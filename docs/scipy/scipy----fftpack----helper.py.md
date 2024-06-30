# `D:\src\scipysrc\scipy\scipy\fftpack\helper.py`

```
# 导入 _sub_module_deprecation 函数从 scipy._lib.deprecation 模块中
from scipy._lib.deprecation import _sub_module_deprecation

# 定义 __all__ 列表，包含需要在模块中公开的函数名
__all__ = [
    'fftshift', 'ifftshift', 'fftfreq', 'rfftfreq', 'next_fast_len'
]

# 定义 __dir__ 函数，返回模块中公开的所有函数名列表
def __dir__():
    return __all__

# 定义 __getattr__ 函数，用于获取指定名称的属性或函数
def __getattr__(name):
    # 调用 _sub_module_deprecation 函数，发出子模块即将弃用的警告，
    # 使用参数说明子模块为 "fftpack"，模块为 "helper"，私有模块为 "_helper"，
    # 所有公开函数为 __all__ 列表中的函数，需要获取的属性为参数 name 指定的名称。
    return _sub_module_deprecation(sub_package="fftpack", module="helper",
                                   private_modules=["_helper"], all=__all__,
                                   attribute=name)
```