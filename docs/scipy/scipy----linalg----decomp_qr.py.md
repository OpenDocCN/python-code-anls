# `D:\src\scipysrc\scipy\scipy\linalg\decomp_qr.py`

```
# 导入 `_sub_module_deprecation` 函数，用于处理子模块的废弃警告
from scipy._lib.deprecation import _sub_module_deprecation

# 声明 `__all__` 列表，用于指定在 `from module import *` 语句中导入的符号
__all__ = [  # noqa: F822
    'qr', 'qr_multiply', 'rq', 'get_lapack_funcs'
]

# 自定义函数 `__dir__()`，返回模块的公开符号列表
def __dir__():
    return __all__

# 自定义函数 `__getattr__(name)`，在访问不存在的属性时被调用，
# 用于处理模块属性的废弃警告和转向
def __getattr__(name):
    return _sub_module_deprecation(sub_package="linalg", module="decomp_qr",
                                   private_modules=["_decomp_qr"], all=__all__,
                                   attribute=name)
```