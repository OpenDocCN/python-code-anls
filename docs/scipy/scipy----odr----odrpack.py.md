# `D:\src\scipysrc\scipy\scipy\odr\odrpack.py`

```
# 本文件不适合公共使用，并将在 SciPy v2.0.0 中移除。
# 使用 `scipy.odr` 命名空间来导入以下列出的函数。

# 从 scipy._lib.deprecation 模块中导入 _sub_module_deprecation 函数
from scipy._lib.deprecation import _sub_module_deprecation

# 定义一个包含所有公开对象名称的列表
__all__ = [  # noqa: F822
    'odr', 'OdrWarning', 'OdrError', 'OdrStop',
    'Data', 'RealData', 'Model', 'Output', 'ODR',
    'odr_error', 'odr_stop'
]

# 定义 __dir__() 函数，返回所有公开对象名称的列表
def __dir__():
    return __all__

# 定义 __getattr__(name) 函数，用于获取对象的属性，实现向后兼容和废弃警告
def __getattr__(name):
    return _sub_module_deprecation(sub_package="odr", module="odrpack",
                                   private_modules=["_odrpack"], all=__all__,
                                   attribute=name)
```