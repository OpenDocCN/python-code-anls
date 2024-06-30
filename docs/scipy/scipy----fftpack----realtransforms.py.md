# `D:\src\scipysrc\scipy\scipy\fftpack\realtransforms.py`

```
# 这个文件不是为公共使用而设计的，将在 SciPy v2.0.0 中移除。
# 使用 `scipy.fftpack` 命名空间来导入以下列出的函数。

# 导入用于处理废弃警告的模块
from scipy._lib.deprecation import _sub_module_deprecation

# 定义 __all__ 列表，这个列表包含了模块中公开的函数名称
__all__ = [
    'dct', 'idct', 'dst', 'idst', 'dctn', 'idctn', 'dstn', 'idstn'
]


# 定义 __dir__() 函数，返回模块中公开的函数名称列表
def __dir__():
    return __all__


# 定义 __getattr__(name) 函数，处理动态属性访问时的操作
def __getattr__(name):
    # 调用 _sub_module_deprecation 函数，处理模块废弃警告，并指定相关参数
    return _sub_module_deprecation(sub_package="fftpack", module="realtransforms",
                                   private_modules=["_realtransforms"], all=__all__,
                                   attribute=name)
```