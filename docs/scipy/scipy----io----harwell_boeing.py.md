# `D:\src\scipysrc\scipy\scipy\io\harwell_boeing.py`

```
# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.io` namespace for importing the functions
# included below.
# 导入 `_sub_module_deprecation` 函数，用于处理子模块废弃警告
from scipy._lib.deprecation import _sub_module_deprecation

# 模块的公开接口列表，用于控制导出的公开函数
__all__ = ["hb_read", "hb_write"]  # noqa: F822

# 定义 `__dir__()` 特殊方法，返回模块的公开接口列表
def __dir__():
    return __all__

# 定义 `__getattr__(name)` 特殊方法，处理动态获取属性的操作
def __getattr__(name):
    # 调用 `_sub_module_deprecation` 函数，发出子模块废弃警告
    return _sub_module_deprecation(sub_package="io", module="harwell_boeing",
                                   private_modules=["_harwell_boeing"], all=__all__,
                                   attribute=name)
```