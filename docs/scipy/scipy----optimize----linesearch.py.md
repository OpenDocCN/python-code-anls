# `D:\src\scipysrc\scipy\scipy\optimize\linesearch.py`

```
# 导入 `_sub_module_deprecation` 函数，用于处理子模块废弃警告
from scipy._lib.deprecation import _sub_module_deprecation

# 声明仅导出 `line_search` 函数给外部使用，忽略 F822 错误
__all__ = ["line_search"]  # noqa: F822

# 定义 `__dir__()` 特殊方法，返回当前模块可导出的全部内容
def __dir__():
    return __all__

# 定义 `__getattr__(name)` 特殊方法，处理动态获取属性的请求
def __getattr__(name):
    # 调用 `_sub_module_deprecation` 函数，发出子模块废弃警告
    return _sub_module_deprecation(sub_package="optimize", module="linesearch",
                                   private_modules=["_linesearch"], all=__all__,
                                   attribute=name)
```