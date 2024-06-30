# `D:\src\scipysrc\scipy\scipy\stats\kde.py`

```
# 这个文件不打算供公共使用，并将在 SciPy v2.0.0 中移除。
# 使用 `scipy.stats` 命名空间来导入下面包含的函数。

# 从 `scipy._lib.deprecation` 模块中导入 `_sub_module_deprecation` 函数
from scipy._lib.deprecation import _sub_module_deprecation

# 定义一个公开的名为 `gaussian_kde` 的列表，用于 `__all__` 变量，并禁用 F822 错误提示
__all__ = ["gaussian_kde"]  # noqa: F822

# 定义 `__dir__()` 函数，返回 `__all__` 变量，用于在导入时展示可用的子模块列表
def __dir__():
    return __all__

# 定义 `__getattr__(name)` 函数，用于在动态访问时处理子模块的弃用
def __getattr__(name):
    return _sub_module_deprecation(sub_package="stats", module="kde",
                                   private_modules=["_kde"], all=__all__,
                                   attribute=name)
```