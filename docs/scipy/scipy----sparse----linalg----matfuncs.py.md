# `D:\src\scipysrc\scipy\scipy\sparse\linalg\matfuncs.py`

```
# 导入模块 `_sub_module_deprecation` 从 `scipy._lib.deprecation` 中，用于处理子模块的弃用警告
# 此文件不适合公开使用，并将在 SciPy v2.0.0 中移除。
# 使用 `scipy.sparse.linalg` 命名空间来导入以下列出的函数。

from scipy._lib.deprecation import _sub_module_deprecation

# 定义了模块的公开接口列表，包括 "expm", "inv", "spsolve", "LinearOperator"，
# 并通过 `noqa: F822` 禁止 flake8 错误 F822（"undefined name"）。
__all__ = ["expm", "inv", "spsolve", "LinearOperator"]  # noqa: F822

# 定义 __dir__() 函数，返回模块的公开接口列表
def __dir__():
    return __all__

# 定义 __getattr__(name) 函数，当模块属性未找到时调用 _sub_module_deprecation 函数，
# 发出有关 sparse.linalg 子模块（module="matfuncs"）和私有模块（private_modules=["_matfuncs"]）的弃用警告，
# 并传递公开接口列表（__all__）和请求的属性名（attribute=name）。
def __getattr__(name):
    return _sub_module_deprecation(sub_package="sparse.linalg", module="matfuncs",
                                   private_modules=["_matfuncs"], all=__all__,
                                   attribute=name)
```