# `D:\src\scipysrc\scipy\scipy\io\matlab\mio.py`

```
# 这个文件不适合公共使用，并将在 SciPy v2.0.0 中移除。
# 使用 `scipy.io.matlab` 命名空间来导入以下列出的函数。

# 从 scipy._lib.deprecation 模块中导入 _sub_module_deprecation 函数
from scipy._lib.deprecation import _sub_module_deprecation

# 定义 __all__ 列表，这是模块的公开接口，其中包含 loadmat、savemat 和 whosmat 函数名称
__all__ = ["loadmat", "savemat", "whosmat"]  # noqa: F822

# 定义 __dir__() 函数，返回模块的公开接口列表 __all__
def __dir__():
    return __all__

# 定义 __getattr__(name) 函数，当访问未定义的属性时调用
def __getattr__(name):
    # 返回 _sub_module_deprecation 函数的调用结果，用于处理子模块废弃警告
    return _sub_module_deprecation(sub_package="io.matlab", module="mio",
                                   private_modules=["_mio"], all=__all__,
                                   attribute=name)
```