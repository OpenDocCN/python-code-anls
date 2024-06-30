# `D:\src\scipysrc\scipy\scipy\io\mmio.py`

```
# 这个文件不是公共使用的，将在 SciPy v2.0.0 中移除。
# 使用 `scipy.io` 命名空间来导入下面列出的函数。

# 从 scipy._lib.deprecation 模块中导入 _sub_module_deprecation 函数
from scipy._lib.deprecation import _sub_module_deprecation

# 定义 __all__ 列表，这是模块中导出的公共接口列表
__all__ = ["mminfo", "mmread", "mmwrite"]  # noqa: F822

# 定义 __dir__() 函数，返回当前模块的所有公共接口名称
def __dir__():
    return __all__

# 定义 __getattr__(name) 函数，当访问不存在的属性时会调用此函数
def __getattr__(name):
    # 调用 _sub_module_deprecation 函数，用于向用户发出关于子模块已弃用的警告信息
    return _sub_module_deprecation(sub_package="io", module="mmio",
                                   private_modules=["_mmio"], all=__all__,
                                   attribute=name)
```