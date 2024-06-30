# `D:\src\scipysrc\scipy\scipy\signal\spline.py`

```
# 这个文件不适合公共使用，将在未来的 SciPy 版本中移除。请使用 `scipy.signal` 命名空间来导入下面列出的函数。

# 导入警告模块，用于发出关于过时功能的警告
import warnings

# 从当前目录导入 `_spline` 模块
from . import _spline

# 定义导出的函数列表，只包括 `sepfir2d` 函数
__all__ = ['sepfir2d']  # noqa: F822


# 自定义特殊方法 `__dir__()`，返回导出函数列表 `__all__`
def __dir__():
    return __all__


# 自定义特殊方法 `__getattr__(name)`，用于动态获取属性
def __getattr__(name):
    # 如果请求的属性不在导出函数列表 `__all__` 中，则抛出属性错误
    if name not in __all__:
        raise AttributeError(
            f"scipy.signal.spline 已过时，不具备属性 {name}。请尝试在 scipy.signal 中查找。")

    # 发出过时警告，建议从 `scipy.signal` 命名空间中使用相应属性
    warnings.warn(f"请使用 `scipy.signal` 命名空间中的 `{name}`，`scipy.signal.spline` 命名空间已过时。",
                  category=DeprecationWarning, stacklevel=2)
    
    # 返回 `_spline` 模块中对应名称的属性
    return getattr(_spline, name)
```