# `.\numpy\numpy\core\_dtype_ctypes.py`

```
# 定义一个特殊的函数 __getattr__，用于在对象中获取指定属性的方法
def __getattr__(attr_name):
    # 从 numpy._core 模块中导入 _dtype_ctypes 对象
    from numpy._core import _dtype_ctypes
    # 从 ._utils 模块中导入 _raise_warning 函数
    from ._utils import _raise_warning
    # 尝试获取 _dtype_ctypes 对象中名为 attr_name 的属性
    ret = getattr(_dtype_ctypes, attr_name, None)
    # 如果未找到指定属性，则抛出 AttributeError 异常
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core._dtype_ctypes' has no attribute {attr_name}")
    # 调用 _raise_warning 函数，提醒用户该属性来自 _dtype_ctypes 对象
    _raise_warning(attr_name, "_dtype_ctypes")
    # 返回获取到的属性
    return ret
```