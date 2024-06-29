# `.\numpy\numpy\core\_internal.py`

```py
# 从 numpy._core._internal 模块导入 _reconstruct 函数
# 该函数用于从 pickle 数据中重建 ndarray 数组对象
# 注意：numpy.core._internal._reconstruct 函数名称在 NumPy 1.0 之前的版本中用于 ndarray 的 pickle 数据，
# 因此不能在此处删除名称，否则会破坏向后兼容性。
def _reconstruct(subtype, shape, dtype):
    # 导入 ndarray 类型并使用其 __new__ 方法创建新的数组对象
    from numpy import ndarray
    return ndarray.__new__(subtype, shape, dtype)


# Pybind11（版本 <= 2.11.1）从 _internal 子模块导入 _dtype_from_pep3118 函数，
# 因此必须能够无警告地导入它。
_dtype_from_pep3118 = _internal._dtype_from_pep3118

# 定义一个 __getattr__ 函数，用于在运行时动态获取属性
def __getattr__(attr_name):
    # 导入 numpy._core._internal 模块
    from numpy._core import _internal
    # 导入自定义工具函数 _raise_warning
    from ._utils import _raise_warning
    # 尝试从 _internal 模块获取指定名称的属性
    ret = getattr(_internal, attr_name, None)
    # 如果未找到指定名称的属性，则抛出 AttributeError 异常
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core._internal' has no attribute {attr_name}")
    # 发出警告，说明此属性来自 _internal 模块
    _raise_warning(attr_name, "_internal")
    return ret
```