# `.\numpy\numpy\core\multiarray.py`

```py
# 从 numpy._core 模块中导入 multiarray 对象
from numpy._core import multiarray

# 将 "multiarray" 模块中的 "_reconstruct" 和 "scalar" 属性复制给全局命名空间，
# 以支持旧的 pickle 文件
for item in ["_reconstruct", "scalar"]:
    globals()[item] = getattr(multiarray, item)

# Pybind11（在版本 <= 2.11.1 中）从 multiarray 子模块导入 _ARRAY_API 作为 NumPy 初始化的一部分，
# 因此它必须可以无警告导入。
_ARRAY_API = multiarray._ARRAY_API

# 定义 __getattr__ 函数，用于动态获取属性
def __getattr__(attr_name):
    # 重新导入 multiarray 对象，用于获取特定属性
    from numpy._core import multiarray
    # 导入 _raise_warning 函数，用于抛出警告
    from ._utils import _raise_warning
    # 尝试从 multiarray 中获取指定属性
    ret = getattr(multiarray, attr_name, None)
    # 如果获取失败，则抛出 AttributeError 异常
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core.multiarray' has no attribute {attr_name}")
    # 发出警告，说明属性来自 multiarray
    _raise_warning(attr_name, "multiarray")
    return ret

# 删除已导入的 multiarray 对象的引用，清理命名空间
del multiarray
```