# `.\numpy\numpy\core\records.py`

```py
# 定义一个特殊的方法，用于动态获取对象的属性
def __getattr__(attr_name):
    # 从 numpy._core.records 模块中导入 records 对象
    from numpy._core import records
    # 从当前模块的 ._utils 模块中导入 _raise_warning 函数
    from ._utils import _raise_warning
    # 尝试获取 records 对象中名为 attr_name 的属性
    ret = getattr(records, attr_name, None)
    # 如果未找到指定属性，则抛出 AttributeError 异常
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core.records' has no attribute {attr_name}")
    # 调用 _raise_warning 函数，发出警告，说明属性来自 records 模块
    _raise_warning(attr_name, "records")
    # 返回获取到的属性对象
    return ret
```