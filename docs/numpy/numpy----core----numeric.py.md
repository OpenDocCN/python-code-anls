# `.\numpy\numpy\core\numeric.py`

```py
# 定义一个特殊方法 __getattr__，用于在当前作用域中动态获取属性
def __getattr__(attr_name):
    # 从 numpy._core 模块中导入 numeric 对象
    from numpy._core import numeric
    # 从当前模块中导入 _raise_warning 函数
    from ._utils import _raise_warning

    # 定义一个特殊的对象 sentinel，用于标记属性是否存在
    sentinel = object()
    # 尝试从 numeric 对象中获取指定名称的属性，如果属性不存在，则使用 sentinel 标记
    ret = getattr(numeric, attr_name, sentinel)
    # 如果 ret 是 sentinel，则表示属性不存在，抛出 AttributeError 异常
    if ret is sentinel:
        raise AttributeError(
            f"module 'numpy.core.numeric' has no attribute {attr_name}")
    # 调用 _raise_warning 函数，向用户发出警告，说明属性从 numeric 模块中获取
    _raise_warning(attr_name, "numeric")
    # 返回获取到的属性对象
    return ret
```