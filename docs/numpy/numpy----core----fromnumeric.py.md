# `.\numpy\numpy\core\fromnumeric.py`

```py
# 定义一个特殊的方法，用于动态获取属性值
def __getattr__(attr_name):
    # 从 numpy._core 模块中导入 fromnumeric 对象
    from numpy._core import fromnumeric
    # 从当前模块的 _utils 导入 _raise_warning 函数
    from ._utils import _raise_warning
    # 尝试获取 fromnumeric 对象的属性 attr_name
    ret = getattr(fromnumeric, attr_name, None)
    # 如果未找到对应属性，则抛出 AttributeError 异常
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core.fromnumeric' has no attribute {attr_name}")
    # 调用 _raise_warning 函数，发出警告信息
    _raise_warning(attr_name, "fromnumeric")
    # 返回获取到的属性值
    return ret
```