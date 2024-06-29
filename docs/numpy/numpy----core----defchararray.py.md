# `.\numpy\numpy\core\defchararray.py`

```py
# 定义一个特殊方法 __getattr__，用于在获取不存在的属性时进行处理
def __getattr__(attr_name):
    # 从 numpy._core 模块中导入 defchararray 对象
    from numpy._core import defchararray
    # 从 ._utils 模块中导入 _raise_warning 函数
    from ._utils import _raise_warning
    # 尝试获取 defchararray 对象中的属性 attr_name
    ret = getattr(defchararray, attr_name, None)
    # 如果获取不到该属性，则抛出 AttributeError 异常
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core.defchararray' has no attribute {attr_name}")
    # 调用 _raise_warning 函数，向用户发出警告信息
    _raise_warning(attr_name, "defchararray")
    # 返回获取到的属性或方法对象
    return ret
```