# `.\numpy\numpy\core\arrayprint.py`

```
# 定义一个特殊方法 __getattr__，用于在对象中获取指定属性的值
def __getattr__(attr_name):
    # 从 numpy._core.arrayprint 模块中导入 arrayprint 函数
    from numpy._core import arrayprint
    # 从 ._utils 模块中导入 _raise_warning 函数
    from ._utils import _raise_warning
    # 尝试从 arrayprint 模块中获取指定名称的属性值
    ret = getattr(arrayprint, attr_name, None)
    # 如果获取的属性值为 None，则抛出 AttributeError 异常
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core.arrayprint' has no attribute {attr_name}")
    # 调用 _raise_warning 函数，提醒获取到的属性名称和其所在模块
    _raise_warning(attr_name, "arrayprint")
    # 返回获取到的属性值
    return ret
```