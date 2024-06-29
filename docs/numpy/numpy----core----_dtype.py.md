# `.\numpy\numpy\core\_dtype.py`

```
# 定义一个特殊方法 __getattr__，用于在对象没有指定属性时进行处理
def __getattr__(attr_name):
    # 从 numpy._core 模块中导入 _dtype 对象
    from numpy._core import _dtype
    # 从 ._utils 模块中导入 _raise_warning 函数
    from ._utils import _raise_warning
    # 尝试获取 _dtype 对象的属性 attr_name
    ret = getattr(_dtype, attr_name, None)
    # 如果未找到指定属性，则引发 AttributeError 异常
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core._dtype' has no attribute {attr_name}")
    # 调用 _raise_warning 函数，向用户发出警告，说明属性被访问
    _raise_warning(attr_name, "_dtype")
    # 返回获取到的属性值
    return ret
```