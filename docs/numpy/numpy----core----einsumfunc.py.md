# `.\numpy\numpy\core\einsumfunc.py`

```
# 定义一个特殊的属性获取方法，用于获取指定名称的属性值
def __getattr__(attr_name):
    # 从 numpy._core 模块中导入 einsumfunc 函数或属性
    from numpy._core import einsumfunc
    # 从 ._utils 模块中导入 _raise_warning 函数
    from ._utils import _raise_warning
    # 尝试获取 einsumfunc 模块中指定名称的属性值
    ret = getattr(einsumfunc, attr_name, None)
    # 如果未找到指定属性，抛出 AttributeError 异常
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core.einsumfunc' has no attribute {attr_name}")
    # 调用 _raise_warning 函数，发出警告，提醒 einsumfunc 模块中的属性被访问
    _raise_warning(attr_name, "einsumfunc")
    # 返回获取到的属性值
    return ret
```