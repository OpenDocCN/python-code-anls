# `.\numpy\numpy\core\getlimits.py`

```py
# 定义一个特殊的属性访问方法，用于动态获取属性值
def __getattr__(attr_name):
    # 从 numpy._core 模块中导入 getlimits 函数
    from numpy._core import getlimits
    # 从当前模块的 _utils 导入 _raise_warning 函数
    from ._utils import _raise_warning
    # 尝试获取 getlimits 模块中名为 attr_name 的属性值
    ret = getattr(getlimits, attr_name, None)
    # 如果获取的属性值为 None，则抛出 AttributeError 异常
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core.getlimits' has no attribute {attr_name}")
    # 调用 _raise_warning 函数，发出警告
    _raise_warning(attr_name, "getlimits")
    # 返回获取到的属性值
    return ret
```