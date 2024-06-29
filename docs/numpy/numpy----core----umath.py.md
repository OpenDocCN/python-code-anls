# `.\numpy\numpy\core\umath.py`

```
# 定义一个特殊方法 __getattr__，用于在对象中动态获取属性
def __getattr__(attr_name):
    # 从 numpy._core 模块中导入 umath 对象
    from numpy._core import umath
    # 从当前模块中导入 _raise_warning 函数
    from ._utils import _raise_warning
    # 尝试获取 umath 对象中名为 attr_name 的属性，如果不存在则返回 None
    ret = getattr(umath, attr_name, None)
    # 如果未找到指定的属性，则抛出 AttributeError 异常
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core.umath' has no attribute {attr_name}")
    # 调用 _raise_warning 函数，警告 umath 模块中已获取属性
    _raise_warning(attr_name, "umath")
    # 返回获取到的属性
    return ret
```