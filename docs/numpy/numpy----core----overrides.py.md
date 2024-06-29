# `.\numpy\numpy\core\overrides.py`

```py
# 定义一个特殊方法 __getattr__，用于动态获取属性的值
def __getattr__(attr_name):
    # 从 numpy._core 模块中导入 overrides 函数
    from numpy._core import overrides
    # 从当前模块的 _utils 模块中导入 _raise_warning 函数
    from ._utils import _raise_warning
    # 获取 overrides 模块中名为 attr_name 的属性
    ret = getattr(overrides, attr_name, None)
    # 如果未找到对应属性，抛出 AttributeError 异常
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core.overrides' has no attribute {attr_name}")
    # 调用 _raise_warning 函数，生成警告信息
    _raise_warning(attr_name, "overrides")
    # 返回获取到的属性值
    return ret
```