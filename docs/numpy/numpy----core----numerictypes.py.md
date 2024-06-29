# `.\numpy\numpy\core\numerictypes.py`

```py
# 定义一个特殊方法 __getattr__，用于在属性未找到时执行
def __getattr__(attr_name):
    # 从 numpy._core 模块导入 numerictypes 对象
    from numpy._core import numerictypes
    # 从 ._utils 模块导入 _raise_warning 函数
    from ._utils import _raise_warning
    # 尝试获取 numerictypes 模块中的属性 attr_name
    ret = getattr(numerictypes, attr_name, None)
    # 如果未找到该属性，则抛出 AttributeError 异常
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core.numerictypes' has no attribute {attr_name}")
    # 调用 _raise_warning 函数，传递 attr_name 和 "numerictypes" 作为参数
    _raise_warning(attr_name, "numerictypes")
    # 返回获取到的属性 ret
    return ret
```