# `.\numpy\numpy\core\function_base.py`

```
# 定义一个函数 __getattr__，用于获取指定属性名的属性值
def __getattr__(attr_name):
    # 从 numpy._core 中导入 function_base 模块
    from numpy._core import function_base
    # 从 ._utils 模块导入 _raise_warning 函数
    from ._utils import _raise_warning
    # 获取 function_base 模块中名称为 attr_name 的属性值，如果不存在则返回 None
    ret = getattr(function_base, attr_name, None)
    # 如果未找到指定的属性值，则抛出 AttributeError 异常
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core.function_base' has no attribute {attr_name}")
    # 发出警告，说明找到的属性值来自 function_base 模块
    _raise_warning(attr_name, "function_base")
    # 返回找到的属性值
    return ret
```