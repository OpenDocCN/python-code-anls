# `.\numpy\numpy\core\shape_base.py`

```
# 定义一个特殊方法 __getattr__，用于在访问对象的属性失败时自定义处理
def __getattr__(attr_name):
    # 从 numpy._core 模块导入 shape_base 对象
    from numpy._core import shape_base
    # 从当前模块的 _utils 模块导入 _raise_warning 函数
    from ._utils import _raise_warning
    # 尝试获取 shape_base 对象中的属性 attr_name
    ret = getattr(shape_base, attr_name, None)
    # 如果未找到对应属性，抛出 AttributeError 异常
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core.shape_base' has no attribute {attr_name}")
    # 发出警告，说明属性 attr_name 未直接在 shape_base 中找到
    _raise_warning(attr_name, "shape_base")
    # 返回找到的属性对象
    return ret
```