# `.\pytorch\functorch\dim\wrap_type.py`

```
# 导入需要的模块和类
from types import (
    BuiltinMethodType,
    FunctionType,
    GetSetDescriptorType,
    MethodDescriptorType,
    WrapperDescriptorType,
)
# 从functorch._C模块中导入dim作为_C
from functorch._C import dim as _C

# 将_C._wrap_method赋值给_wrap_method，用于后续方法包装
_wrap_method = _C._wrap_method

# 定义函数类型的元组，包括FunctionType、MethodDescriptorType、BuiltinMethodType、WrapperDescriptorType
FUNC_TYPES = (
    FunctionType,
    MethodDescriptorType,
    BuiltinMethodType,
    WrapperDescriptorType,
)
# 定义属性类型的元组，包括GetSetDescriptorType和property
PROPERTY_TYPES = (GetSetDescriptorType, property)

# 定义用于包装方法的函数_py_wrap_method，接受原始方法orig和__torch_function__作为参数
def _py_wrap_method(orig, __torch_function__):
    # 定义实现函数impl，调用__torch_function__处理原始方法的参数和关键字参数
    def impl(*args, **kwargs):
        return __torch_function__(orig, None, args, kwargs)
    return impl

# 定义wrap_type函数，用于根据use_c标志选择方法包装方式，并根据pattern和__torch_function__对to_patch进行方法包装
def wrap_type(use_c, to_patch, pattern, __torch_function__):
    # 根据use_c标志选择方法包装函数
    if use_c:
        wrap_method = _wrap_method
    else:
        wrap_method = _py_wrap_method

    # 构建所有祖先类的字典all，跳过object类
    all = {}
    for t in reversed(pattern.mro()[:-1]):  # 跳过object类
        all.update(t.__dict__)

    # 定义wrap_attr函数，用于包装属性
    def wrap_attr(orig):
        return property(wrap_method(orig.__get__, __torch_function__))

    # 遍历所有的类成员
    for name, obj in all.items():
        # 跳过特定的成员名
        if name in (
            "__dict__",
            "__new__",
            "__init__",
            "__repr__",
            "__weakref__",
            "__doc__",
            "__module__",
            "__dir__",
        ):
            continue

        # 如果to_patch已经有该成员并且不等于来自object类的同名成员，则跳过
        if hasattr(to_patch, name) and getattr(to_patch, name) is not getattr(
            object, name, None
        ):
            continue

        # 根据成员类型进行方法或属性包装
        if isinstance(obj, FUNC_TYPES):
            setattr(to_patch, name, wrap_method(obj, __torch_function__))
        elif isinstance(obj, PROPERTY_TYPES):
            setattr(to_patch, name, wrap_attr(obj))
```