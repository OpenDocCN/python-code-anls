# `.\pytorch\torch\_dynamo\variables\torch_function.py`

```py
# 忽略 mypy 错误，不进行类型检查
# 导入 inspect 模块，用于获取对象信息
import inspect
# 导入类型提示模块 Dict、List、TYPE_CHECKING
from typing import Dict, List, TYPE_CHECKING

# 导入 torch.utils._pytree 模块作为 pytree
import torch.utils._pytree as pytree

# 从 torch.overrides 模块导入 _get_overloaded_args、get_default_nowrap_functions 函数
from torch.overrides import _get_overloaded_args, get_default_nowrap_functions

# 导入自定义异常 unimplemented
from ..exc import unimplemented
# 导入 guards 模块中的 GuardBuilder、install_guard 函数
from ..guards import GuardBuilder, install_guard
# 导入 source 模块中的 AttrSource、GlobalSource、TypeSource 类
from ..source import AttrSource, GlobalSource, TypeSource
# 导入 utils 模块中的 has_torch_function、is_tensor_base_attr_getter 函数
from ..utils import has_torch_function, is_tensor_base_attr_getter

# 从本地模块中导入 constant 模块中的 ConstantVariable 类
from .constant import ConstantVariable
# 从本地模块中导入 lists 模块中的 TupleVariable 类
from .lists import TupleVariable
# 从本地模块中导入 tensor 模块中的 TensorSubclassVariable、TensorVariable 类
from .tensor import TensorSubclassVariable, TensorVariable
# 从本地模块中导入 user_defined 模块中的 UserDefinedObjectVariable 类
from .user_defined import UserDefinedObjectVariable

# 如果 TYPE_CHECKING 为真，则从 base 模块中导入 VariableTracker 类
if TYPE_CHECKING:
    from .base import VariableTracker


# 禁止调用以下默认不包装函数的属性名
banned_attrs = [
    fn.__self__.__name__  # 获取默认不包装函数的对象的名称
    for fn in get_default_nowrap_functions()  # 遍历所有默认不包装函数
    if is_tensor_base_attr_getter(fn)  # 如果函数是获取张量基础属性的函数
]


def _get_all_args(args, kwargs):
    # 返回所有参数和关键字参数的扁平化变量追踪树叶子节点
    return _flatten_vts(pytree.arg_tree_leaves(*args, **kwargs))


def _flatten_vts(vts):
    # 导入所需模块和类
    from collections import deque
    from .dicts import ConstDictVariable
    from .lazy import LazyVariableTracker
    from .lists import ListVariable

    vts = deque(vts)  # 使用双端队列处理变量追踪系统
    output = []

    while vts:
        vt = vts.pop()  # 弹出队列中的变量追踪对象
        LazyVariableTracker.realize_all(vt)  # 确保懒惰变量追踪的所有内容已实现
        if isinstance(vt, ListVariable):
            vts.extend(vt.items)  # 如果是列表变量，则将其元素添加到队列中
        elif isinstance(vt, ConstDictVariable):
            vts.extend(vt.items.values())  # 如果是常量字典变量，则将其值添加到队列中
        else:
            output.append(vt)  # 否则将变量追踪对象添加到输出列表中

    return output  # 返回扁平化后的变量追踪对象列表


def _get_subclass_type(var):
    # 断言变量是 TensorWithTFOverrideVariable 或 UserDefinedObjectVariable 类型的实例
    assert isinstance(var, (TensorWithTFOverrideVariable, UserDefinedObjectVariable))
    # 返回一个变量的 Python 类型的结果
    return var.python_type()
def _get_subclass_type_var(tx, var):
    # 断言变量是 TensorWithTFOverrideVariable 或者 UserDefinedObjectVariable 类的实例
    assert isinstance(var, (TensorWithTFOverrideVariable, UserDefinedObjectVariable))
    
    if isinstance(var, TensorWithTFOverrideVariable):
        # 如果是 TensorWithTFOverrideVariable 类型的变量，则调用其 class_type_var 方法
        return var.class_type_var(tx)
    elif isinstance(var, UserDefinedObjectVariable):
        # 如果是 UserDefinedObjectVariable 类型的变量
        from .builder import SourcelessBuilder, VariableBuilder
        
        if var.source:
            # 如果有源代码，则使用 VariableBuilder 创建变量构建器
            return VariableBuilder(tx, TypeSource(var.source))(var.python_type())
        else:
            # 如果没有源代码，则使用 SourcelessBuilder 创建变量构建器
            return SourcelessBuilder.create(tx, var.python_type())


def _is_attr_overidden(tx, var, name):
    # 导入 torch 模块
    import torch

    overridden = False
    try:
        # 尝试获取静态属性值
        attr_val = inspect.getattr_static(var.python_type(), name)
        # 检查是否被重载
        overridden |= attr_val != getattr(torch.Tensor, name)
    except AttributeError:
        pass

    return overridden


def call_torch_function(
    tx, torch_function_type, torch_function_var, fn, types, args, kwargs
):
    # 导入 SourcelessBuilder 类
    from .builder import SourcelessBuilder

    # 构建 tf_args 元组，包含调用 torch 函数所需的参数
    tf_args = (
        torch_function_type,
        fn,
        types,
        SourcelessBuilder.create(tx, tuple(args)),
        SourcelessBuilder.create(tx, kwargs),
    )
    # 调用 tx 的 inline_user_function_return 方法，并返回结果
    return tx.inline_user_function_return(torch_function_var, tf_args, {})


def build_torch_function_fn(tx, value, source):
    # 导入 SourcelessBuilder 和 VariableBuilder 类
    from .builder import SourcelessBuilder, VariableBuilder

    if source:
        # 如果有源代码，则创建 AttrSource 和 VariableBuilder 对象
        return VariableBuilder(
            tx,
            AttrSource(AttrSource(source, "__torch_function__"), "__func__"),
        )(value.__torch_function__.__func__)
    else:
        # 如果没有源代码，则使用 SourcelessBuilder 创建对象
        return SourcelessBuilder.create(tx, value.__torch_function__.__func__)


def can_dispatch_torch_function(tx, args, kwargs):
    # 返回 tx.output.torch_function_enabled 和 args 中是否存在具有 torch function 的参数的逻辑与结果
    return tx.output.torch_function_enabled and any(
        has_torch_function(arg) for arg in _get_all_args(args, kwargs)
    )


def dispatch_torch_function(tx, fn, args, kwargs):
    """Gathers all args that are TensorWithTFOverrideVariable and dispatches based on the ordering in _get_overloaded_args"""

    # 获取所有具有 torch function 的参数
    all_args = _get_all_args(args, kwargs)
    # 根据 _get_overloaded_args 方法的顺序获取重载参数
    overloaded_args = _get_overloaded_args(
        [arg for arg in all_args if has_torch_function(arg)],
        _get_subclass_type,
    )

    for arg in overloaded_args:
        # 对每个重载参数调用 call_torch_function 方法
        res = arg.call_torch_function(
            tx,
            fn,
            TupleVariable([_get_subclass_type_var(tx, arg) for arg in overloaded_args]),
            args,
            kwargs,
        )

        if not (isinstance(res, ConstantVariable) and res.value is NotImplemented):
            return res

    # 如果所有重载的 __torch_function__ 方法都返回了 NotImplemented，则抛出未实现异常
    unimplemented(
        f"All __torch_function__ overrides for call {fn} with args {args} and kwargs {kwargs} returned NotImplemented"
    )


class TensorWithTFOverrideVariable(TensorVariable):
    """
    Represents a tensor subclass instance with a __torch_function__ override.
    """
    def __init__(self, *args, **kwargs):
        # 将关键字参数中的 torch_function_fn 弹出并保存到实例变量中
        self.torch_function_fn = kwargs.pop("torch_function_fn")
        # 调用父类的构造函数，传递其余的位置参数和关键字参数
        super().__init__(*args, **kwargs)

    @classmethod
    def from_tensor_var(cls, tx, tensor_var, class_type, torch_function_fn):
        import torch

        # 创建一个包含 tensor_var 的字典副本
        kwargs = dict(tensor_var.__dict__)
        # 断言 class_type 是 torch.Tensor 类型，否则抛出异常
        assert (
            kwargs.pop("class_type") is torch.Tensor
        ), "invalid class type in TensorWithTFOverrideVariable.from_tensor_var"
        # 使用给定的参数实例化类，并返回该实例
        var = cls(torch_function_fn=torch_function_fn, class_type=class_type, **kwargs)
        # 在 tx 上安装全局设置
        var.install_global(tx)
        return var

    def install_global(self, tx):
        # 将子类类型存储起来，以便在需要时重新包装输出张量
        if self.global_mangled_class_name(tx) not in tx.output.global_scope:
            # 安全地安装全局设置，使用全局名和类类型
            tx.output.install_global_unsafe(
                self.global_mangled_class_name(tx), self.class_type
            )

    def python_type(self):
        # 返回实例的类类型
        return self.class_type

    def class_type_var(self, tx):
        # 返回一个 TensorSubclassVariable 对象，使用类类型和全局源
        return TensorSubclassVariable(
            self.class_type, source=GlobalSource(self.global_mangled_class_name(tx))
        )

    def global_mangled_class_name(self, tx):
        # global_mangled_class_name 应该对于不同的 torch.compile 调用是不同的。
        # 否则，可能会出现多个 torch.compile 调用重用相同的全局名称的情况，
        # 但全局的生命周期与第一个调用相关联（并且可能在第一个 torch.compile 调用被删除时被删除）。
        # 根据输出图的 id 对其进行混淆处理。
        compile_id = tx.output.compile_id
        return f"__subclass_{self.class_type.__name__}_{id(self.class_type)}_c{compile_id}"
    # 定义一个方法，用于获取对象的属性
    def var_getattr(self, tx, name):
        # 引入 torch 模块和 SourcelessBuilder 类
        import torch
        from .builder import SourcelessBuilder

        # 如果属性名在 banned_attrs 列表中，报错并提示不支持操作
        if name in banned_attrs:
            unimplemented(
                f"Accessing {name} on a tensor subclass with a __torch_function__ override is not supported"
            )

        # 如果检测到属性被覆盖，则报错并提示不支持操作
        if _is_attr_overidden(tx, self, name):
            unimplemented(
                f"Accessing overridden method/attribute {name} on a tensor"
                " subclass with a __torch_function__ override is not supported"
            )

        # 如果 tx.output.torch_function_enabled 为真且 torch.Tensor 拥有属性名 name
        if tx.output.torch_function_enabled and hasattr(torch.Tensor, name):
            # 如果 self.source 存在，则安装一个保护来保证属性访问的正确性
            if self.source:
                install_guard(
                    AttrSource(AttrSource(self.source, "__class__"), name).make_guard(
                        GuardBuilder.FUNCTION_MATCH
                    )
                )
            # 使用 SourcelessBuilder 创建一个函数调用对象 get_fn
            get_fn = SourcelessBuilder.create(tx, getattr(torch.Tensor, name).__get__)

            # 调用 call_torch_function 方法来执行 torch 函数的调用
            return self.call_torch_function(
                tx,
                get_fn,
                TupleVariable([self.class_type_var(tx)]),
                [self],
                {},
            )
        else:
            # 否则调用父类的 var_getattr 方法处理属性访问
            return super().var_getattr(tx, name)

    # 定义一个方法，用于调用 torch 函数
    def call_torch_function(self, tx, fn, types, args, kwargs):
        return call_torch_function(
            tx,
            self.class_type_var(tx),
            self.torch_function_fn,
            fn,
            types,
            args,
            kwargs,
        )

    # 定义一个方法，用于调用对象的方法
    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
        # 定义函数的返回类型为 "VariableTracker"
        # 这段代码实现了在 `call_method` 方法中内联 `__torch_function__` 的重写
        if tx.output.torch_function_enabled:
            # 导入必要的库和模块
            import torch
            from .builder import SourcelessBuilder, VariableBuilder

            # 检查方法是否被子类重写，如果是则抛出异常
            if _is_attr_overidden(tx, self, name):
                unimplemented(
                    f"Calling overridden method {name} on a tensor"
                    " subclass with a __torch_function__ override is not supported"
                )

            # [注意: __torch_function__] 目前我们只支持在张量上定义的方法
            # 如果方法不是在张量上定义的，我们会在这里中断图形构建，需要对提取方法/比较方法进行更大的改进
            # 通过上述检查我们已经确认方法没有被重写，因此我们保证方法与在张量上定义的实现相同，并获取它
            if self.source:
                # 如果存在源码，则使用 VariableBuilder 根据源码和方法名创建函数变量
                func_var = VariableBuilder(
                    tx, AttrSource(AttrSource(self.source, "__class__"), name)
                )(inspect.getattr_static(self.python_type(), name))
            else:
                # 如果没有源码，则使用 SourcelessBuilder 根据 torch.Tensor 的方法名创建函数变量
                func_var = SourcelessBuilder.create(tx, getattr(torch.Tensor, name))
            
            # 调用 dispatch_torch_function 函数来处理 torch_function 调度
            return dispatch_torch_function(tx, func_var, [self] + args, kwargs)
        else:
            # 如果 torch_function 未启用，则调用父类的 call_method 方法
            return super().call_method(tx, name, args, kwargs)
```