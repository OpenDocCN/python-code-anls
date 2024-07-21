# `.\pytorch\torchgen\api\native.py`

```py
# 从 __future__ 导入 annotations，确保支持类型注解的早期版本兼容性
from __future__ import annotations

# 从 typing 模块导入 Sequence 类型用于类型提示
from typing import Sequence

# 从 torchgen 模块导入 local, cpp 和 types 模块
from torchgen import local
from torchgen.api import cpp
from torchgen.api.types import (
    ArgName,
    BaseCType,
    Binding,
    boolT,
    ConstRefCType,
    CType,
    deviceT,
    layoutT,
    ListCType,
    MutRefCType,
    NamedCType,
    OptionalCType,
    scalarT,
    scalarTypeT,
    tensorT,
)

# 从 torchgen.model 模块导入多个类和函数
from torchgen.model import (
    Argument,
    FunctionSchema,
    Return,
    SelfArgument,
    TensorOptionsArguments,
    Type,
)

# 从 torchgen.utils 模块导入 assert_never 函数

# 本文件描述了将 JIT schema 翻译为本地函数 API 的过程。
# 这个 API 非常类似于 C++ API（这在历史上是有道理的，因为最初是希望通过编写本地函数来实现 C++ API 中的函数），
# 但随着时间的推移，我们已经演变了 C++ API，而没有实际更改我们的 native:: 内核。
# 目的是尽可能使本地 API 和调度器 API 尽可能接近，因为这样可以最大程度地减少开销（不需要从调度器 API 转换为本地 API）。
#
# 注意：这里是 symint 意识的，对于某些分发条目，你将得到非 SymInt 变体，而对于其他分发条目，则是 SymInt 变体。

# 定义一个函数，根据给定的 FunctionSchema 对象返回其名称的字符串表示
def name(func: FunctionSchema) -> str:
    name = str(func.name.name)
    # TODO: delete this!
    # 如果函数是输出函数，则在函数名后面添加 "_out"
    if func.is_out_fn():
        name += "_out"
    # 如果存在重载名称，则在函数名后面添加下划线和重载名称
    if func.name.overload_name:
        name += f"_{func.name.overload_name}"
    return name

# 定义一个函数，根据给定的 Type 对象返回相应的 NamedCType 类型
def argumenttype_type(
    t: Type, *, mutable: bool, binds: ArgName, symint: bool
) -> NamedCType:
    if str(t) == "Tensor?":
        # 如果类型是 Tensor?，返回一个 OptionalCType 类型的 NamedCType 对象
        tensor_type: OptionalCType = OptionalCType(BaseCType(tensorT))
        # 根据 mutable 参数决定返回的 NamedCType 对象是否是 MutRefCType 类型
        if mutable and not local.use_const_ref_for_mutable_tensors():
            return NamedCType(binds, MutRefCType(tensor_type))
        else:
            return NamedCType(binds, ConstRefCType(tensor_type))
    elif str(t) == "Tensor?[]":
        # 如果类型是 Tensor?[]，返回一个 ListCType 包裹的 OptionalCType 类型的 NamedCType 对象
        return NamedCType(
            binds, ConstRefCType(ListCType(OptionalCType(BaseCType(tensorT))))
        )
    elif str(t) == "Scalar":
        # 如果类型是 Scalar，返回一个 ConstRefCType 类型的 NamedCType 对象
        return NamedCType(binds, ConstRefCType(BaseCType(scalarT)))
    elif str(t) == "Scalar?":
        # 如果类型是 Scalar?，返回一个 ConstRefCType 包裹的 OptionalCType 类型的 NamedCType 对象
        return NamedCType(binds, ConstRefCType(OptionalCType(BaseCType(scalarT))))
    # 对于其他类型，调用 cpp 模块的 argumenttype_type 函数进行处理
    return cpp.argumenttype_type(t, mutable=mutable, binds=binds, symint=symint)

# 定义一个函数，根据给定的 Return 对象序列返回相应的 CType 类型
def returns_type(rs: Sequence[Return], *, symint: bool) -> CType:
    return cpp.returns_type(rs, symint=symint)

# 定义一个函数，根据给定的 Argument 对象返回相应的 NamedCType 类型
def argument_type(a: Argument, *, binds: ArgName, symint: bool) -> NamedCType:
    return argumenttype_type(a.type, mutable=a.is_write, binds=binds, symint=symint)

# 定义一个函数，根据给定的 Argument、SelfArgument 或 TensorOptionsArguments 对象返回 Binding 对象列表
def argument(
    a: Argument | SelfArgument | TensorOptionsArguments,
    *,
    is_out: bool,
    symint: bool,
) -> list[Binding]:
    # 理想情况下，我们永远不会默认本地函数。但是，有一些函数直接调用 native:: 并依赖于默认存在。
    # 因此，为了向后兼容性，我们为非输出变体生成默认值（但不生成输出变体的默认值）。
    # 根据是否为输出参数确定是否需要生成默认值
    should_default = not is_out
    
    # 如果参数 a 是 Argument 类型
    if isinstance(a, Argument):
        # 默认值初始化为 None
        default: str | None = None
        
        # 如果需要生成默认值且参数 a 有默认值，则使用 cpp.default_expr 生成默认表达式
        if should_default and a.default is not None:
            default = cpp.default_expr(a.default, a.type, symint=symint)
        
        # 返回一个 Binding 对象列表，表示参数绑定
        return [
            Binding(
                nctype=argument_type(a, binds=a.name, symint=symint),
                name=a.name,
                default=default,
                argument=a,
            )
        ]
    
    # 如果参数 a 是 SelfArgument 类型
    elif isinstance(a, SelfArgument):
        # 忽略 SelfArgument 的区别，继续处理其内部的 argument
        return argument(a.argument, is_out=is_out, symint=symint)
    
    # 如果参数 a 是 TensorOptionsArguments 类型
    elif isinstance(a, TensorOptionsArguments):
        # 默认值初始化为 None
        default = None
        
        # 如果需要生成默认值，则使用空的字符串表示默认值
        if should_default:
            default = "{}"
        
        # 返回包含多个 Binding 对象的列表，分别表示 dtype、layout、device、pin_memory 参数的绑定
        return [
            Binding(
                nctype=NamedCType("dtype", OptionalCType(BaseCType(scalarTypeT))),
                name="dtype",
                default=default,
                argument=a,
            ),
            Binding(
                nctype=NamedCType("layout", OptionalCType(BaseCType(layoutT))),
                name="layout",
                default=default,
                argument=a,
            ),
            Binding(
                nctype=NamedCType("device", OptionalCType(BaseCType(deviceT))),
                name="device",
                default=default,
                argument=a,
            ),
            Binding(
                nctype=NamedCType("pin_memory", OptionalCType(BaseCType(boolT))),
                name="pin_memory",
                default=default,
                argument=a,
            ),
        ]
    
    # 如果以上条件都不满足，断言永远不应该执行到这里，即参数 a 的类型不应该存在
    else:
        assert_never(a)
# 定义一个函数 arguments，接受一个 FunctionSchema 类型的参数 func 和一个命名关键字参数 symint，返回一个列表，列表元素类型为 Binding
def arguments(func: FunctionSchema, *, symint: bool) -> list[Binding]:
    # 初始化一个空列表 args，用于存储不同类型的参数对象
    args: list[Argument | TensorOptionsArguments | SelfArgument] = []
    # 将 func.arguments.non_out 中的所有参数对象添加到 args 列表中
    args.extend(func.arguments.non_out)
    # 将 func.arguments.out 中的所有参数对象添加到 args 列表中
    args.extend(func.arguments.out)
    # 返回一个列表推导式，遍历 args 列表中的每个参数对象 arg，并调用 argument 函数处理，将结果添加到最终的返回列表中
    return [
        r for arg in args for r in argument(arg, symint=symint, is_out=func.is_out_fn())
    ]
```