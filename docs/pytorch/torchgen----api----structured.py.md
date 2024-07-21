# `.\pytorch\torchgen\api\structured.py`

```py
# 引入将来版本的注解功能
from __future__ import annotations

# 从 torchgen.api 模块导入 cpp 对象
from torchgen.api import cpp
# 从 torchgen.api.types 模块导入多个类型
from torchgen.api.types import (
    ArgName,             # 参数名称类型
    ArrayRefCType,       # 数组引用类型
    BaseCType,           # 基本类型
    Binding,             # 绑定类型
    ConstRefCType,       # 常量引用类型
    dimnameListT,        # 维度名称列表类型
    intArrayRefT,        # 整型数组引用类型
    iOptTensorListRefT,  # 可选张量列表引用类型
    iTensorListRefT,     # 张量列表引用类型
    NamedCType,          # 命名类型
    OptionalCType,       # 可选类型
    optionalIntArrayRefT,# 可选整型数组引用类型
    optionalScalarRefT,  # 可选标量引用类型
    optionalTensorRefT,  # 可选张量引用类型
    scalarT,             # 标量类型
    tensorT,             # 张量类型
)
# 从 torchgen.model 模块导入多个类型
from torchgen.model import (
    Argument,                 # 参数类型
    BaseTy,                   # 基础类型
    BaseType,                 # 基础类型
    ListType,                 # 列表类型
    NativeFunctionsGroup,     # 原生函数组类型
    OptionalType,             # 可选类型
    SelfArgument,             # 自身参数类型
    TensorOptionsArguments,   # 张量选项参数类型
    Type,                     # 类型类型
)
# 从 torchgen.utils 模块导入 assert_never 函数
from torchgen.utils import assert_never


# 本文件描述了 JIT 架构到结构化函数 API 的转换。
# 这与原生 API 类似，但修复了一些原生 API 的历史问题。

# 将出现在 JIT 参数中的类型转换为 C++ 参数类型。
# 注意：目前 mutable 参数没有实际作用；但如果我们增加一些名义类型，它可能会起作用。
def argumenttype_type(t: Type, *, mutable: bool, binds: ArgName) -> NamedCType:
    # 如果是值类型，则进行值类型的转换
    # 注意：结构化内核始终关闭 symint，因为它们涉及需要真实整数的实际内核。
    # 唯一的例外是 CompositeExplicitAutograd 和元函数（理论上可以是 SymInt），
    # 但为简单起见，我们计划仅在 Python 中处理它们。
    r = cpp.valuetype_type(t, symint=False, binds=binds)
    if r is not None:
        return r

    if isinstance(t, BaseType):
        if t.name == BaseTy.Tensor:
            return NamedCType(binds, ConstRefCType(BaseCType(tensorT)))
        elif t.name == BaseTy.Scalar:
            return NamedCType(binds, ConstRefCType(BaseCType(scalarT)))
        else:
            raise AssertionError(f"base type should have been value type {t}")
    elif isinstance(t, OptionalType):
        if t.elem == BaseType(BaseTy.Tensor):
            return NamedCType(binds, BaseCType(optionalTensorRefT))
        elif t.elem == BaseType(BaseTy.Scalar):
            return NamedCType(binds, BaseCType(optionalScalarRefT))
        elif isinstance(t.elem, ListType) and str(t.elem.elem) == "int":
            return NamedCType(binds, BaseCType(optionalIntArrayRefT))
        elem = argumenttype_type(t.elem, mutable=mutable, binds=binds)
        return NamedCType(binds, OptionalCType(elem.type))
    # 如果对象 t 是 ListType 的实例
    elif isinstance(t, ListType):
        # 如果列表的元素类型是 BaseType(BaseTy.Tensor)
        if t.elem == BaseType(BaseTy.Tensor):
            # 返回命名的 C 类型，使用 binds 和 ConstRefCType 封装的 BaseCType(iTensorListRefT)
            return NamedCType(binds, ConstRefCType(BaseCType(iTensorListRefT)))
        # 如果列表的元素类型是 OptionalType(BaseType(BaseTy.Tensor))
        elif t.elem == OptionalType(BaseType(BaseTy.Tensor)):
            # 返回命名的 C 类型，使用 binds 封装的 BaseCType(iOptTensorListRefT)
            return NamedCType(binds, BaseCType(iOptTensorListRefT))
        # 对于特殊情况 "int"
        # TODO: 删除这些特殊情况；参见 torchgen.api.cpp--这些必须同时更改，但存在问题；参见 https://github.com/pytorch/pytorch/pull/51485
        elif str(t.elem) == "int":
            # 返回命名的 C 类型，使用 binds 封装的 BaseCType(intArrayRefT)
            return NamedCType(binds, BaseCType(intArrayRefT))
        # 对于特殊情况 "Dimname"
        elif str(t.elem) == "Dimname":
            # 返回命名的 C 类型，使用 binds 封装的 BaseCType(dimnameListT)
            return NamedCType(binds, BaseCType(dimnameListT))
        # 对列表的元素类型进行进一步处理，递归调用 argumenttype_type 函数
        elem = argumenttype_type(t.elem, mutable=mutable, binds=binds)
        # 返回命名的 C 类型，使用 binds 封装的 ArrayRefCType(elem.type)
        return NamedCType(binds, ArrayRefCType(elem.type))
    # 如果 t 不是 ListType 的实例，则抛出断言错误，显示未识别的类型信息
    else:
        raise AssertionError(f"unrecognized type {repr(t)}")
# 定义一个函数 argument_type，接受一个类型为 Argument 的参数 a 和一个名为 binds 的关键字参数，返回一个 NamedCType 类型的值
def argument_type(a: Argument, *, binds: ArgName) -> NamedCType:
    # 调用 argumenttype_type 函数，传入参数 a.type 作为类型，a.is_write 作为 mutable 参数，binds 参数作为 binds 参数
    return argumenttype_type(a.type, mutable=a.is_write, binds=binds)


# 返回类型被有意省略，因为结构化内核永远不会 "返回"；相反，它们总是间接报告它们的输出
# （在元函数的情况下，通过调用 set_output；在实现函数的情况下，通过直接写入提供的 out 参数）。

# 结构化内核永不使用默认值
def argument(a: Argument | SelfArgument | TensorOptionsArguments) -> list[Binding]:
    # 如果 a 是 Argument 类型
    if isinstance(a, Argument):
        # 返回包含一个 Binding 对象的列表，其中 nctype 是 argument_type 返回的值，name 是 a.name，default 是 None，argument 是 a
        return [
            Binding(
                nctype=argument_type(a, binds=a.name),
                name=a.name,
                default=None,
                argument=a,
            )
        ]
    # 如果 a 是 SelfArgument 类型
    elif isinstance(a, SelfArgument):
        # 递归调用 argument 函数，传入 a.argument 作为参数，并返回其结果
        return argument(a.argument)
    # 如果 a 是 TensorOptionsArguments 类型
    elif isinstance(a, TensorOptionsArguments):
        # 抛出断言错误，因为结构化内核尚不支持 TensorOptions
        raise AssertionError("structured kernels don't support TensorOptions yet")
    else:
        # 否则，使用 assert_never 函数，确保对于 a 的所有类型，都会引发 AssertionError
        assert_never(a)


# 返回实现函数的参数列表
def impl_arguments(g: NativeFunctionsGroup) -> list[Binding]:
    # 初始化 args 列表，包含 Argument、TensorOptionsArguments 和 SelfArgument 类型的元素
    args: list[Argument | TensorOptionsArguments | SelfArgument] = []

    # 如果 g.out.precomputed 为 True
    if g.out.precomputed:
        # non_out_args_replaced 是 g.out.func.arguments.non_out 中某些参数被预先计算版本替换后的列表
        non_out_args_replaced: list[
            Argument | TensorOptionsArguments | SelfArgument
        ] = []
        # 遍历 g.out.func.arguments.non_out 中的每个参数 a
        for a in g.out.func.arguments.non_out:
            # 如果 a 是 Argument 类型，并且 a.name 在 g.out.precomputed.replace 中
            if isinstance(a, Argument) and a.name in g.out.precomputed.replace:
                # 将应替换 a 的参数添加到 non_out_args_replaced 列表中
                non_out_args_replaced.extend(g.out.precomputed.replace[a.name])
            else:
                # 否则，将 a 添加到 non_out_args_replaced 列表中
                non_out_args_replaced.append(a)

        # 将 non_out_args_replaced 列表中的参数添加到 args 列表末尾
        args.extend(non_out_args_replaced)
        # 将 g.out.precomputed.add 中的参数添加到 args 列表末尾
        args.extend(g.out.precomputed.add)
    else:
        # 如果 g.out.precomputed 为 False，则将 g.out.func.arguments.non_out 中的参数添加到 args 列表末尾
        args.extend(g.out.func.arguments.non_out)

    # 将 g.out.func.arguments.out 中的参数添加到 args 列表末尾
    args.extend(g.out.func.arguments.out)
    # 返回由调用 argument 函数处理后的 args 列表的结果列表
    return [r for arg in args for r in argument(arg)]


# 返回元函数的参数列表
def meta_arguments(g: NativeFunctionsGroup) -> list[Binding]:
    # 初始化 args 列表，包含 Argument、TensorOptionsArguments 和 SelfArgument 类型的元素
    args: list[Argument | TensorOptionsArguments | SelfArgument] = []
    # 将 g.functional.func.arguments.non_out 中的参数添加到 args 列表末尾
    args.extend(g.functional.func.arguments.non_out)
    # 返回由调用 argument 函数处理后的 args 列表的结果列表
    return [r for arg in args for r in argument(arg)]


# 返回输出参数列表
def out_arguments(g: NativeFunctionsGroup) -> list[Binding]:
    # 初始化 args 列表，包含 Argument、TensorOptionsArguments 和 SelfArgument 类型的元素
    args: list[Argument | TensorOptionsArguments | SelfArgument] = []
    # 将 g.out.func.arguments.out 中的参数添加到 args 列表末尾
    args.extend(g.out.func.arguments.out)
    # 返回由调用 argument 函数处理后的 args 列表的结果列表
    return [r for arg in args for r in argument(arg)]
```