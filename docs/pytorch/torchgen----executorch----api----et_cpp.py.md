# `.\pytorch\torchgen\executorch\api\et_cpp.py`

```
# 从未来导入特定注释，确保向后兼容性
from __future__ import annotations

# 导入类型提示模块
from typing import Sequence

# 导入本地模块和库
from torchgen import local

# 导入 TorchGen API 中的类型定义
from torchgen.api.types import (
    ArgName,
    ArrayCType,
    BaseCType,
    Binding,
    ConstRefCType,
    CType,
    MutRefCType,
    NamedCType,
    SpecialArgName,
    TupleCType,
    VectorCType,
    voidT,
)

# 导入 TorchGen Executorch API 中的类型定义
from torchgen.executorch.api.types import (
    ArrayRefCType,
    BaseTypeToCppMapping,
    OptionalCType,
    scalarT,
    tensorListT,
    tensorT,
)

# 导入 TorchGen 模型定义
from torchgen.model import (
    Argument,
    Arguments,
    BaseTy,
    BaseType,
    ListType,
    NativeFunction,
    OptionalType,
    Return,
    SelfArgument,
    TensorOptionsArguments,
    Type,
)

# 导入 TorchGen 工具函数
from torchgen.utils import assert_never

"""
This file describes the translation of JIT schema to the public C++ API, which is what people use when they call
functions like at::add. It also serves as a native function API, which is the signature of kernels,
since in Executorch CppSignature is the same as NativeSignature.

Difference between this file and torchgen.api.cpp.py:

  - Executorch doesn't support TensorOptions, however in this file we still keep the logic here to be compatible with
    torchgen.api.cpp, so that we can do stuff like ATen mode (running ATen kernels in Executorch).

  - Executorch doesn't support Dimname.

  - Executorch runtime doesn't support SymInt, will treat it as int.
"""

# "值类型" 在 JIT schema 中到 C++ API 类型的转换。值类型无论是作为参数类型还是返回类型，外观都是相同的。
# 如果所讨论的类型不是值类型，则返回 None。
def valuetype_type(
    t: Type,
    *,
    binds: ArgName,
    remove_non_owning_ref_types: bool = False,
) -> NamedCType | None:
    if isinstance(t, BaseType):
        if t.name == BaseTy.Tensor or t.name == BaseTy.Scalar:
            return None
        # 对于 SymInt，我们简单地将其视为 int。
        elif str(t) == "SymInt":
            return NamedCType(binds, BaseCType(BaseTypeToCppMapping[BaseTy.int]))
        if remove_non_owning_ref_types:
            if t.name == BaseTy.str:
                raise AssertionError(
                    "string ref->value conversion: not implemented yet"
                )
        # 所有其他 BaseType 目前直接映射到 BaseCppTypes。
        return NamedCType(binds, BaseCType(BaseTypeToCppMapping[t.name]))
    elif isinstance(t, OptionalType):
        elem = valuetype_type(t.elem, binds=binds)
        if elem is None:
            return None
        return NamedCType(binds, OptionalCType(elem.type))
    elif isinstance(t, ListType):
        if str(t.elem) == "bool":
            assert t.size is not None
            return NamedCType(
                binds, ArrayCType(BaseCType(BaseTypeToCppMapping[BaseTy.bool]), t.size)
            )
        else:
            return None
    else:
        raise AssertionError(f"unrecognized type {repr(t)}")

# JIT 参数中出现的类型到 C++ 参数类型的转换。
# 如果设置了 remove_non_owning_ref_types，则确保输出的 CType 不是非拥有引用类型。
# 例如，我们会返回 std::vector<int> 而不是 IntArrayRef。
# 参见注释 [translation from C++ reference to value types]

# 将 JIT 参数类型转换为对应的 C++ 类型
def argumenttype_type(
    t: Type,
    *,
    mutable: bool,
    binds: ArgName,
    remove_non_owning_ref_types: bool = False,
) -> NamedCType:
    # 如果是值类型，进行值类型的转换
    r = valuetype_type(
        t,
        binds=binds,
        remove_non_owning_ref_types=remove_non_owning_ref_types,
    )
    if r is not None:
        return r
    # 如果是基础类型
    if isinstance(t, BaseType):
        if t.name == BaseTy.Tensor:
            # 对于 Tensor 类型的特殊处理
            if mutable and not local.use_const_ref_for_mutable_tensors():
                return NamedCType(binds, MutRefCType(BaseCType(tensorT)))
            else:
                return NamedCType(binds, ConstRefCType(BaseCType(tensorT)))
        elif t.name == BaseTy.Scalar:
            # 对于 Scalar 类型的处理
            return NamedCType(binds, ConstRefCType(BaseCType(scalarT)))
        else:
            raise AssertionError(f"base type should have been value type {t}")
    # 如果是可选类型
    elif isinstance(t, OptionalType):
        if str(t.elem) == "Tensor":
            # 对于可选的 Tensor 类型的处理
            if mutable and not local.use_const_ref_for_mutable_tensors():
                return NamedCType(
                    binds, MutRefCType(BaseCType(tensorT))
                )  # TODO: fix this discrepancy
            else:
                return NamedCType(
                    binds, ConstRefCType(OptionalCType(BaseCType(tensorT)))
                )
        elif str(t.elem) == "Scalar":
            # 对于可选的 Scalar 类型的处理
            return NamedCType(binds, ConstRefCType(OptionalCType(BaseCType(scalarT))))
        # 对于其他类型的可选类型进行递归处理
        elem = argumenttype_type(t.elem, mutable=mutable, binds=binds)
        return NamedCType(binds, OptionalCType(elem.type))
    # 如果是列表类型
    elif isinstance(t, ListType):
        # TODO: 对于 Tensor[] 和 Tensor?[] 保留这些特殊情况，以便与 ATen kernels 连接。
        if str(t.elem) == "Tensor":
            return NamedCType(binds, BaseCType(tensorListT))
        elif str(t.elem) == "Dimname":
            raise NotImplementedError("Executorch doesn't support Dimname")
        elif str(t.elem) == "Tensor?":
            # 对于可选的 Tensor[] 类型的处理
            return NamedCType(binds, ArrayRefCType(OptionalCType(BaseCType(tensorT))))
        # 对于其他类型的列表进行递归处理
        elem = argumenttype_type(t.elem, mutable=mutable, binds=binds)
        return NamedCType(binds, ArrayRefCType(elem.type))
    else:
        raise AssertionError(f"unrecognized type {repr(t)}")


# 将 JIT 参数转换为对应的 C++ 类型
def argument_type(a: Argument, *, binds: ArgName) -> NamedCType:
    return argumenttype_type(a.type, mutable=a.is_write, binds=binds)


# 将 JIT 的返回类型（非多返回类型）从 JIT 转换为 C++
# 注意：returntype_type 返回一个 CType，而不是 NamedCType。
# 主要是因为返回类型和返回名称之间的不匹配。
# 例如，返回类型为 'void' 的函数没有返回名称。
# 定义一个返回类型为 'std::tuple' 的函数，参数类型为 Type，返回类型为 CType
def returntype_type(t: Type, *, mutable: bool) -> CType:
    # 调用 valuetype_type 函数，传入参数 t 和绑定 binds="__placeholder__"
    r = valuetype_type(t, binds="__placeholder__")
    # 如果返回值 r 不为 None，则返回 r 的类型
    if r is not None:
        return r.type

    # 如果 t 是 BaseType 类型
    if isinstance(t, BaseType):
        # 如果 BaseType 的名称为 BaseTy.Tensor
        if t.name == BaseTy.Tensor:
            # 如果 mutable 为 True
            if mutable:
                # 如果本地设置使用 const 引用来处理可变张量
                if local.use_const_ref_for_mutable_tensors():
                    return ConstRefCType(BaseCType(tensorT))
                else:
                    return MutRefCType(BaseCType(tensorT))
            else:
                # 返回 BaseCType(tensorT)，即不可变张量的基本类型
                # 注意 [Tensor Copy Returns]
                # 目前，我们使用 "Argument.is_write" 来确定张量返回类型是复制还是引用。
                # 如果这一点有所改变，请查看此注释的其他位置！
                return BaseCType(tensorT)
        # 如果 BaseType 的名称为 BaseTy.Scalar
        elif t.name == BaseTy.Scalar:
            # 返回 BaseCType(scalarT)，标量的基本类型
            return BaseCType(scalarT)
    # 如果 t 是 ListType 类型
    elif isinstance(t, ListType):
        # 断言不可变的情况下，原生函数不应返回可变张量列表，应该返回 void。
        assert not mutable, "Native functions should never return a mutable tensor list. They should return void."
        # 递归调用 returntype_type 函数，获取列表元素的 CType 类型
        elem = returntype_type(t.elem, mutable=False)
        # 断言列表没有固定的大小
        assert t.size is None, f"fixed size list returns not supported: {t}"
        # 返回 VectorCType(elem)，即向量类型的 CType
        return VectorCType(elem)

    # 抛出断言错误，表示无法识别的返回类型 t
    raise AssertionError(f"unrecognized return type {t}")


# 将单个返回值 r 转换为其对应的 C++ 类型
def return_type(r: Return) -> CType:
    return returntype_type(r.type, mutable=r.is_write)


# 将 JIT 中的一个或多个返回值 rs 翻译为对应的 C++ 类型
def returns_type(rs: Sequence[Return]) -> CType:
    # 如果返回值列表 rs 长度为 0，返回 BaseCType(voidT)
    if len(rs) == 0:
        return BaseCType(voidT)
    # 如果返回值列表 rs 长度为 1，返回第一个返回值的类型
    elif len(rs) == 1:
        return return_type(rs[0])
    else:
        # 返回一个 TupleCType，包含每个返回值对应的类型
        return TupleCType([return_type(r) for r in rs])


# 返回一个 NativeFunction 的返回值名称序列
def return_names(f: NativeFunction, *, fallback_name: str = "result") -> Sequence[str]:
    # 初始化一个空列表 returns，用于存储返回值名称
    returns: list[str] = []
    # 遍历函数对象的返回值列表，并获取每个返回值及其索引
    for i, r in enumerate(f.func.returns):
        # 如果函数是 inplace 函数，则返回值参数隐式命名为 "self"
        # TODO: 考虑将此功能整合到数据模型中
        if f.func.name.name.inplace:
            # 断言只有一个返回值参数是合法的 inplace 函数
            assert i == 0, "illegal inplace function with multiple returns"
            name = "self"
        # 如果函数是输出函数，则返回值参数名为对应输出函数的名称
        elif f.func.is_out_fn():
            name = f.func.arguments.out[i].name
        # 如果返回值参数有显式命名...
        elif r.name:
            # 检查是否存在与其他参数命名冲突的情况
            name_conflict = any(
                r.name == a.name for a in f.func.schema_order_arguments()
            )
            # 如果有命名冲突且不是输出函数，则将返回值参数名添加后缀 "_return"
            if name_conflict and not f.func.is_out_fn():
                name = f"{r.name}_return"
            else:
                name = r.name
        # 如果没有显式命名且没有传入回退名称，则命名为输出结果，如果是多返回值，则命名为 result0, result1 等（从零开始索引）
        else:
            name = fallback_name if len(f.func.returns) == 1 else f"{fallback_name}{i}"
        # 将确定的名称添加到返回值列表中
        returns.append(name)
    # 返回包含所有返回值参数名称的列表
    return returns
JIT_TO_CPP_DEFAULT = {
    "False": "false",
    "True": "true",
    "None": "torch::executorch::nullopt",  # UGH this one is type directed
    "[]": "{}",
    "contiguous_format": "torch::executorch::MemoryFormat::Contiguous",
    "long": "torch::executorch::kLong",
}

# 将 JIT 默认值转换为表示默认值的 C++ 表达式
def default_expr(d: str, t: Type) -> str:
    if d == "None" and str(t) == "Tensor?":
        return "{}"
    if isinstance(t, BaseType) and t.name is BaseTy.str:
        # Schema allows single quotes but C++ needs double
        if len(d) >= 2 and d[0] == "'" and d[-1] == "'":
            s = ""
            i = 1
            while i + 1 < len(d):
                if d[i] != "\\":
                    if d[i] == '"':
                        s += '\\"'
                    else:
                        s += d[i]
                    i += 1
                else:
                    if d[i + 1] == "'":
                        s += "'"
                    else:
                        s += d[i : i + 2]
                    i += 2

            return f'"{s}"'

    if isinstance(t, OptionalType):
        if d == "None":
            return "torch::executor::nullopt"

        return default_expr(d, t.elem)

    if isinstance(t, ListType):
        if d.startswith("[") and d.endswith("]"):
            return "{" + d[1:-1] + "}"
        elif t.size is None:
            # NOTE: Sized lists can have scalar defaults
            raise ValueError(f"Expected a list default '[...]' but found: '{d}'")

    return JIT_TO_CPP_DEFAULT.get(d, d)


# 将参数转换为其对应的 C++ API 表单
def argument(
    a: Argument | TensorOptionsArguments | SelfArgument,
    *,
    cpp_no_default_args: set[str],
    method: bool,
    faithful: bool,
    has_tensor_options: bool,
) -> list[Binding]:
    def sub_argument(
        a: Argument | TensorOptionsArguments | SelfArgument,
    ) -> list[Binding]:
        return argument(
            a,
            cpp_no_default_args=cpp_no_default_args,
            method=method,
            faithful=faithful,
            has_tensor_options=has_tensor_options,
        )

    if isinstance(a, Argument):
        binds: ArgName
        if a.name == "memory_format" and has_tensor_options:
            binds = SpecialArgName.possibly_redundant_memory_format
        else:
            binds = a.name
        default: str | None = None
        if a.name not in cpp_no_default_args and a.default is not None:
            default = default_expr(a.default, a.type)
        return [
            Binding(
                nctype=argument_type(a, binds=binds),
                name=a.name,
                default=default,
                argument=a,
            )
        ]
    elif isinstance(a, TensorOptionsArguments):
        raise NotImplementedError("Need to implement type resolution for TensorOptions")
    elif isinstance(a, SelfArgument):
        # 如果 a 是 SelfArgument 类型的实例
        if method:
            # 如果有传入 method 参数
            # 调用者负责在上下文中安装隐式的 this 对象！
            return []
        else:
            # 如果没有传入 method 参数
            # 递归调用 sub_argument 函数处理 a 的 argument 属性
            return sub_argument(a.argument)
    else:
        # 如果 a 不是 SelfArgument 类型的实例，这里应该永远不会执行
        assert_never(a)
# 定义一个函数 arguments，接受多个参数并返回一个 Binding 对象列表
def arguments(
    arguments: Arguments,
    *,
    faithful: bool,
    method: bool,
    cpp_no_default_args: set[str],
) -> list[Binding]:
    # 初始化一个空列表 args，用于存储 Argument、TensorOptionsArguments 或 SelfArgument 对象
    args: list[Argument | TensorOptionsArguments | SelfArgument] = []
    # 如果 faithful 为 True，则按顺序添加 arguments 的 non_out 和 out 属性到 args 中
    if faithful:
        args.extend(arguments.non_out)
        args.extend(arguments.out)
    # 如果 faithful 不为 True，则按相反的顺序添加 arguments 的 out 和 non_out 属性到 args 中
    else:
        args.extend(arguments.out)
        args.extend(arguments.non_out)
    # 返回一个列表，列表中的每个元素是根据条件返回的结果
    return [
        # 如果 faithful 为 True，则调用 r.no_default() 方法，否则直接返回 r
        r.no_default() if faithful else r
        # 对 args 列表中的每个元素 a，调用 argument 函数处理，并扁平化结果为一个 Binding 对象列表
        for a in args
        for r in argument(
            a,
            # 传递 faithful、method、arguments.tensor_options 是否为 None 以及 cpp_no_default_args 参数给 argument 函数
            faithful=faithful,
            method=method,
            has_tensor_options=arguments.tensor_options is not None,
            cpp_no_default_args=cpp_no_default_args,
        )
    ]
```