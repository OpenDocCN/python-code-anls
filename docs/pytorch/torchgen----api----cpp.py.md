# `.\pytorch\torchgen\api\cpp.py`

```py
# 引入将来版本中的注释特性，使得该文件中可以使用类似于`annotations`的新特性
from __future__ import annotations

# 导入类型提示中的Sequence类型
from typing import Sequence

# 从torchgen中导入local模块
from torchgen import local
# 从torchgen.api.types中导入多个类型
from torchgen.api.types import (
    ArgName,                # 参数名称类型
    ArrayCType,             # 数组类型
    ArrayRefCType,          # 数组引用类型
    BaseCType,              # 基本类型
    BaseTypeToCppMapping,   # 基本类型到C++映射的映射关系
    Binding,                # 绑定类型
    boolT,                  # 布尔类型
    ConstRefCType,          # 常量引用类型
    CType,                  # C类型
    dimnameListT,           # 维度名称列表类型
    intArrayRefT,           # 整型数组引用类型
    iTensorListRefT,        # 张量列表引用类型
    ListCType,              # 列表类型
    longT,                  # 长整型类型
    MutRefCType,            # 可变引用类型
    NamedCType,             # 命名类型
    OptionalCType,          # 可选类型
    optionalIntArrayRefT,   # 可选整型数组引用类型
    optionalSymIntArrayRefT, # 可选符号整型数组引用类型
    scalarT,                # 标量类型
    SpecialArgName,         # 特殊参数名称类型
    symIntArrayRefT,        # 符号整型数组引用类型
    SymIntT,                # 符号整型类型
    tensorListT,            # 张量列表类型
    tensorOptionsT,         # 张量选项类型
    tensorT,                # 张量类型
    TupleCType,             # 元组类型
    VectorCType,            # 向量类型
    voidT,                  # 空类型
)

# 从torchgen.model中导入多个类型
from torchgen.model import (
    Argument,               # 参数类型
    Arguments,              # 参数列表类型
    BaseTy,                 # 基本类型枚举
    BaseType,               # 基本类型
    FunctionSchema,         # 函数模式类型
    ListType,               # 列表类型
    NativeFunction,         # 本地函数类型
    OptionalType,           # 可选类型
    Return,                 # 返回类型
    SelfArgument,           # 自身参数类型
    TensorOptionsArguments, # 张量选项参数类型
    Type,                   # 类型类型
)

# 从torchgen.utils中导入assert_never函数
from torchgen.utils import assert_never


# 该文件描述了将JIT模式翻译成公共C++ API的过程，当人们调用诸如at::add等函数时使用的内容。
#
# C++ API的显著特征：
#
#   - dtype、layout、device和pin_memory被收集到一个单一的C++类型TensorOptions中
#     （本地函数API也有这个，但是张量选项对于C++ API是最相关的；这使得调用关键字工厂函数更加愉快）
#
#   - 默认值设定在这里（事实上，调度程序完全不知道默认值的存在！）
#
# BTW：关于名称冲突的政策：我们尽量不让类型冲突，但是函数可以争夺相同的名称

# 根据给定的函数模式func，生成相应的名称
def name(
    func: FunctionSchema,
    *,
    faithful_name_for_out_overloads: bool = False, # 是否为输出重载保持忠实的名称
    symint_overload: bool = False,                 # 是否进行符号整数重载
) -> str:
    name = str(func.name.name)
    if symint_overload:
        name += "_symint"
    if func.is_out_fn():
        if faithful_name_for_out_overloads:
            name += "_outf"
        else:
            name += "_out"

    return name

# 将JIT模式中的“值类型”翻译为C++ API中的类型。值类型无论是作为参数类型还是返回类型，看起来都是一样的。
# 如果问题类型不是值类型，则返回None。
def valuetype_type(
    t: Type,
    *,
    binds: ArgName,                              # 参数名称绑定
    remove_non_owning_ref_types: bool = False,    # 是否移除非拥有引用类型
    symint: bool = False,                        # 是否为符号整数类型
) -> NamedCType | None:
    if isinstance(t, BaseType):
        if t.name == BaseTy.Tensor or t.name == BaseTy.Scalar:
            return None
        elif str(t) == "SymInt":
            if symint:
                return NamedCType(binds, BaseCType(SymIntT))
            else:
                return NamedCType(binds, BaseCType(longT))
        if remove_non_owning_ref_types:
            if t.name == BaseTy.str:
                raise AssertionError(
                    "string ref->value conversion: not implemented yet"
                )
        # 所有其他的BaseType当前都直接映射到BaseCppTypes。
        return NamedCType(binds, BaseCType(BaseTypeToCppMapping[t.name]))
    # 如果 t 是 OptionalType 类型
    elif isinstance(t, OptionalType):
        # 递归获取 t 元素的值类型
        elem = valuetype_type(t.elem, binds=binds, symint=symint)
        # 如果 elem 为 None，则返回 None
        if elem is None:
            return None
        # 构建一个 NamedCType 对象，表示可选类型的元素类型
        return NamedCType(binds, OptionalCType(elem.type))
    
    # 如果 t 是 ListType 类型
    elif isinstance(t, ListType):
        # 如果列表元素类型为布尔类型
        if str(t.elem) == "bool":
            # 断言列表大小不为空
            assert t.size is not None
            # 构建一个 NamedCType 对象，表示布尔类型数组
            return NamedCType(binds, ArrayCType(BaseCType(boolT), t.size))
        else:
            # 其他类型的列表返回 None
            return None
    
    # 如果 t 类型未被识别，则引发 AssertionError
    else:
        raise AssertionError(f"unrecognized type {repr(t)}")
# Translation of types occurring in JIT arguments to a C++ argument type.
# JIT参数中类型到C++参数类型的转换。

# If remove_non_owning_ref_types is set, we'll guarantee that the outputed CType is not a non-owning reference type.
# 如果设置了remove_non_owning_ref_types参数，则确保输出的CType不是非拥有引用类型。

# For example, we'll return std::vector<int> instead of IntArrayRef.
# 例如，我们将返回std::vector<int>而不是IntArrayRef。

# See Note [translation from C++ reference to value types]
# 参见注释 [translation from C++ reference to value types]
def argumenttype_type(
    t: Type,
    *,
    mutable: bool,
    binds: ArgName,
    remove_non_owning_ref_types: bool = False,
    symint: bool = False,
) -> NamedCType:
    # If it's a value type, do the value type translation
    # 如果是值类型，则进行值类型的转换
    r = valuetype_type(
        t,
        binds=binds,
        symint=symint,
        remove_non_owning_ref_types=remove_non_owning_ref_types,
    )
    if r is not None:
        return r

    if isinstance(t, BaseType):
        if t.name == BaseTy.Tensor:
            if mutable and not local.use_const_ref_for_mutable_tensors():
                return NamedCType(binds, MutRefCType(BaseCType(tensorT)))
            else:
                return NamedCType(binds, ConstRefCType(BaseCType(tensorT)))
        elif t.name == BaseTy.Scalar:
            return NamedCType(binds, ConstRefCType(BaseCType(scalarT)))
        else:
            raise AssertionError(f"base type should have been value type {t}")
    elif isinstance(t, OptionalType):
        if str(t.elem) == "Tensor":
            if mutable and not local.use_const_ref_for_mutable_tensors():
                return NamedCType(
                    binds, MutRefCType(BaseCType(tensorT))
                )  # TODO: fix this discrepancy
            else:
                return NamedCType(
                    binds, ConstRefCType(OptionalCType(BaseCType(tensorT)))
                )
        elif str(t.elem) == "Scalar":
            return NamedCType(binds, ConstRefCType(OptionalCType(BaseCType(scalarT))))
        elif isinstance(t.elem, ListType) and str(t.elem.elem) == "int":
            return NamedCType(binds, BaseCType(optionalIntArrayRefT))
        elif isinstance(t.elem, ListType) and str(t.elem.elem) == "SymInt":
            if symint:
                return NamedCType(binds, BaseCType(optionalSymIntArrayRefT))
            else:
                return NamedCType(binds, BaseCType(optionalIntArrayRefT))
        elem = argumenttype_type(t.elem, mutable=mutable, binds=binds, symint=symint)
        return NamedCType(binds, OptionalCType(elem.type))
    elif isinstance(t, ListType):
        # 如果类型是列表类型
        # TODO: remove these special cases, ArrayRef fallthrough works fine
        # 如果元素类型为整数
        if str(t.elem) == "int":
            # 如果需要移除非拥有引用类型
            if remove_non_owning_ref_types:
                # 返回命名的 C 类型，绑定到长整型的向量 C 类型
                return NamedCType(binds, VectorCType(BaseCType(longT)))
            else:
                # 返回命名的 C 类型，绑定到整数数组引用 C 类型
                return NamedCType(binds, BaseCType(intArrayRefT))
        # 如果元素类型为 SymInt
        if str(t.elem) == "SymInt":
            # 如果需要移除非拥有引用类型
            if remove_non_owning_ref_types:
                # 如果 symint 为真，则返回命名的 C 类型，绑定到 SymInt 类型的向量 C 类型；否则绑定到长整型的向量 C 类型
                if symint:
                    return NamedCType(binds, VectorCType(BaseCType(SymIntT)))
                else:
                    return NamedCType(binds, VectorCType(BaseCType(longT)))
            else:
                # 如果 symint 为真，则返回命名的 C 类型，绑定到 SymInt 数组引用 C 类型；否则绑定到整数数组引用 C 类型
                if symint:
                    return NamedCType(binds, BaseCType(symIntArrayRefT))
                else:
                    return NamedCType(binds, BaseCType(intArrayRefT))
        # 如果元素类型为 Tensor
        if str(t.elem) == "Tensor":
            # 如果本地使用 ilistref 代替 tensor 列表
            if local.use_ilistref_for_tensor_lists():
                # 返回命名的 C 类型，绑定到常量引用类型的基本 Tensor 列表引用 C 类型
                return NamedCType(binds, ConstRefCType(BaseCType(iTensorListRefT)))
            else:
                # 返回命名的 C 类型，绑定到基本 Tensor 列表 C 类型
                return NamedCType(binds, BaseCType(tensorListT))
        # 如果元素类型为 Scalar
        elif str(t.elem) == "Scalar":
            # 返回命名的 C 类型，绑定到基本 Scalar 数组引用 C 类型
            return NamedCType(binds, ArrayRefCType(BaseCType(scalarT)))
        # 如果元素类型为 Dimname
        elif str(t.elem) == "Dimname":
            # 返回命名的 C 类型，绑定到 Dimname 列表的基本 C 类型
            return NamedCType(binds, BaseCType(dimnameListT))
        # 如果元素类型为 Tensor?
        elif str(t.elem) == "Tensor?":
            # 返回命名的 C 类型，绑定到可选类型的基本 Tensor 的列表 C 类型的常量引用 C 类型
            return NamedCType(
                binds, ConstRefCType(ListCType(OptionalCType(BaseCType(tensorT))))
            )
        # 计算参数类型的类型，支持可变性，绑定
        elem = argumenttype_type(t.elem, mutable=mutable, binds=binds, symint=symint)
        # 返回命名的 C 类型，绑定到元素类型的数组引用 C 类型
        return NamedCType(binds, ArrayRefCType(elem.type))
    else:
        # 如果类型未被识别，则抛出断言错误
        raise AssertionError(f"unrecognized type {repr(t)}")
# 将 JIT 参数翻译为其对应的 C++ 类型
def argument_type(a: Argument, *, binds: ArgName, symint: bool = False) -> NamedCType:
    return argumenttype_type(a.type, mutable=a.is_write, symint=symint, binds=binds)


# 将 JIT 的单个（非多返回值）返回类型翻译为 C++ 类型
# 注意：returntype_type 返回一个 CType 而不是 NamedCType。
# 主要是因为返回类型和返回名称之间的不匹配。
# 例如，返回类型为 'void' 的函数没有返回名称，
# 返回类型为 'std::tuple' 的函数有多个返回名称。
def returntype_type(t: Type, *, mutable: bool, symint: bool = False) -> CType:
    # 占位符被忽略
    # 注意：对于返回类型，symint 总是被尊重的。因此这里的 symint 参数被忽略。
    r = valuetype_type(t, binds="__placeholder__", symint=True)
    if r is not None:
        return r.type

    if isinstance(t, BaseType):
        if t.name == BaseTy.Tensor:
            if mutable:
                if local.use_const_ref_for_mutable_tensors():
                    return ConstRefCType(BaseCType(tensorT))
                else:
                    return MutRefCType(BaseCType(tensorT))
            else:
                # 注意 [Tensor Copy Returns]
                # 当前我们使用 "Argument.is_write" 来决定
                # 张量返回类型是副本还是引用。
                # 如果这个有任何改变，请查看这个注释的其他位置！
                return BaseCType(tensorT)
        elif t.name == BaseTy.Scalar:
            return BaseCType(scalarT)
    elif isinstance(t, ListType):
        assert (
            not mutable
        ), "Native functions should never return a mutable tensor list. They should return void."
        elem = returntype_type(t.elem, mutable=False)
        assert t.size is None, f"fixed size list returns not supported: {t}"
        return VectorCType(elem)
    elif isinstance(t, OptionalType):
        elem = returntype_type(t.elem, mutable=mutable)
        if str(t.elem) == "Tensor":
            return OptionalCType(elem)

    raise AssertionError(f"unrecognized return type {t}")


# 将 JIT 的单个返回值翻译为其对应的 C++ 类型
def return_type(r: Return, *, symint: bool = False) -> CType:
    return returntype_type(r.type, mutable=r.is_write, symint=symint)


# 将 JIT 的完整（可能是多返回值）返回值翻译为其对应的 C++ 类型
def returns_type(rs: Sequence[Return], *, symint: bool = False) -> CType:
    if len(rs) == 0:
        return BaseCType(voidT)
    elif len(rs) == 1:
        return return_type(rs[0], symint=symint)
    else:
        return TupleCType([return_type(r, symint=symint) for r in rs])


# 返回 NativeFunction 的返回名称列表
def return_names(f: NativeFunction, *, fallback_name: str = "result") -> Sequence[str]:
    returns: list[str] = []
    # 遍历函数的返回值列表，同时获取索引 i 和返回值对象 r
    for i, r in enumerate(f.func.returns):
        # 如果函数是 inplace 函数，则返回参数隐式命名为 "self"
        # TODO: 考虑将此功能合并到数据模型中
        if f.func.name.name.inplace:
            # 如果有多个返回值，则不允许 inplace 函数操作
            assert i == 0, "illegal inplace function with multiple returns"
            name = "self"
        # 如果函数是输出函数，返回参数名称为对应输出函数的名称（r.name 将在后面的 field_name 中记录）
        elif f.func.is_out_fn():
            name = f.func.arguments.out[i].name
        # 如果返回参数有显式命名...
        elif r.name:
            # 检查是否存在名称冲突，如果不是输出函数则在冲突名称后添加 "_return"
            name_conflict = any(
                r.name == a.name for a in f.func.schema_order_arguments()
            )
            if name_conflict and not f.func.is_out_fn():
                name = f"{r.name}_return"
            else:
                name = r.name
        # 如果没有显式名称且没有传入后备名称，则命名输出结果为 fallback_name，对于多返回值则使用索引命名（从0开始）
        else:
            name = fallback_name if len(f.func.returns) == 1 else f"{fallback_name}{i}"
        # 将生成的名称添加到返回结果列表中
        returns.append(name)
    # 返回最终的返回结果列表
    return returns
# 默认的 JIT 到 C++ 表达式映射
JIT_TO_CPP_DEFAULT = {
    "False": "false",                   # 将 "False" 映射为 "false"
    "True": "true",                     # 将 "True" 映射为 "true"
    "None": "::std::nullopt",           # 将 "None" 映射为 "::std::nullopt"，表示空值
    "Mean": "at::Reduction::Mean",      # 将 "Mean" 映射为 "at::Reduction::Mean"
    "[]": "{}",                         # 将 "[]" 映射为 "{}"，表示空列表
    "contiguous_format": "MemoryFormat::Contiguous",  # 将 "contiguous_format" 映射为 "MemoryFormat::Contiguous"
    "long": "at::kLong",                # 将 "long" 映射为 "at::kLong"
}


# 将 JIT 默认值转换为表示默认值的 C++ 表达式
def default_expr(d: str, t: Type, *, symint: bool) -> str:
    if d == "None" and str(t) == "Tensor?":
        return "{}"  # 如果默认值为 "None"，且类型为 Tensor?，则返回空的初始化表达式 "{}"

    if isinstance(t, BaseType) and t.name is BaseTy.str:
        # Schema 允许单引号，但是在 C++ 中需要双引号
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
            return f'"{s}"'  # 将单引号字符串转换为双引号字符串

    if isinstance(t, OptionalType):
        if d == "None":
            return "::std::nullopt"  # 如果默认值为 "None"，则返回空的 optional 对象
        return default_expr(d, t.elem, symint=symint)  # 递归处理 OptionalType 类型的默认值

    if isinstance(t, ListType):
        if d.startswith("[") and d.endswith("]"):
            return "{" + d[1:-1] + "}"  # 如果默认值是以 "[" 开头，以 "]" 结尾，则转换为 C++ 的列表初始化
        elif symint and d.isdigit() and str(t.elem) == "SymInt":
            return f"c10::SymInt({d})"  # 如果 symint 为 True，且默认值是数字，且元素类型为 SymInt，则转换为 SymInt 类型的初始化
        elif t.size is None:
            # 注意：有大小限制的列表可以有标量默认值
            raise ValueError(f"Expected a list default '[...]' but found: '{d}'")

    return JIT_TO_CPP_DEFAULT.get(d, d)  # 返回 JIT_TO_CPP_DEFAULT 中的映射值，如果没有找到则返回默认值本身


# 将参数转换为其对应的 C++ API 形式
def argument(
    a: Argument | TensorOptionsArguments | SelfArgument,
    *,
    cpp_no_default_args: set[str],
    method: bool,
    faithful: bool,
    symint: bool = False,
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
            symint=symint,
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
            default = default_expr(a.default, a.type, symint=symint)
        return [
            Binding(
                nctype=argument_type(a, binds=binds, symint=symint),
                name=a.name,
                default=default,
                argument=a,
            )
        ]
    elif isinstance(a, TensorOptionsArguments):
        # 如果 a 是 TensorOptionsArguments 类型的对象
        if faithful:
            # 如果 faithful 为 True，则返回子参数的列表
            return (
                sub_argument(a.dtype)
                + sub_argument(a.layout)
                + sub_argument(a.device)
                + sub_argument(a.pin_memory)
            )
        else:
            # 否则，设置默认值为 None
            default = None
            # 强制执行 NativeFunction.__post_init__ 中的规定
            assert "options" not in cpp_no_default_args
            # 如果所有参数的默认值都为 "None"
            if all(x.default == "None" for x in a.all()):
                default = "{}"
            # 如果 dtype 的默认值为 "long"
            elif a.dtype.default == "long":
                default = "at::kLong"  # TODO: 这里是错误的
            # 返回一个 Binding 对象的列表
            return [
                Binding(
                    nctype=NamedCType("options", BaseCType(tensorOptionsT)),
                    name="options",
                    default=default,
                    argument=a,
                )
            ]
    elif isinstance(a, SelfArgument):
        # 如果 a 是 SelfArgument 类型的对象
        if method:
            # 如果 method 为 True，返回空列表
            # 调用方负责在上下文中安装隐式的 this！
            return []
        else:
            # 否则，返回 a.argument 的子参数列表
            return sub_argument(a.argument)
    else:
        # 如果以上条件都不满足，引发断言错误
        assert_never(a)
# 定义函数 arguments，接受一系列参数并返回绑定对象列表
def arguments(
    arguments: Arguments,  # 参数对象，包含函数参数信息
    *,  # 使用命名参数语法，以下参数为强制命名
    faithful: bool,  # 是否忠实处理参数的标志
    symint: bool = False,  # 是否进行符号整数处理，默认为 False
    method: bool,  # 方法标志，指示函数的处理方式
    cpp_no_default_args: set[str],  # 不使用默认参数的 C++ 参数集合
) -> list[Binding]:  # 函数返回绑定对象列表
    # 初始化参数列表 args
    args: list[Argument | TensorOptionsArguments | SelfArgument] = []
    
    # 根据 faithful 参数的值选择不同的处理顺序
    if faithful:
        # 如果 faithful 为真，先添加非输出参数，再添加输出参数
        args.extend(arguments.non_out)
        args.extend(arguments.out)
    else:
        # 如果 faithful 为假，先添加输出参数，再添加非输出参数
        args.extend(arguments.out)
        args.extend(arguments.non_out)
    
    # 构建返回值列表，根据 faithful 参数决定是否调用 no_default 方法
    return [
        r.no_default() if faithful else r  # 如果 faithful 为真，则调用 no_default 方法；否则直接返回 r
        for a in args  # 遍历参数列表 args
        for r in argument(  # 对每个参数调用 argument 函数生成结果列表
            a,
            faithful=faithful,
            symint=symint,
            method=method,
            has_tensor_options=arguments.tensor_options is not None,
            cpp_no_default_args=cpp_no_default_args,
        )
    ]
```