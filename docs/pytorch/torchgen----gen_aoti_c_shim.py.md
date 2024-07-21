# `.\pytorch\torchgen\gen_aoti_c_shim.py`

```
# 导入必要的模块和函数
from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Sequence

# 导入 TorchGen 库中的特定类型和函数签名类
from torchgen.api.types import DispatcherSignature
from torchgen.api.types.signatures import CppSignature, CppSignatureGroup
from torchgen.context import method_with_native_function
from torchgen.model import (
    Argument,
    BackendIndex,
    BaseTy,
    BaseType,
    DispatchKey,
    FunctionSchema,
    ListType,
    NativeFunction,
    NativeFunctionsGroup,
    OperatorName,
    OptionalType,
    Type,
)
# 导入 TorchGen 提供的实用工具函数
from torchgen.utils import mapMaybe

# 定义字典，将 TorchGen 中的基本类型映射到 C 类型
base_type_to_c_type = {
    BaseTy.Tensor: "AtenTensorHandle",
    BaseTy.bool: "int32_t",  # 用 int 类型传递布尔值
    BaseTy.int: "int64_t",
    BaseTy.SymInt: "int64_t",  # 由于生成器生成的代码看不到 SymInt，因此直接使用 int64_t
    BaseTy.Scalar: "double",  # 使用 double 类型传递整数和浮点数
    BaseTy.float: "double",  # TODO: 如何处理其他浮点数类型？
    BaseTy.str: "const char*",  # 字符串类型用 const char* 表示
    BaseTy.DeviceIndex: "int32_t",
    BaseTy.Layout: "int32_t",  # 将枚举表示为 int
    BaseTy.MemoryFormat: "int32_t",  # 将枚举表示为 int
    BaseTy.ScalarType: "int32_t",  # 将枚举表示为 int
    BaseTy.Generator: "AtenGeneratorHandle",
}

# 定义字典，将 TorchGen 中的基本类型映射到 ATen 库中的类型
base_type_to_aten_type = {
    BaseTy.Tensor: "at::Tensor",
    BaseTy.bool: "bool",
    BaseTy.int: "int64_t",
    BaseTy.SymInt: "c10::SymInt",
    BaseTy.Scalar: "c10::Scalar",
    BaseTy.float: "double",
    BaseTy.str: "c10::string_view",
    BaseTy.DeviceIndex: "c10::DeviceIndex",
    BaseTy.Layout: "c10::Layout",
    BaseTy.MemoryFormat: "c10::MemoryFormat",
    BaseTy.ScalarType: "c10::ScalarType",
    BaseTy.Generator: "at::Generator",
}

# 定义字典，将 TorchGen 中的基本类型映射到调用点表达式
base_type_to_callsite_expr = {
    BaseTy.Tensor: "*tensor_handle_to_tensor_pointer",  # 将 Tensor 转换为指针表达式
    BaseTy.bool: "",  # 布尔类型为空字符串，表示不需要特定的转换
    BaseTy.int: "",  # 整数类型为空字符串，表示不需要特定的转换
    BaseTy.SymInt: "",  # SymInt 类型为空字符串，表示不需要特定的转换
    BaseTy.Scalar: "",  # 标量类型为空字符串，表示不需要特定的转换
    BaseTy.float: "",  # 浮点数类型为空字符串，表示不需要特定的转换
    BaseTy.str: "",  # 字符串类型为空字符串，表示不需要特定的转换
    BaseTy.DeviceIndex: "static_cast<c10::DeviceIndex>",  # 将 DeviceIndex 转换为静态类型转换表达式
    BaseTy.Layout: "static_cast<c10::Layout>",  # 将 Layout 转换为静态类型转换表达式
    BaseTy.MemoryFormat: "static_cast<c10::MemoryFormat>",  # 将 MemoryFormat 转换为静态类型转换表达式
    BaseTy.ScalarType: "static_cast<c10::ScalarType>",  # 将 ScalarType 转换为静态类型转换表达式
    BaseTy.Generator: "*generator_handle_to_generator_pointer",  # 将 Generator 转换为指针表达式
}

# 定义函数，将参数类型和名称转换为 C 类型、声明中的名称、函数体中的表达式
def convert_arg_type_and_name(typ: Type, name: str) -> tuple[list[str], list[str], list[str], list[str]]:  # type: ignore[return]
    # 如果类型是 BaseType 的实例
    if isinstance(typ, BaseType):
        # 检查类型名是否在基本类型到 C 类型的映射中
        if typ.name in base_type_to_c_type:
            # 返回单个 C 类型、名称、ATen 类型和调用表达式的列表
            return (
                [base_type_to_c_type[typ.name]],
                [name],
                [base_type_to_aten_type[typ.name]],
                [
                    f"{base_type_to_callsite_expr[typ.name]}({name})"
                    if base_type_to_callsite_expr[typ.name]
                    else name
                ],
            )
        # 如果类型名是 BaseTy.Device
        elif typ.name == BaseTy.Device:
            # 返回两个 int32_t 类型、名称和名称加上索引后缀的列表、c10::Device 类型，以及构造设备对象的调用表达式列表
            return (
                ["int32_t", "int32_t"],
                [name, name + "_index_"],
                ["c10::Device"],
                [
                    f"c10::Device(static_cast<c10::DeviceType>({name}), static_cast<c10::DeviceIndex>({name}_index_))"
                ],
            )
        else:
            # 抛出未实现错误，提示添加对当前类型支持的 TODO
            raise NotImplementedError(f"TODO: add support for arg type {repr(typ)}")
    # 如果类型是 OptionalType 的实例
    elif isinstance(typ, OptionalType):
        # 转换参数类型和名称，获取 C 类型、名称、ATen 类型和调用表达式的列表
        c_types, names, aten_types, callsite_exprs = convert_arg_type_and_name(
            typ.elem, name
        )
        j = 0  # 用于名称索引
        new_aten_types = []
        new_callsite_exprs = []
        # 遍历 ATen 类型列表
        for aten_type in aten_types:
            # 使用指针表示可选类型
            c_types[j] = c_types[j] + "*"
            # 如果 ATen 类型以 "c10::ArrayRef<" 开头
            if aten_type.startswith("c10::ArrayRef<"):
                # 添加可选类型模板到新的 ATen 类型列表
                new_aten_types.append(f"::std::optional<{aten_type}>")
                # 提取基本类型，构造调用表达式用于指向可选列表的指针
                base_type = aten_type[len("c10::ArrayRef<") : -1]
                new_callsite_exprs.append(
                    f"pointer_to_optional_list<{base_type}>({names[j]}, {names[j+1]})"
                )
                j += 2
            # 如果 ATen 类型是 "c10::Device"
            elif aten_type == "c10::Device":
                # 添加可选 c10::Device 到新的 ATen 类型列表
                new_aten_types.append("::std::optional<c10::Device>")
                # 构造调用表达式用于指向可选设备对象的指针
                new_callsite_exprs.append(
                    f"pointer_to_optional_device({names[j]}, {names[j+1]})"
                )
                j += 2
            else:
                # 添加可选类型模板到新的 ATen 类型列表
                new_aten_types.append(f"::std::optional<{aten_type}>")
                # 构造调用表达式用于指向可选类型的指针
                new_callsite_exprs.append(
                    f"pointer_to_optional<{aten_type}>({names[j]})"
                )
                j += 1

        # 返回更新后的 C 类型、名称、ATen 类型和调用表达式的列表
        return (
            c_types,
            names,
            new_aten_types,
            new_callsite_exprs,
        )
    elif isinstance(typ, ListType):
        # 如果 typ 是 ListType 的实例，则需要将列表作为指针 + 长度显式传递
        c_types, names, aten_types, _ = convert_arg_type_and_name(typ.elem, name)
        # 断言列表只包含一个元素类型
        assert len(c_types) == 1, "ListType with unsupported element type " + repr(typ)

        # 列表内容不应被修改，因此将其元素类型声明为 const 指针
        c_types[0] = f"const {c_types[0]}*"
        # 添加用于存储列表长度的 int64_t 类型变量
        c_types.append("int64_t")
        # 更新变量名和长度变量名
        name = names[0]
        names.append(name + "_len_")

        atype = aten_types[0]
        callsite_exprs = []
        if atype == "bool":
            # 没有从 std::vector<bool> 到 c10::ArrayRef<bool> 的转换器
            # 使用 std::array<bool, N> 进行替代
            assert typ.size is not None
            callsite_exprs.append(f"pointer_to_list<{typ.size}>({name})")
        elif atype == "::std::optional<at::Tensor>":
            # 将 std::vector<::std::optional<at::Tensor>> 转换为 c10::List<::std::optional<at::Tensor>>
            callsite_exprs.append(
                f"c10::List<{atype}>(c10::ArrayRef<{atype}>(pointer_to_list<{atype}>({name}, {name}_len_)))"
            )
        else:
            # 对于其他类型，直接将列表转换为 c10::ArrayRef
            callsite_exprs.append(f"pointer_to_list<{atype}>({name}, {name}_len_)")

        # 更新返回类型列表，将每个类型转换为 c10::ArrayRef
        aten_types = [f"c10::ArrayRef<{t}>" for t in aten_types]
        # 返回转换后的结果：c_types, names, aten_types, callsite_exprs
        return (
            c_types,
            names,
            aten_types,
            callsite_exprs,
        )
# 根据输入的类型列表和名称列表生成类型和名称组合后的列表
def zip_type_and_name(types: list[str], names: list[str]) -> list[str]:
    return [typ + " " + name for typ, name in zip(types, names)]


# 生成参数声明和调用点表达式
def gen_arguments(flat_arguments: Sequence[Argument]) -> tuple[list[str], list[str]]:
    types = []           # 存储所有参数的类型
    new_names = []       # 存储所有参数的名称
    callsite_exprs = []  # 存储所有调用点表达式
    for arg in flat_arguments:
        # 转换参数类型和名称，并获取新的类型列表、名称列表、空列表和新的调用点表达式列表
        new_types, names, _, new_callsite_exprs = convert_arg_type_and_name(
            arg.type, arg.name
        )
        types.extend(new_types)             # 将新类型列表扩展到types中
        new_names.extend(names)             # 将新名称列表扩展到new_names中
        callsite_exprs.extend(new_callsite_exprs)  # 将新调用点表达式列表扩展到callsite_exprs中
    return zip_type_and_name(types, new_names), callsite_exprs  # 返回类型和名称的组合列表及所有调用点表达式列表


# 生成返回值的声明和调用点表达式
def gen_returns(schema: FunctionSchema) -> tuple[list[str], list[str]]:
    types = []  # 存储所有返回值的类型
    names = []  # 存储所有返回值的名称
    for idx, ret in enumerate(schema.returns):
        names.append(f"ret{idx}")  # 添加返回值名称，例如：ret0, ret1, ...
        if isinstance(ret.type, BaseType) and ret.type.name in base_type_to_c_type:
            types.append(base_type_to_c_type[ret.type.name] + "*")  # 如果返回类型是基本类型，则添加其对应的C类型指针
        else:
            raise NotImplementedError(
                f"TODO: add support for return type {repr(ret.type)}"  # 抛出未实现的异常，需要添加对该返回类型的支持
            )

    # 将返回值转换为C++中的表达式
    def convert_return(typ: BaseType, val: str) -> str:
        if typ.name == BaseTy.Tensor:
            return f"new_tensor_handle(std::move({val}));"  # 如果是Tensor类型返回值，生成对应的表达式
        elif typ.name == BaseTy.SymInt:
            return f"{val}.expect_int()"  # 如果是SymInt类型返回值，生成对应的表达式
        elif typ.name == BaseTy.Scalar:
            return f"{val}.toDouble()"  # 如果是Scalar类型返回值，生成对应的表达式
        else:
            return val  # 否则直接返回原始值

    ret_pointer_can_be_null = False  # 返回指针是否可以为空
    unambiguous_name = schema.name.unambiguous_name()  # 获取函数名称的唯一非模糊名称
    for name in [
        "_scaled_dot_product_flash_attention",
        "_scaled_dot_product_efficient_attention",
        "convolution_backward",
    ]:
        if name in unambiguous_name:
            ret_pointer_can_be_null = True  # 如果函数名称匹配特定名称，则返回指针可以为空
            break

    callsite_exprs: list[str] = []  # 存储所有调用点表达式
    for idx, ret in enumerate(schema.returns):
        tmp = "tmp_result" if len(names) == 1 else f"std::get<{idx}>(tmp_result)"  # 根据返回值数量生成临时变量名
        assert isinstance(ret.type, BaseType)
        rval = convert_return(ret.type, tmp)  # 将返回值转换为对应的表达式
        if ret_pointer_can_be_null:
            callsite_exprs.append(f"if ({names[idx]}) {{ *{names[idx]} = {rval}; }}")  # 如果返回指针可以为空，则生成对应的赋值语句
        else:
            callsite_exprs.append(f"*{names[idx]} = {rval};")  # 否则生成普通的赋值语句

    return zip_type_and_name(types, names), callsite_exprs  # 返回类型和名称的组合列表及所有调用点表达式列表


# gen.py首先生成头文件，然后生成源文件，因此在此处缓存声明和定义的结果以避免重复工作
declaration_definition_cache: dict[tuple[str, str, str], tuple[str, str]] = {}


# 生成声明和定义
def gen_declaration_and_definition(
    schema: FunctionSchema, device: str, backend_call: str
) -> tuple[str, str]:
    func_name = schema.name.unambiguous_name()  # 获取函数名称的唯一非模糊名称

    global declaration_definition_cache
    # 如果函数名、设备和后端调用方式在声明-定义缓存中已经存在，则直接返回缓存中的结果
    if (func_name, device, backend_call) in declaration_definition_cache:
        return declaration_definition_cache[(func_name, device, backend_call)]

    # 如果模式表明这是一个输出函数
    if schema.is_out_fn():
        # out_variant 中的参数在前面，可以忽略返回值，因为 C 语言的 shim 函数只返回 AOTITorchError
        args, callsite_exprs = gen_arguments(
            [*schema.arguments.out, *schema.arguments.flat_non_out]
        )
        # 返回值赋值语句的列表，对于输出函数，这里先初始化为空列表
        ret_assignments: list[str] = []
    else:
        # 生成全部参数及其调用表达式
        args, callsite_exprs = gen_arguments(schema.arguments.flat_all)
        # 对于不是 inplace 操作的情况，生成返回值的声明和赋值语句
        ret_declarations, ret_assignments = (
            ([], []) if schema.name.name.inplace else gen_returns(schema)
        )
        # 将返回值的声明加入到参数列表中
        args.extend(ret_declarations)

    # 生成函数声明，形如 "AOTITorchError aoti_torch_{device}_{func_name}(arg1, arg2, ...)"
    declaration = f"AOTITorchError aoti_torch_{device}_{func_name}({', '.join(args)})"

    # 如果有返回值赋值语句，则生成临时结果的声明
    tmp_result = "auto tmp_result = " if ret_assignments else ""
    # 如果有返回值赋值语句，则生成这些语句的字符串形式，每个语句占一行
    ret_assignments_str = "\n" + "\n".join(ret_assignments) if ret_assignments else ""

    # 定义函数体的字符串，包含临时结果声明和返回值赋值语句
    definition = f"""
{
    # 定义一个 Python 字典，用于存储声明和定义的元组，以便缓存
    declaration_definition_cache[(func_name, device, backend_call)] = (
        declaration,
        definition,
    )
    # 返回生成的声明和定义
    return declaration, definition
}
"""

def gen_static_dispatch_backend_call_signature(
    sig: CppSignature | DispatcherSignature,
    f: NativeFunction,
) -> CppSignature:
    # 从函数的 schema 生成调度器签名
    sig = DispatcherSignature.from_schema(f.func)
    # 从本地函数生成 C++ 签名组
    cpp_sigs = CppSignatureGroup.from_native_function(
        f, method=False, fallback_binding=False
    )
    # 如果存在符号整数（symint）并且函数包含 symint，则选择符号整数签名
    if sig.symint and f.func.has_symint():
        cpp_sig = cpp_sigs.symint_signature
    else:
        cpp_sig = cpp_sigs.signature
    # 断言 C++ 签名不为 None
    assert cpp_sig is not None
    # 返回 C++ 签名
    return cpp_sig


def gen_static_dispatch_backend_call(
    f: NativeFunction,
    backend_index: BackendIndex,
) -> str:
    # 从函数的 schema 生成调度器签名
    sig = DispatcherSignature.from_schema(f.func)
    # 生成静态分发的后端调用的 C++ 签名
    cpp_sig = gen_static_dispatch_backend_call_signature(sig, f)
    # 构造静态分发后端调用的字符串表示
    return f"at::{backend_index.dispatch_key.lower()}::{cpp_sig.name()}"


def get_backend_index_for_aoti(
    func: NativeFunction,
    func_group_mapping: dict[OperatorName, NativeFunctionsGroup],
    dispatch_key: DispatchKey,
    backend_indices: dict[DispatchKey, BackendIndex],
) -> BackendIndex | None:
    # 初始化后端索引为 None
    backend_index = None
    # 如果给定调度键的后端索引包含当前函数的内核或者结构化委托的内核，则设置为该后端索引
    if backend_indices[dispatch_key].has_kernel(func) or (
        func.structured_delegate is not None
        and func.structured_delegate in func_group_mapping
        and backend_indices[dispatch_key].has_kernel(
            func_group_mapping[func.structured_delegate]
        )
    ):
        backend_index = backend_indices[dispatch_key]
    # 否则，如果 CompositeExplicitAutograd 调度键的后端索引包含当前函数的内核，则设置为该后端索引
    elif backend_indices[DispatchKey.CompositeExplicitAutograd].has_kernel(func):
        # 需要为 CompositeExplicitAutograd 内核创建 C 语言的包装器
        backend_index = backend_indices[DispatchKey.CompositeExplicitAutograd]
    # 否则，如果 CompositeExplicitAutogradNonFunctional 调度键的后端索引包含当前函数的内核，则设置为该后端索引
    elif backend_indices[DispatchKey.CompositeExplicitAutogradNonFunctional].has_kernel(
        func
    ):
        # 需要为 CompositeExplicitAutogradNonFunctional 内核创建 C 语言的包装器
        backend_index = backend_indices[
            DispatchKey.CompositeExplicitAutogradNonFunctional
        ]
    # 否则，如果 CompositeImplicitAutograd 调度键的后端索引包含当前函数的内核，则设置为该后端索引
    elif backend_indices[DispatchKey.CompositeImplicitAutograd].has_kernel(func):
        backend_index = backend_indices[DispatchKey.CompositeImplicitAutograd]

    # 返回确定的后端索引或者 None
    return backend_index


def get_header_for_aoti(
    func: NativeFunction,
    func_group_mapping: dict[OperatorName, NativeFunctionsGroup],
    dispatch_key: DispatchKey,
    backend_indices: dict[DispatchKey, BackendIndex],
) -> str | None:
    # 获取适用于 AOTI 的后端索引
    backend_index = get_backend_index_for_aoti(
        func, func_group_mapping, dispatch_key, backend_indices
    )
    # 如果 backend_index 为 None，则返回 None
    # 否则，根据 func 的 root_name 和 backend_index 的 dispatch_key 构造一个特定的头文件路径字符串
    return (
        None
        if backend_index is None
        else f"#include <ATen/ops/{func.root_name}_{backend_index.dispatch_key.lower()}_dispatch.h>"
    )
# 获取回退操作名称，根据给定的 NativeFunction 对象生成操作名称的字符串
def get_fallback_op_name(func: NativeFunction) -> str:
    return (
        f"{func.namespace}.{func.func.name.name}.{func.func.name.overload_name}"
        if func.func.name.overload_name  # 如果存在重载名称，则使用重载名称
        else f"{func.namespace}.{func.func.name.name}.default"  # 否则使用默认名称
    )


# 生成 C 语言的桥接代码
def gen_c_shim(
    func: NativeFunction,  # NativeFunction 对象，表示需要生成桥接代码的函数
    func_group_mapping: dict[OperatorName, NativeFunctionsGroup],  # 操作符名称到函数组的映射
    dispatch_key: DispatchKey,  # 分发键，用于选择后端实现
    backend_indices: dict[DispatchKey, BackendIndex],  # 分发键到后端索引的映射
    header: bool,  # True 表示生成头文件声明，False 表示生成源文件定义
) -> str | None:
    # 获取适合给定函数的后端索引
    backend_index = get_backend_index_for_aoti(
        func, func_group_mapping, dispatch_key, backend_indices
    )
    if backend_index is None:  # 如果未找到适合的后端索引，则返回 None
        return None

    schema = func.func  # 函数的 schema 信息
    device = dispatch_key.lower()  # 分发键的小写形式
    # 生成静态分发后端调用的代码
    backend_call = gen_static_dispatch_backend_call(
        func,
        backend_index,
    )

    try:
        if header:
            # 如果是生成头文件声明，则生成函数声明和定义
            declaration, _ = gen_declaration_and_definition(
                schema, device, backend_call
            )
            return f"AOTI_TORCH_EXPORT {declaration};"  # 返回导出声明
        else:
            # 否则生成函数定义
            _, definition = gen_declaration_and_definition(schema, device, backend_call)
            return definition  # 返回函数定义

    except NotImplementedError:
        return None  # 如果生成失败，则返回 None


# 装饰了 NativeFunction 的方法，用于生成桥接代码的类
@dataclass(frozen=True)
class ShimGenerator:
    func_group_mapping: dict[OperatorName, NativeFunctionsGroup]  # 操作符名称到函数组的映射
    dispatch_key: DispatchKey  # 分发键，用于选择后端实现
    backend_indices: dict[DispatchKey, BackendIndex]  # 分发键到后端索引的映射
    header: bool  # True 表示生成头文件声明，False 表示生成源文件定义

    @method_with_native_function  # 装饰的方法，处理 NativeFunction 对象
    def __call__(
        self,
        func: NativeFunction,  # 需要生成桥接代码的 NativeFunction 对象
    ) -> str | None:
        # 调用 gen_c_shim 生成桥接代码
        result = gen_c_shim(
            func,
            self.func_group_mapping,
            self.dispatch_key,
            self.backend_indices,
            self.header,
        )
        return result  # 返回生成的桥接代码字符串或 None


# 生成针对 AOTI 的 C 语言桥接代码
def gen_aoti_c_shim(
    native_functions: Sequence[NativeFunction],  # 需要生成桥接代码的 NativeFunction 序列
    func_group_mapping: dict[OperatorName, NativeFunctionsGroup],  # 操作符名称到函数组的映射
    dispatch_key: DispatchKey,  # 分发键，用于选择后端实现
    backend_indices: dict[DispatchKey, BackendIndex],  # 分发键到后端索引的映射
    header: bool,  # True 表示生成头文件声明，False 表示生成源文件定义
    includes: str = "",  # 需要包含的额外头文件
) -> str:
    # 使用 mapMaybe 函数，通过 ShimGenerator 对象生成桥接代码，并且过滤掉返回值为 None 的结果
    body = "\n".join(
        list(
            mapMaybe(
                ShimGenerator(
                    func_group_mapping, dispatch_key, backend_indices, header
                ),
                native_functions,
            )
        )
    )
    device = dispatch_key.lower()  # 分发键的小写形式

    warning = """
// WARNING: THIS FILE IS AUTOGENERATED BY torchgen. DO NOT MODIFY BY HAND.
// See https://github.com/pytorch/pytorch/blob/7e86a7c0155295539996e0cf422883571126073e/torchgen/gen.py#L2424-L2436 for details"""

    if header:
        # 如果生成头文件声明，则返回包含头文件保护和声明的字符串
        return f"""
{warning}

#pragma once

#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#ifdef __cplusplus
extern "C" {{
#endif

{body}

#ifdef __cplusplus
}} // extern "C"
#endif
"""
    else:
        # 否则生成包含特定设备头文件和工具函数的源文件字符串
        return f"""
{warning}

#include <torch/csrc/inductor/aoti_torch/generated/c_shim_{device}.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/{str(dispatch_key)}Functions.h>
#include <ATen/CompositeExplicitAutogradFunctions.h>
#include <ATen/CompositeExplicitAutogradNonFunctionalFunctions.h>
#include <ATen/CompositeImplicitAutogradFunctions.h>
#else
{includes}
#endif

using namespace torch::aot_inductor;

{body}"""



// 包含特定的 ATen 模块中的函数声明，使用给定的调度键来确定模块
#include <ATen/{str(dispatch_key)}Functions.h>

// 包含组合显式自动求导函数的声明
#include <ATen/CompositeExplicitAutogradFunctions.h>

// 包含组合显式自动求导非功能函数的声明
#include <ATen/CompositeExplicitAutogradNonFunctionalFunctions.h>

// 包含组合隐式自动求导函数的声明
#include <ATen/CompositeImplicitAutogradFunctions.h>

// 如果不满足上述条件，则包含指定的其他头文件（由 includes 变量决定）
#else
{includes}
#endif

// 使用 torch::aot_inductor 命名空间
using namespace torch::aot_inductor;

// 定义函数体的起始点
{body}"""
```