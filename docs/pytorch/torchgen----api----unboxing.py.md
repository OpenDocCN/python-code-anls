# `.\pytorch\torchgen\api\unboxing.py`

```
# 引入未来版本兼容性，允许在当前版本使用未来的注解语法
from __future__ import annotations

# 从torchgen.api模块中导入cpp子模块
from torchgen.api import cpp
# 从torchgen.api.types模块中导入Binding, CppSignatureGroup, CType类
from torchgen.api.types import Binding, CppSignatureGroup, CType
# 从torchgen.model模块中导入Argument, BaseTy, BaseType, ListType, NativeFunction, OptionalType, Type类
from torchgen.model import (
    Argument,
    BaseTy,
    BaseType,
    ListType,
    NativeFunction,
    OptionalType,
    Type,
)
# 定义一个换行和制表符组成的连接器，用于格式化输出
connector = "\n\t"


# 返回一个 NativeFunction 的函数名，用于生成 C++ 代码
def name(f: NativeFunction) -> str:
    return f.func.name.unambiguous_name()


# 将 NativeFunction 中的所有参数转换为对应的 C++ 代码
def convert_arguments(f: NativeFunction) -> tuple[list[Binding], list[str]]:
    # 获取 NativeFunction 的参数列表，需要设置 method=False 来排除 self 参数
    args = (
        CppSignatureGroup.from_native_function(f, method=False)
        .most_faithful_signature()
        .arguments()
    )
    # 生成参数转换的代码列表
    code_list = [
        f"c10::IValue {args[i].name} = std::move(peek(stack, {i}, {len(args)}));"
        for i in range(len(args))
    ] + [""]  # 添加一个空字符串作为分隔
    binding_list = []
    for arg in args:
        # 检查参数类型是否为 Argument
        if not isinstance(arg.argument, Argument):
            raise Exception(
                f"Unexpected argument type, expecting `Argument` but got {arg}"
            )
        argument: Argument = arg.argument
        # 调用 argumenttype_ivalue_convert 函数进行参数类型转换
        unboxed_name, _, code, decl = argumenttype_ivalue_convert(
            argument.type,
            argument.name,
            mutable=argument.is_write,
        )
        code_list.extend(decl)  # 添加声明相关的代码
        code_list.extend(code)   # 添加转换代码
        binding_list.append(arg.with_name(unboxed_name))  # 将转换后的参数添加到 binding_list 中
    return binding_list, code_list


# 根据参数类型、名称和可变性，生成解包参数所需的 C++ 代码和 Binding 对象
def argumenttype_ivalue_convert(
    t: Type, arg_name: str, *, mutable: bool = False
) -> tuple[str, CType, list[str], list[str]]:
    # 函数体待实现，根据参数类型生成相应的代码和声明
    # Unboxing is for mobile, which doesn't care about SymInts
    # 根据参数类型生成对应的 C++ 类型，针对移动设备，不考虑 SymInts
    ctype = cpp.argumenttype_type(
        t=t, mutable=mutable, binds=arg_name, symint=False
    ).type

    if isinstance(t, BaseType):
        # 如果参数类型是基本类型，生成基本类型的代码和声明
        out_name = f"{arg_name}_base"
        code, decl = _gen_code_base_type(
            arg_name=arg_name, out_name=out_name, ctype=ctype
        )
    elif isinstance(t, OptionalType):
        # 如果参数类型是可选类型，生成可选类型的代码和声明
        out_name = f"{arg_name}_opt_out"
        code, decl = _gen_code_optional_type(
            arg_name=arg_name,
            out_name=out_name,
            t=t,
            ctype=ctype,
        )
    elif isinstance(t, ListType):
        # 如果参数类型是列表类型，生成列表类型的代码和声明
        out_name = f"{arg_name}_list_out"
        code, decl = _gen_code_list_type(
            arg_name=arg_name,
            out_name=out_name,
            t=t,
            ctype=ctype,
        )
    else:
        # 如果无法处理给定的参数类型，抛出异常
        raise Exception(f"Cannot handle type {t}. arg_name: {arg_name}")  # noqa: TRY002
    # 返回生成的输出名称、C++ 类型、代码和声明
    return out_name, ctype, code, decl
# 生成基本类型的代码。将参数转换为指定的 C++ 类型，并赋给输出变量。
def _gen_code_base_type(
    arg_name: str, out_name: str, ctype: CType
) -> tuple[list[str], list[str]]:
    return [
        f"{ctype.cpp_type(strip_ref=True)} {out_name} = {arg_name}.to<{ctype.cpp_type(strip_ref=True)}>();"
    ], []


# 生成可选类型的代码。根据输入参数是否有值，进行相应的类型转换和赋值操作。
def _gen_code_optional_type(
    arg_name: str, out_name: str, t: OptionalType, ctype: CType
) -> tuple[list[str], list[str]]:
    in_name = f"{arg_name}_opt_in"
    res_name, _, res_code, decl = argumenttype_ivalue_convert(t.elem, in_name)
    return (
        f"""
auto {arg_name}_opt = {arg_name}.toOptional<c10::IValue>();
{ctype.cpp_type(strip_ref=True)} {out_name};
if ({arg_name}_opt.has_value()) {{
    const c10::IValue {in_name} = {arg_name}_opt.value();
    {connector.join(res_code)}
    {out_name} = {ctype.cpp_type(strip_ref=True)}({res_name});
}} else {{
    {out_name} = {ctype.cpp_type(strip_ref=True)}();
}}
        """.split(
            "\n"
        ),
        decl,
    )


# 生成列表类型的代码。根据列表元素的不同类型，执行不同的处理逻辑。
def _gen_code_list_type(
    arg_name: str, out_name: str, t: ListType, ctype: CType
) -> tuple[list[str], list[str]]:
    in_name = f"{arg_name}_list_in"
    elem_name = f"{arg_name}_elem"
    code = [f"const c10::List<c10::IValue> {in_name} = {arg_name}.toList();"]
    res_name, res_ctype, res_code, decl = argumenttype_ivalue_convert(t.elem, elem_name)
    # 处理带有大小的列表类型，例如 bool[4]
    if isinstance(t.elem, BaseType) and t.elem.name == BaseTy.bool and t.size:
        code.extend(
            f"""
{ctype.cpp_type(strip_ref=True)} {out_name} = as_array<{res_ctype.cpp_type(strip_ref=True)}, {t.size}>({in_name});
            """.split(
                "\n"
            )
        )
    # 对于可选元素的列表，使用 c10::List，例如 Tensor?[]
    elif isinstance(t.elem, OptionalType):
        code.extend(
            f"""
{ctype.cpp_type(strip_ref=True)} {out_name};
for (c10::IValue {elem_name}: {in_name}) {{
    {connector.join(res_code)}
    {out_name}.push_back({res_name});
}}
            """.split(
                "\n"
            )
        )
    else:
        # 默认使用 ArrayRef 处理列表
        vec_name = arg_name + "_vec"
        # 将 vector 实例化放在作用域外，确保 ArrayRef 有有效的数据
        decl.append(f"std::vector<{res_ctype.cpp_type(strip_ref=True)}> {vec_name};")
        code.extend(
            f"""
for (c10::IValue {elem_name}: {in_name}) {{
    {connector.join(res_code)}
    {vec_name}.push_back({res_name});
}}
{ctype.cpp_type(strip_ref=True)} {out_name}({vec_name});
            """.split(
                "\n"
            )
        )
    return code, decl
```