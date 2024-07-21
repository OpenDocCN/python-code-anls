# `.\pytorch\torchgen\executorch\api\unboxing.py`

```
# 从未来导入 annotations 模块，确保代码在 Python 3.7 及以上版本中兼容类型注解
from __future__ import annotations

# 导入 dataclass 模块，用于创建不可变的数据类；导入 Callable 和 Sequence 类型用于类型注解；导入 TYPE_CHECKING 用于类型检查
from dataclasses import dataclass
from typing import Callable, Sequence, TYPE_CHECKING

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 从 torchgen.api.types 模块导入 Binding、CType 和 NamedCType 类型
    from torchgen.api.types import Binding, CType, NamedCType

# 定义一个换行连接符字符串
connector = "\n\t"


# 定义一个函数 name，接受 NativeFunction 类型参数 f，返回其函数名的唯一名称
def name(f: NativeFunction) -> str:
    return f.func.name.unambiguous_name()


# 定义一个装饰器类 Unboxing，用于生成正确的解包代码
@dataclass(frozen=True)
class Unboxing:
    """
    Takes a sequence of Bindings and unbox EValues to these Bindings. Return generated code that performs correct unboxing.
    A sample generated code:
    // aten::mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    void mul_out(EValue** stack) {
        EValue& self = *stack[0];
        EValue& other = *stack[1];
        EValue& out = *stack[2];
        const torch::executor::Tensor & self_base = self.to<torch::executor::Tensor>();
        const torch::executor::Tensor & other_base = other.to<torch::executor::Tensor>();
        torch::executor::Tensor & out_base = out.to<torch::executor::Tensor>();

        EXECUTORCH_SCOPE_PROF("native_call_mul.out");
        torch::executor::mul_outf(self_base, other_base, out_base);
    }
    """

    # 定义一个可调用对象 argument_type_gen，用于将 JIT 参数转换为其 C++ 类型的 NamedCType
    argument_type_gen: Callable[..., NamedCType]

    # 定义一个方法 convert_arguments，将 NativeFunction 中的所有参数转换为 C++ 代码
    def convert_arguments(
        self, args: Sequence[Binding]
    ) -> tuple[list[Binding], list[str]]:
        # 生成代码列表，为每个参数创建 EValue 引用
        code_list = [f"EValue& {args[i].name} = *stack[{i}];" for i in range(len(args))]
        binding_list = []
        for arg in args:
            # 检查参数是否为 Argument 类型
            if not isinstance(arg.argument, Argument):
                raise Exception(
                    f"Unexpected argument type, expecting `Argument` but got {arg}"
                )
            argument: Argument = arg.argument
            # 调用 argumenttype_evalue_convert 方法，将参数类型转换为 EValue 类型
            unboxed_name, _, code, decl = self.argumenttype_evalue_convert(
                argument.type, argument.name, mutable=argument.is_write
            )
            code_list.extend(decl)  # 扩展声明代码到代码列表
            code_list.extend(code)   # 扩展转换代码到代码列表
            binding_list.append(arg.with_name(unboxed_name))  # 将转换后的参数添加到绑定列表
        return binding_list, code_list

    # 定义方法 argumenttype_evalue_convert，将类型 t 和参数名 arg_name 转换为 EValue 类型
    def argumenttype_evalue_convert(
        self, t: Type, arg_name: str, *, mutable: bool = False
    ):
    ) -> tuple[str, CType, list[str], list[str]]:
        """
        Takes in the type, name and mutability corresponding to an argument, and generates a tuple of:
        (1) the C++ code necessary to unbox the argument
        (2) A Binding corresponding to the newly created unboxed variable, including variable name and its CType
        :param t: a `Type` of an argument
        :param arg_name: argument name
        :param mutable: boolean for whether this argument type is mutable
        :return: unboxed result
        """
        # Generate CType for the argument using argument_type_gen method
        ctype = self.argument_type_gen(t, mutable=mutable, binds=arg_name).type

        if isinstance(t, BaseType):
            # For BaseType arguments, generate code to unbox the argument
            out_name = f"{arg_name}_base"
            code, decl = self._gen_code_base_type(
                arg_name=arg_name, out_name=out_name, ctype=ctype
            )
        elif isinstance(t, OptionalType):
            # For OptionalType arguments, generate code to handle optional types
            out_name = f"{arg_name}_opt_out"
            code, decl = self._gen_code_optional_type(
                arg_name=arg_name, out_name=out_name, t=t, ctype=ctype
            )
        elif isinstance(t, ListType):
            # For ListType arguments, generate code to handle lists
            out_name = f"{arg_name}_list_out"
            code, decl = self._gen_code_list_type(
                arg_name=arg_name, out_name=out_name, t=t, ctype=ctype
            )
        else:
            # Raise an exception if the argument type is not handled
            raise Exception(
                f"Cannot handle type {t}. arg_name: {arg_name}"
            )

        # Return the generated variables and code as a tuple
        return out_name, ctype, code, decl

    def _gen_code_base_type(
        self, arg_name: str, out_name: str, ctype: CType
    ) -> tuple[list[str], list[str]]:
        """
        Generate C++ code to unbox a BaseType argument.
        :param arg_name: argument name
        :param out_name: name of the output variable
        :param ctype: CType of the argument
        :return: tuple containing generated code and declarations
        """
        # Generate code to unbox a BaseType argument
        return [
            f"{ctype.cpp_type()} {out_name} = {arg_name}.to<{ctype.cpp_type(strip_ref=True)}>();"
        ], []

    def _gen_code_optional_type(
        self, arg_name: str, out_name: str, t: OptionalType, ctype: CType
    ) -> tuple[list[str], list[str]]:
        """
        Generate C++ code to handle an OptionalType argument.
        :param arg_name: argument name
        :param out_name: name of the output variable
        :param t: OptionalType instance
        :param ctype: CType of the argument
        :return: tuple containing generated code and declarations
        """
        # Generate names for input and result variables
        in_name = f"{arg_name}_opt_in"
        # Convert the optional type argument and generate corresponding code
        res_name, base_type, res_code, decl = self.argumenttype_evalue_convert(
            t.elem, in_name
        )
        return (
            # Return formatted C++ code for handling OptionalType
            f"""
    {ctype.cpp_type(strip_ref=True)} {out_name} = {arg_name}.toOptional<{base_type.cpp_type(strip_ref=True)}>();
            """.split(
                "\n"
            ),
            decl,
        )

    def _gen_code_list_type(
        self, arg_name: str, out_name: str, t: ListType, ctype: CType
    ) -> tuple[list[str], list[str]]:
        """
        Generate C++ code to handle a ListType argument.
        :param arg_name: argument name
        :param out_name: name of the output variable
        :param t: ListType instance
        :param ctype: CType of the argument
        :return: tuple containing generated code and declarations
        """
        # Generate names for input, element, and result variables
        in_name = f"{arg_name}_list_in"
        elem_name = f"{arg_name}_elem"
        code = []
        # Convert the list type argument and generate corresponding code
        res_name, res_ctype, res_code, decl = self.argumenttype_evalue_convert(
            t.elem, elem_name
        )

        if isinstance(t.elem, BaseType) and t.elem.name == BaseTy.Tensor:
            # Extend code with handling for BaseType elements of type Tensor
            code.extend(
                f"""
    {ctype.cpp_type(strip_ref=True)} {out_name} = {arg_name}.toTensorList();
                """.split(
                    "\n"
                )

这段代码根据输入参数 `arg_name` 将其转换为 `Tensor` 对象的列表，并将结果赋给变量 `out_name`。


        elif isinstance(t.elem, BaseType) and (
            t.elem.name == BaseTy.int or t.elem.name == BaseTy.SymInt
        ):
            code.extend(
                f"""
    {ctype.cpp_type(strip_ref=True)} {out_name} = {arg_name}.toIntList();
                """.split(
                    "\n"
                )
            )

在输入参数 `arg_name` 是整数 (`int`) 或符号整数 (`SymInt`) 类型时，将其转换为整数列表，并将结果赋给变量 `out_name`。


        elif isinstance(t.elem, BaseType) and t.elem.name == BaseTy.float:
            code.extend(
                f"""
    {ctype.cpp_type(strip_ref=True)} {out_name} = {arg_name}.toDoubleList();
                """.split(
                    "\n"
                )
            )

在输入参数 `arg_name` 是浮点数 (`float`) 类型时，将其转换为双精度浮点数列表，并将结果赋给变量 `out_name`。


        elif isinstance(t.elem, BaseType) and t.elem.name == BaseTy.bool:
            # handle list type with size, e.g., bool[4]
            code.extend(
                f"""
    {ctype.cpp_type(strip_ref=True)} {out_name} = {arg_name}.toBoolList();
                """.split(
                    "\n"
                )
            )

在输入参数 `arg_name` 是布尔值 (`bool`) 类型时，将其转换为布尔值列表，并将结果赋给变量 `out_name`。


        # pytorch codegen:
        # we have to use c10::List for optional element. e.g., Tensor?[] -> c10::List<::std::optional<at::Tensor>>
        elif (
            isinstance(t.elem, OptionalType)
            and isinstance(t.elem.elem, BaseType)
            and t.elem.elem.name == BaseTy.Tensor
        ):
            code.extend(
                f"""
    {ctype.cpp_type(strip_ref=True)} {out_name} = {arg_name}.toTensorOptionalList();
                """.split(
                    "\n"
                )
            )

对于 PyTorch 代码生成，处理可选元素时，例如 `Tensor?[]`，使用 `c10::List<::std::optional<at::Tensor>>` 类型，将输入参数 `arg_name` 转换为包含可选张量的列表，并将结果赋给变量 `out_name`。
#ifdef USE_ATEN_LIB
// 如果定义了 USE_ATEN_LIB 宏，则使用 ATen 库提供的方法将输入的张量列表转换为可选张量列表
auto {in_name} = {arg_name}.toListOptionalTensor();
// 声明一个 C++ 的 List，存储可选的 ATen 张量
c10::List<::std::optional<at::Tensor>> {out_name};
// 遍历输入的张量列表，将每个张量转换为可选张量并添加到输出列表中
for (auto {elem_name}: {in_name}) {{
    {out_name}.push_back({elem_name});
}}
#else
// 如果未定义 USE_ATEN_LIB 宏，则使用 Torch 库提供的方法将输入的张量列表转换为可选张量列表
torch::executor::ArrayRef<torch::executor::optional<torch::executor::Tensor>> {out_name} = {arg_name}.toListOptionalTensor();
#endif
```