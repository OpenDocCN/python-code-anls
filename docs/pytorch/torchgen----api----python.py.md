# `.\pytorch\torchgen\api\python.py`

```
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from torchgen.api import cpp  # 导入cpp模块，用于与C++交互
from torchgen.api.types import Binding, CppSignature, CppSignatureGroup  # 导入Binding、CppSignature和CppSignatureGroup类型
from torchgen.gen import pythonify_default  # 导入pythonify_default函数，用于生成Python代码
from torchgen.model import (  # 导入多个数据模型类，包括Argument、BaseTy等

    Argument,
    BaseTy,
    BaseType,
    FunctionSchema,
    ListType,
    NativeFunction,
    OptionalType,
    Return,
    Type,
    Variant,
)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                           Data Models
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
# [Notes] python binding codegen
#
# The Python binding codegen produces code that takes the input list of
# PyObjects, finds the matching ATen C++ function using PythonArgParser,
# converts the PyObjects into C++ types and calls the ATen C++ function:
#
# +--------+  parsing   +------------------------+  binding   +-----------------------+
# | PyObjs | ---------> | PythonArgParser Output | ---------> | Cpp Function Dispatch |
# +--------+            +------------------------+            +-----------------------+
#
# The following examples demonstrate the data models the Python binding
# codegen needs to deal with and the tasks it needs to accomplish. It
# helps understand the purpose of the new data types we introduced below.
#
#  - Function Schema (source of truth)
#
#      aten::empty.names(int[] size, *, Dimname[]? names,
#                        ScalarType? dtype=None, Layout? layout=None,
#                        Device? device=None, bool? pin_memory=None,
#                        MemoryFormat? memory_format=None) -> Tensor
#
#  - Python Signature
#
#    It's used to generate input schema string for PythonArgParser.
#    Note: TensorOptions fields are reordered and the additional
#    'requires_grad' field is added:
#
#      empty(IntArrayRef size, *, DimnameList? names,
#            MemoryFormat? memory_format=None, ScalarType dtype=None,
#            Layout layout=torch.strided, Device device=None,
#            bool pin_memory=False, bool requires_grad=False)
#
#  - C++ Signature
#
#    It's used to generate C++ lambda formals & dispatch call.
#    Note: the scattered TensorOptions fields are packed into 'options'.
#
#      auto dispatch_empty =
#          [](IntArrayRef size, std::optional<DimnameList> names,
#             const TensorOptions & options,
#             std::optional<MemoryFormat> memory_format) -> Tensor {
#          pybind11::gil_scoped_release no_gil;
#          return torch::empty(size, names, options, memory_format);
#      };
#
#  - Binding between Python Arguments and C++ Arguments
#
#    Given a set of Python Arguments in scope, we need produce the
#    binding expressions that translate the Python API into C++ API:
#
#            Python Args               Cpp Args       Binding Exprs
#     -----------------------------------------------------------------
//         0: size                      size           '_r.intlist(0)'
// 获取参数中的 size，对应 Python 中的 _r.intlist(0)
//         1: names                     names          'names' [special init]
// 获取参数中的 names，对应 Python 中的 names [special init]
//         2: memory_format -------+
// 获取参数中的 memory_format，对应 Python 中的 _r.memoryformatOptional(2)
//         3: dtype         -----+-|--> options        'options' [special packing]
// 获取参数中的 dtype，对应 Python 中的 _r.scalartype(3)，与其他选项一起打包成 TensorOptions 对象
//         4: layout            /  |
// 获取参数中的 layout，对应 Python 中的 _r.layoutOptional(4)
//         5: device           /   +--> memory_format  '_r.memoryformatOptional(2)'
// 获取参数中的 device，对应 Python 中的 _r.device(5)，与其他选项一起打包成 TensorOptions 对象
//         6: pin_memory      /
// 获取参数中的 pin_memory，对应 Python 中的 _r.toBool(6)，与其他选项一起打包成 TensorOptions 对象
//         7: requires_grad -+
// 获取参数中的 requires_grad，对应 Python 中的 _r.toBool(7)，与其他选项一起打包成 TensorOptions 对象
//
//    So the full dispatch expression would look like:
//
//      dispatch_empty(_r.intlist(0), names, options,
//                     _r.memoryformatOptional(2))
//
//    Where does 'names' come from? It involves special local init:
//
//      auto __names = _r.toDimnameListOptional(1);
//      std::optional<DimnameList> names =
//          __names ? std::make_optional(DimnameList(__names.value()))
//                  : std::nullopt;
//
//    Where does 'options' come from? It involves special local init
//    for TensorOptions. Note that Python side has the additional
//    'requires_grad' field:
//
//      const auto options = TensorOptions()
//          .dtype(_r.scalartype(3))
//          .device(_r.device(5))
//          .layout(_r.layoutOptional(4))
//          .requires_grad(_r.toBool(7))
//          .pinned_memory(_r.toBool(6));
//
//    In some other cases one Python Argument can map to multiple C++
//    Arguments. For example:
//
//     aten::max.names_dim(Tensor self, Dimname dim, bool keepdim=False)
//       -> (Tensor values, Tensor indices)
//
//            Python Args               Cpp Args          Binding Exprs
//     ---------------------------------------------------------------------
//                               +----> max               'out[0]'
//                              /-----> max_values        'out[1]
//         0: input            /        self              '_r.tensor(0)'
//         1: dim             /         dim               '_r.dimname(1)'
//         2: keepdim        /          keepdim           '_r.toBool(2)'
//         3: out      -----+           [local init] out  '_r.tensorlist_n<2>(3)'
//
//    As demonstrated above, the binding can involve reordering,
//    packing, unpacking and special local inits.
//
//
//  Let's look at a concrete example:
//
//      static PythonArgParser parser({
//        "abs(Tensor input, *, Tensor out=None)",
//        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//         ^
//         +--- Python Schema, represented by PythonSignature and PythonArgument
//
//      }, /*traceable=*/true);
//
//      ParsedArgs<2> parsed_args;
//      auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
//
//      ...
//
//      if (_r.isNone(1)) {
//          ~~~~~~~~~~~~  <--- Scattered PythonArgParser output (arg name = 'out')
//                             represented by PythonArgParserOutputExpr
//
//        // aten::abs(Tensor self) -> Tensor
//        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//         ^
//         +--- NativeFunction schema, base version
//
#        auto dispatch_abs = [](const Tensor & self) -> Tensor {
#                            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                             ^
#                             +--- dispatch_lambda_args / dispatch_lambda_return_str
#                                  generated from NativeFunction / CppSignature
#                                  (deprecated PythonSignature is special)
#                                  arguments are represented by DispatchLambdaArgument
#
#          pybind11::gil_scoped_release no_gil;
#          // Release GIL (Global Interpreter Lock) to allow concurrent execution
#          return self.abs();
#                 ~~~~~~~~~~~  <--- cpp_dispatch_target / cpp_dispatch_exprs
#                                   generated from NativeFunction / CppSignature
#        };
#        // Wrap the lambda function `dispatch_abs` and apply it to the first argument
#        // retrieved from Python as a tensor
#        return wrap(dispatch_abs(_r.tensor(0)));
#                                 ~~~~~~~~~~~~~
#                                  ^
#                                  +--- dispatch_lambda_exprs
#                                       binding PythonArgParserOutputExpr (python args)
#                                       and DispatchLambdaArgument (c++ args)
#
#      } else {
#        // aten::abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
#        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         ^
#         +--- NativeFunction schema, out-variant
#
#        auto dispatch_abs_out = [](Tensor out, const Tensor & self) -> Tensor {
#          pybind11::gil_scoped_release no_gil;
#          // Release GIL (Global Interpreter Lock) to allow concurrent execution
#          return at::abs_out(out, self);
#        };
#        // Wrap the lambda function `dispatch_abs_out` and apply it to the tensors
#        // retrieved from Python as arguments
#        return wrap(dispatch_abs_out(_r.tensor(1), _r.tensor(0)));
#      }
#
#
# [Notes] python interface codegen
# The python dataclasses below are used used to generate both python binding code
# and pyi type hint signatures.
# In theory these two should look very similar, but there are number of differences
# in how pyi signatures vs. python_arg_parser signatures are generated.
# These differences have been encapsulated in signature_str() vs. signature_str_pyi()
# to display the full signatures, and argument_str() vs argument_str_pyi() to display arguments.
# For examples, only pyi signatures include return types.
    # 生成描述参数类型的字符串，排除 const 和 & 符号
    def argument_str(self, *, method: bool = False, symint: bool = True) -> str:
        # 获取参数类型的字符串表示，根据需要替换特定字符串
        type_str = (
            argument_type_str(self.type, symint=symint)
            .replace("const ", "")
            .replace(" &", "")
        )

        # 获取参数的名称
        name = self.name

        # 在方法绑定外部分，将名称 "self" 替换为 "input"
        # [旧的代码生成] TODO: 是否移除？代码生成中不重命名，仅用于解析字符串
        if name == "self" and type_str in ["Tensor", "Number"] and not method:
            name = "input"

        # 如果参数有默认值
        if self.default is not None:
            # 将默认值映射到相应的 Python 表示
            default = {
                "nullptr": "None",
                "c10::nullopt": "None",
                "::std::nullopt": "None",
                "std::nullopt": "None",
                "{}": "None",
            }.get(self.default, self.default)
            # 返回参数类型、名称和默认值的字符串表示
            return f"{type_str} {name}={default}"
        else:
            # 返回参数类型和名称的字符串表示
            return f"{type_str} {name}"

    # 为 Python 类型提示生成描述参数的字符串
    def argument_str_pyi(
        self, *, method: bool = False, deprecated: bool = False
    ) -> str:
        # 获取参数类型的 Python 类型提示字符串表示
        type_str = argument_type_str_pyi(self.type)

        # 获取参数的名称
        name = self.name

        # 在方法绑定外部分，将名称 "self" 替换为 "input"
        # [旧的代码生成] TODO: 是否移除？代码生成中不重命名，仅用于解析字符串
        if name == "self" and type_str == "Tensor" and not method and not deprecated:
            name = "input"

        # 如果参数名为 "from"，在 Python 中需要添加下划线
        if name == "from":
            name += "_"

        # 对于参数名为 "out"，且类型为 "Tensor"，且未被弃用，将类型更改为可选的 Tensor
        if name == "out" and type_str == "Tensor" and not deprecated:
            type_str = "Optional[" + type_str + "]"

        # 对于被弃用的签名，如果输出参数没有默认值，标记为不使用默认值
        treat_as_no_default = (
            deprecated
            and isinstance(self, PythonOutArgument)
            and self.default == "None"
        )

        # 如果参数有默认值且不被视为不使用默认值的情况
        if self.default is not None and not treat_as_no_default:
            # 特定情况下处理默认值的表示
            if (
                isinstance(self.type, ListType)
                and self.type.elem == BaseType(BaseTy.int)
                and self.default.startswith("{")
                and self.default.endswith("}")
            ):
                default = "(" + self.default[1:-1] + ")"
            else:
                # 将默认值映射到相应的 Python 表示
                default = {
                    "nullptr": "None",
                    "c10::nullopt": "None",
                    "::std::nullopt": "None",
                    "std::nullopt": "None",
                    "{}": "None",
                    "MemoryFormat::Contiguous": "contiguous_format",
                    "QScheme::PER_TENSOR_AFFINE": "per_tensor_affine",
                }.get(self.default, self.default)
            # 返回参数名称、类型和默认值的 Python 类型提示字符串表示
            return f"{name}: {type_str} = {default}"
        else:
            # 返回参数名称和类型的 Python 类型提示字符串表示
            return f"{name}: {type_str}"
@dataclass(frozen=True)
class PythonOutArgument(PythonArgument):
    # PythonOutArgument 类继承自 PythonArgument 类，用于表示 Python 签名中的输出参数。
    # 在 Python 签名中，多个输出字段被打包到一个 'out' 参数中。
    # 在绑定到 C++ 时，首先绑定到本地的 'out' 变量：
    #   'auto out = _r.tensorlist_n<2>(2);',
    # 然后作为散布的 C++ 输出参数 'out[0]', 'out[1]' 等等。
    # TODO: 或许不需要为 Python 签名保留散布的 'out' 字段？
    outputs: tuple[PythonArgument, ...]

    @staticmethod
    def from_outputs(outputs: tuple[PythonArgument, ...]) -> PythonOutArgument | None:
        # 如果 outputs 为空，则返回 None
        if not outputs:
            return None

        # 获取输出参数的数量
        size = len(outputs)
        # 如果只有一个输出参数
        if size == 1:
            # 返回一个 PythonOutArgument 对象，表示单个输出参数
            return PythonOutArgument(
                name=outputs[0].name,
                type=outputs[0].type,
                default="None",
                default_init=None,
                outputs=outputs,
            )
        # 如果有多个输出参数
        elif size > 1:
            # 检查是否有任何一个输出参数不是张量类型
            if any(not a.type.is_tensor_like() for a in outputs):
                raise RuntimeError(f"Unsupported output type: {outputs}")
            # 返回一个 PythonOutArgument 对象，表示多个输出参数，用 'out' 表示
            return PythonOutArgument(
                name="out",
                # TODO: 这里应该是 OptionalType[ListType[...]] 吗？因为它默认为 None。
                type=ListType(BaseType(BaseTy.Tensor), size),
                default="None",
                default_init=None,
                outputs=outputs,
            )
        # 如果出现意外的 PythonOutArgument 大小，则引发断言错误
        raise AssertionError(r"Unexpected PythonOutArgument size")


@dataclass(frozen=True)
class PythonSignature:
    # 操作符的基本名称，不包含 inplace/outplace 后缀。
    name: str

    # 位置参数
    # TODO: 为 'self' 创建一个专用的 SelfArgument 类型？
    input_args: tuple[PythonArgument, ...]

    # 关键字参数，不包括 'out' 参数和属于 TensorOptions 的散布关键字参数
    input_kwargs: tuple[PythonArgument, ...]

    output_args: PythonOutArgument | None

    # 返回类型，仅在 pyi 中使用
    returns: PythonReturns

    # 属于 TensorOptions 的散布关键字参数
    # 在绑定到 C++ 时，它们被打包成一个 TensorOptions 对象 'options'。
    # 可能的情况是 C++ 签名不接受 TensorOptions 对象（例如 out 变种），在这种情况下，
    # 它们将被用作散布字段，而不被打包到 'options' 中。
    # TODO: 或许创建一个 PythonTensorOptionsArgument？
    tensor_options_args: tuple[PythonArgument, ...]

    # 方法还是函数签名？
    method: bool

    @property
    def deprecated(self) -> bool:
        return False

    def arguments(
        self, *, skip_outputs: bool = False, skip_tensor_options: bool = False
        # 返回方法的所有参数，可以选择跳过输出参数和 TensorOptions 参数
    # 返回由输入参数和关键字参数组成的元组，可能包含输出参数和张量选项参数，根据条件决定是否包含输出参数和是否跳过张量选项参数
    -> tuple[PythonArgument | PythonOutArgument, ...]:
        # 初始化一个空列表来存放结果
        result: list[PythonArgument | PythonOutArgument] = []
        # 将对象的输入参数添加到结果列表中
        result.extend(self.input_args)
        # 将对象的关键字参数添加到结果列表中
        result.extend(self.input_kwargs)
        # 如果存在输出参数且不跳过输出参数，则将输出参数添加到结果列表中
        if self.output_args is not None and not skip_outputs:
            result.append(self.output_args)
        # 如果不跳过张量选项参数，则将张量选项参数添加到结果列表中
        if not skip_tensor_options:
            result.extend(self.tensor_options_args)
        # 将结果列表转换为元组并返回
        return tuple(result)

    # 返回对象参数的总数，包括输入参数和关键字参数
    def arguments_count(self) -> int:
        # 调用 arguments 方法获取参数列表的长度并返回
        return len(self.arguments())

    # 返回输出参数的索引位置，即输入参数和关键字参数的总数
    def output_idx(self) -> int:
        # 返回输入参数和关键字参数的总数，这个值也是输出参数的索引位置
        return len(self.input_args) + len(self.input_kwargs)

    # 生成用于解析参数的Python函数签名字符串，根据指定的条件来决定是否跳过输出参数和符号整数化
    def signature_str(self, *, skip_outputs: bool = False, symint: bool = True) -> str:
        # 获取参数列表，根据条件来决定是否包含输出参数
        args = self.arguments(skip_outputs=skip_outputs)
        # 初始化一个空列表来存放参数字符串形式
        schema_formals: list[str] = [
            # 调用 PythonArgument 对象的 argument_str 方法来获取参数的字符串表示形式
            a.argument_str(method=self.method, symint=symint) for a in args
        ]
        # 计算输入参数的数量
        positional_argc = len(self.input_args)
        # 如果参数列表的长度大于输入参数的数量，则在适当的位置插入一个 "*" 表示多余的位置参数
        if len(schema_formals) > positional_argc:
            schema_formals.insert(positional_argc, "*")

        # 返回格式化后的函数签名字符串
        return f'{self.name}({", ".join(schema_formals)})'

    # 生成用于 mypy-valid 类型签名的Python函数签名字符串，根据指定的条件来决定是否跳过输出参数
    def signature_str_pyi(self, *, skip_outputs: bool = False) -> str:
        # 获取参数列表，根据条件来决定是否包含输出参数
        args = self.arguments(skip_outputs=skip_outputs)
        # 初始化一个空列表来存放参数字符串形式
        schema_formals: list[str] = [
            # 调用 PythonArgument 对象的 argument_str_pyi 方法来获取参数的字符串表示形式
            a.argument_str_pyi(method=self.method) for a in args
        ]
        # 计算输入参数的数量
        positional_argc = len(self.input_args)
        # 如果参数列表的长度大于输入参数的数量，则在适当的位置插入一个 "*" 表示多余的位置参数
        if len(schema_formals) > positional_argc:
            schema_formals.insert(positional_argc, "*")

        # 获取返回值的字符串形式
        returns_str = returns_str_pyi(self)
        # 如果是方法，则在参数列表的开头加入 "self"，表示方法的self参数
        if self.method:
            schema_formals.insert(0, "self")
        # 返回格式化后的函数签名字符串
        return f'def {self.name}({", ".join(schema_formals)}) -> {returns_str}: ...'
    # 仅用于 pyi 的 vararg 签名
    args = self.arguments(skip_outputs=skip_outputs)
    # 创建一个存储参数签名字符串的列表，用于生成 pyi 签名
    schema_formals: list[str] = [
        a.argument_str_pyi(method=self.method) for a in args
    ]
    # vararg 只适用于 pyi 签名。并非所有签名都会生成 vararg 变体
    num_args = self.arguments_count()
    num_positionalargs = len(self.input_args)

    have_vararg_version = False
    # 如果有参数存在
    if num_args > 0:
        vararg_type = args[0].type
        # 如果第一个参数是列表类型，并且元素类型为 "int" 或 "SymInt"，且仅有一个位置参数
        if (
            isinstance(vararg_type, ListType)
            and str(vararg_type.elem) in ["int", "SymInt"]
            and num_positionalargs == 1
        ):
            have_vararg_version = True

    # 如果没有 vararg 版本，则返回 None
    if not have_vararg_version:
        return None

    # 下面是 vararg 与常规 pyi 签名之间的主要区别
    # vararg 签名不包含星号
    assert isinstance(vararg_type, ListType)
    schema_formals[0] = (
        "*" + args[0].name + ": " + argument_type_str_pyi(vararg_type.elem)
    )

    # 生成 pyi 中的返回类型字符串
    returns_str = returns_str_pyi(self)
    # 对于方法，pyi 还包括 self 参数（没有类型和默认值）
    if self.method:
        schema_formals.insert(0, "self")
    # 返回生成的 pyi 签名字符串
    return f'def {self.name}({", ".join(schema_formals)}) -> {returns_str}: ...'
# The deprecated python signature involves some special logic, so create a
# dedicated data model to store these extra properties.
@dataclass(frozen=True)
class PythonSignatureDeprecated(PythonSignature):
    # Schema for the deprecated function
    deprecated_schema: FunctionSchema

    # The deprecated signature might miss some arguments that the corresponding
    # C++ signature expects. We need store the constant default values to pass in.
    # For example:
    #   [deprecate signature]: addmm(Scalar beta, Tensor self, Tensor mat1, Tensor mat2)
    #   [func schema]: aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
    #   [func call]: self.addmm(mat1, mat2, beta, 1)
    # We store ['self', 'mat1', 'mat2', 'beta', '1'] in this case.
    deprecated_args_exprs: tuple[str, ...]

    @property
    def deprecated(self) -> bool:
        return True

    def signature_str(self, *, skip_outputs: bool = False, symint: bool = True) -> str:
        return (
            PythonSignature.signature_str(
                self, skip_outputs=skip_outputs, symint=symint
            )
            + "|deprecated"
        )

    def signature_str_pyi(self, *, skip_outputs: bool = False) -> str:
        # Generate Python type hint string for the deprecated signature
        args = self.arguments(skip_outputs=skip_outputs)
        schema_formals: list[str] = [
            a.argument_str_pyi(method=self.method, deprecated=True) for a in args
        ]
        positional_argc = len(self.input_args)
        if len(schema_formals) > positional_argc:
            schema_formals.insert(positional_argc, "*")

        returns_str = returns_str_pyi(self)
        return f'def {self.name}({", ".join(schema_formals)}) -> {returns_str}: ...'

    def signature_str_pyi_vararg(self, *, skip_outputs: bool = False) -> str | None:
        # Return None as codegen doesn't include vararg variants for deprecated signatures
        return None


# This struct is used to hold the PythonSignature and its corresponding
# NativeFunction BEFORE grouping base and out-variant functions.
# Why not store NativeFunction in PythonSignature or construct PythonSignature
# from NativeFunction? Because they are not 1-1 mapped.
# One native function could have both deprecated and non-deprecated python
# signatures - NativeFunction doesn't contain information to construct the
# deprecated python signature.
# One python signature is used to handle both the base and the out-variant
# function - see 'PythonSignatureGroup'.
@dataclass(frozen=True)
class PythonSignatureNativeFunctionPair:
    signature: PythonSignature
    function: NativeFunction


# We merge pairs of functions with signatures that are equivalent mod
# output arguments, and use a single entry in the python_arg_parser sig
# list for both (output arguments become optional).
@dataclass(frozen=True)
class PythonSignatureGroup:
    # The signature used for Python argument parsing. The outplace signature
    # is preferred if exists, because it can be used to parse inputs for both
    # 表示Python函数签名的对象，包括非就地操作和基本版本（输出被省略）。
    signature: PythonSignature

    # 常规的ATen声明（例如conv2d）
    base: NativeFunction

    # 就地操作的变体（例如conv2d_out），可以为None
    outplace: NativeFunction | None

    @classmethod
    def from_pairs(
        cls,
        functional: PythonSignatureNativeFunctionPair,
        out: PythonSignatureNativeFunctionPair | None,
    ) -> PythonSignatureGroup:
        # 如果没有提供就地操作的签名，返回只包含功能操作的PythonSignatureGroup
        if out is None:
            return PythonSignatureGroup(
                signature=functional.signature,
                base=functional.function,
                outplace=None,
            )

        # 首选带有可选的out=...参数的签名，因为它是可以用来解析输入的超集，适用于基本操作和就地操作。
        signature_kwargs = out.signature.__dict__.copy()

        # C++中的就地操作重载不包含TensorOptions参数，因此从功能操作中获取这些参数。
        signature_kwargs[
            "tensor_options_args"
        ] = functional.signature.tensor_options_args

        # 返回一个PythonSignatureGroup对象，包括基本操作、就地操作以及更新后的签名参数。
        return PythonSignatureGroup(
            signature=type(out.signature)(**signature_kwargs),
            base=functional.function,
            outplace=out.function,
        )
# C++ function dispatch is wrapped in a lambda function. The lambda function
# has almost the same signature as the C++ function, only with some small
# variants - see details below.
# This data model is used to represent arguments of the lambda function
# signature.
@dataclass(frozen=True)
class DispatchLambdaArgument:
    name: str               # 参数名
    type_str: str           # 参数类型字符串
    is_out_arg: bool        # 是否为输出参数


# To pass PyObjects arguments to C++ function (via the lambda wrapper),
# we need first convert PyObjects into simple C++ objects. This work
# is done by PythonArgParser.
# This data model is used to represent the output of PythonArgParser.
# It has 1-1 mapping with PythonArgument in PythonSignature.
@dataclass(frozen=True)
class PythonArgParserOutputExpr:
    name: str                # 参数名
    expr: str                # 右手边表达式，引用 PythonArgParser 的输出
    index: int               # 索引位置，用于特殊情况下的表达式生成
    argument: PythonArgument # 映射的 Python 参数对象

    @property
    def is_none_expr(self) -> str:
        return f"_r.isNone({self.index})"  # 返回一个表示是否为 None 的表达式


# To pass PythonArgParser output to the lambda wrapper, we need bind
# PythonArgParserOutputExpr to DispatchLambdaArgument.
# They are not always 1-1 mapped, e.g. scattered TensorOptions fields
# need be packed into a TensorOptions object, which is the argument
# that the lambda function wrapper takes.
@dataclass(frozen=True)
class DispatchLambdaArgumentExprs:
    exprs: Sequence[str]     # 提供给 lambda 参数绑定的表达式列表
    inits: Sequence[str]     # 特殊的本地初始化语句列表，引入新变量供上述表达式引用


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                          Helper Functions
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def _cpp_signature(f: NativeFunction, *, method: bool = False) -> CppSignature:
    return CppSignatureGroup.from_native_function(f, method=method).signature
    # 根据 NativeFunction 获取 C++ 签名，并返回 CppSignature 对象


def has_tensor_options(f: NativeFunction) -> bool:
    return f.func.arguments.tensor_options is not None
    # 检查 NativeFunction 是否具有 tensor_options 参数


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                          Python Signature
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


# 'simple_type' was introduced by the old codegen, which is slightly
# different from the python schema type, e.g.: doesn't have '?' suffix
# for optional Tensor/TensorList; doesn't have '[size]' suffix for list type.
def argument_type_str(
    t: Type, *, simple_type: bool = False, symint: bool = True
) -> str:
    # 根据给定的 Type 返回参数的类型字符串表示，可选参数控制简化类型和符号整数类型
    # 如果类型 t 是 BaseType 的实例
    if isinstance(t, BaseType):
        # 检查 t 的名称是否为 BaseTy.Tensor
        if t.name == BaseTy.Tensor:
            # 返回字符串 "Tensor"
            return "Tensor"
        # 检查 t 的名称是否为 BaseTy.int
        elif t.name == BaseTy.int:
            # 返回字符串 "int64_t"
            return "int64_t"
        # 检查 t 的名称是否为 BaseTy.float
        elif t.name == BaseTy.float:
            # 返回字符串 "double"
            return "double"
        # 检查 t 的名称是否为 BaseTy.str
        elif t.name == BaseTy.str:
            # 返回字符串 "c10::string_view"
            return "c10::string_view"
        # 检查 t 的名称是否在指定的列表中
        elif t.name in [
            BaseTy.bool,
            BaseTy.QScheme,
            BaseTy.Scalar,
            BaseTy.ScalarType,
            BaseTy.Generator,
            BaseTy.Storage,
            BaseTy.Layout,
            BaseTy.Device,
            BaseTy.DeviceIndex,
            BaseTy.MemoryFormat,
            BaseTy.Dimname,
            BaseTy.Stream,
            BaseTy.ConstQuantizerPtr,
            BaseTy.SymInt,
        ]:
            # 这些 Python schema 类型名称与它们的函数 schema 名称相匹配
            return t.name.name

    # 如果类型 t 是 OptionalType 的实例
    elif isinstance(t, OptionalType):
        # 如果 t 的元素类型为 "Tensor"
        if str(t.elem) == "Tensor":
            # 返回字符串 "Tensor?"
            return "Tensor?"
        # 递归调用 argument_type_str 函数，处理 t 的元素类型
        elem = argument_type_str(t.elem, simple_type=simple_type, symint=symint)
        # 返回包含 "?" 的元素类型字符串
        return f"{elem}?"
    
    # 如果类型 t 是 ListType 的实例
    elif isinstance(t, ListType):
        # 如果不是简单类型，确定数组的大小
        size = t.size if not simple_type else None
        # 检查元素类型是否为 "bool"
        if str(t.elem) == "bool":
            # 断言数组的大小不为 None
            assert t.size is not None
            # 返回带有布尔值数组的类型字符串
            return f"::std::array<bool,{t.size}>"
        # 检查元素类型是否为 "int"
        elif str(t.elem) == "int":
            # 返回带有 IntArrayRef 类型及可选的数组大小的字符串
            return f"IntArrayRef[{size}]" if size is not None else "IntArrayRef"
        # 检查元素类型是否为 "SymInt"
        elif str(t.elem) == "SymInt":
            # 如果 symint 为 True，返回带有 SymIntArrayRef 类型及可选的数组大小的字符串
            if symint:
                return (
                    f"SymIntArrayRef[{size}]" if size is not None else "SymIntArrayRef"
                )
            # 如果 symint 不为 True，返回带有 IntArrayRef 类型及可选的数组大小的字符串
            else:
                return f"IntArrayRef[{size}]" if size is not None else "IntArrayRef"
        # 检查元素类型是否为 "Tensor"
        elif str(t.elem) == "Tensor":
            # 返回带有 TensorList 类型及可选的数组大小的字符串
            return f"TensorList[{size}]" if size is not None else "TensorList"
        # 检查元素类型是否为 "Scalar"
        elif str(t.elem) == "Scalar":
            # 返回带有 ScalarList 类型及可选的数组大小的字符串
            return f"ScalarList[{size}]" if size is not None else "ScalarList"
        # 检查元素类型是否为 "Tensor?"
        elif str(t.elem) == "Tensor?":
            # 如果是简单类型，返回包含 std::optional<Tensor> 的类型字符串
            if simple_type:
                return "c10::List<::std::optional<Tensor>>"
            # 如果不是简单类型，返回包含 const c10::List<std::optional<Tensor>> & 的类型字符串
            else:
                return "const c10::List<::std::optional<Tensor>> &"
        # 检查元素类型是否为 "Dimname"
        elif str(t.elem) == "Dimname":
            # 返回带有 DimnameList 类型及可选的数组大小的字符串
            return f"DimnameList[{size}]" if size is not None else "DimnameList"
        # 递归调用 argument_type_str 函数，处理 t 的元素类型
        elem = argument_type_str(t.elem, simple_type=simple_type, symint=symint)
        # 返回带有 ArrayRef<elem> 类型的字符串
        return f"ArrayRef<{elem}>"
    
    # 如果类型 t 未被识别，引发运行时错误
    raise RuntimeError(f"unrecognized type {repr(t)}")
# 返回类型为 int 或 None 的函数，根据给定的类型 t 判断其是否类似于列表，并检查元素类型是否不是布尔型，若满足条件则返回列表的大小，否则返回 None
def argument_type_size(t: Type) -> int | None:
    # 检查类型 t 是否类似于列表
    l = t.is_list_like()
    # 如果类型 t 是类似列表的类型，并且元素类型不是布尔型
    if l is not None and str(l.elem) != "bool":
        # 返回列表的大小
        return l.size
    else:
        # 否则返回 None
        return None


# 将给定的 Argument 对象转换为 PythonArgument 对象
def argument(a: Argument) -> PythonArgument:
    return PythonArgument(
        # 使用给定 Argument 对象的名称作为 PythonArgument 的名称
        name=a.name,
        # 使用给定 Argument 对象的类型作为 PythonArgument 的类型
        type=a.type,
        # 将 C++ 表达式 a.default 转换为 Python 默认值表达式，若 a.default 为 None 则返回 None
        default=(
            str(pythonify_default(cpp.default_expr(a.default, a.type, symint=False)))
            if a.default is not None
            else None
        ),
        # PythonArgument 的默认初始化设为 None
        default_init=None,
    )


# 根据 FunctionSchema 生成一个 PythonSignature，用于 .pyi 或 PythonArgParser 代码生成
def signature(
    f: NativeFunction, *, method: bool = False, pyi: bool = False
) -> PythonSignature:
    return signature_from_schema(
        # 使用 FunctionSchema 作为参数调用 signature_from_schema 函数生成 PythonSignature
        f.func, category_override=f.category_override, method=method, pyi=pyi
    )


# 根据 FunctionSchema 生成一个 PythonSignature
def signature_from_schema(
    func: FunctionSchema,
    *,
    category_override: str | None,
    method: bool = False,
    pyi: bool = False,
) -> PythonSignature:
    # 初始化参数列表为 func 的前 self 位置参数
    args: list[Argument] = []
    args.extend(func.arguments.pre_self_positional)
    # 如果不是方法，并且存在 self 参数，则将其添加到参数列表中
    if not method and func.arguments.self_arg is not None:
        args.append(func.arguments.self_arg.argument)
    # 将 func 的后 self 位置参数添加到参数列表中
    args.extend(func.arguments.post_self_positional)
    # 将 func 的前 tensor options 仅关键字参数添加到参数列表中
    args.extend(func.arguments.pre_tensor_options_kwarg_only)
    # 将 func 的后 tensor options 仅关键字参数添加到参数列表中
    args.extend(func.arguments.post_tensor_options_kwarg_only)
    # 将 func 的输出参数列表添加到参数列表中
    args.extend(func.arguments.out)

    # 根据参数列表中的 flat_positional 参数生成输入参数元组
    input_arg_set = {a.name for a in func.arguments.flat_positional}
    input_args = tuple(map(argument, filter(lambda a: a.name in input_arg_set, args)))
    # 根据参数列表中的 flat_kwarg_only 参数生成输入关键字参数元组
    kwarg_only_set = {a.name for a in func.arguments.flat_kwarg_only}
    input_kwargs = tuple(
        map(argument, filter(lambda a: a.name in kwarg_only_set, args))
    )
    # 根据参数列表中的输出参数生成输出参数元组
    out_arg_set = {a.name for a in func.arguments.out}
    outputs = tuple(map(argument, filter(lambda a: a.name in out_arg_set, args)))

    # 检查 func 中是否存在张量输入参数
    has_tensor_input_arg = any(
        a.type.is_tensor_like() for a in func.arguments.flat_non_out
    )
    # 如果 func 的 schema_order_arguments 中存在名为 "requires_grad" 的参数，则引发 ValueError
    if any(a.name == "requires_grad" for a in func.schema_order_arguments()):
        raise ValueError(
            "argument named requires_grad is reserved, should not explicitly add it in the schema"
        )

    # 返回根据 FunctionSchema 生成的 PythonSignature
    return PythonSignature(
        name=func.name,
        input_args=input_args,
        input_kwargs=input_kwargs,
        outputs=outputs,
        method=method,
        category_override=category_override,
        has_tensor_input_arg=has_tensor_input_arg,
        pyi=pyi,
    )
    # 检查函数的返回类型是否包含张量类型
    has_tensor_return = any(r.type.is_tensor_like() for r in func.returns)

    # 获取函数的名称
    name: str = cpp.name(func)

    # 判断函数是否为工厂函数，或者具有张量返回但没有张量输入参数
    is_factory_function = category_override == "factory" or (
        has_tensor_return and not has_tensor_input_arg
    )

    # 判断函数是否为类似或新建函数
    is_like_or_new_function = (
        category_override in ("new", "like")
        or name.startswith("new_")
        or name.endswith("_like")
    )

    # 判断函数是否为虚拟函数
    is_dummy_function = category_override == "dummy"

    # 初始化张量选项参数列表
    tensor_options_args: list[PythonArgument] = []

    # 如果函数是工厂函数或类似/新建函数且不是虚拟函数
    if (is_factory_function or is_like_or_new_function) and not is_dummy_function:

        # 定义获取默认初始化的函数
        def topt_default_init(name: str) -> str | None:
            # 获取函数参数中的张量选项
            topt_args = func.arguments.tensor_options
            if topt_args is None:
                return None
            a = getattr(topt_args, name)
            # 如果默认值为空或为字符串 "None"，则返回 None
            if a.default is None or a.default == "None":
                return None
            # 使用 C++ 代码生成默认表达式
            return cpp.default_expr(a.default, a.type, symint=False)

        # 添加 dtype 参数
        tensor_options_args.append(
            PythonArgument(
                name="dtype",
                type=OptionalType(BaseType(BaseTy.ScalarType)),
                default="None",
                default_init=(
                    None if is_like_or_new_function else topt_default_init("dtype")
                ),
            )
        )

        # 添加 layout 参数
        tensor_options_args.append(
            PythonArgument(
                name="layout",
                type=OptionalType(BaseType(BaseTy.Layout)),
                default="None",
                default_init=(
                    None if is_like_or_new_function else topt_default_init("layout")
                ),
            )
        )

        # 添加 device 参数
        tensor_options_args.append(
            PythonArgument(
                name="device",
                type=OptionalType(BaseType(BaseTy.Device)),
                default="None",
                default_init=(
                    None
                    if is_like_or_new_function
                    else (
                        topt_default_init("device")
                        or "torch::tensors::get_default_device()"
                    )
                ),
            )
        )

        # 添加 pin_memory 参数
        tensor_options_args.append(
            PythonArgument(
                name="pin_memory",
                type=OptionalType(BaseType(BaseTy.bool)),
                default="False",
                default_init=None,
            )
        )

        # 添加 requires_grad 参数
        tensor_options_args.append(
            PythonArgument(
                name="requires_grad",
                type=OptionalType(BaseType(BaseTy.bool)),
                default="False",
                default_init=None,
            )
        )

    # 创建函数返回对象
    returns = PythonReturns(returns=func.returns)
    # 创建 PythonSignature 对象并返回，用于描述函数签名信息
    return PythonSignature(
        # 函数名作为字符串传入
        name=str(func.name.name),
        # 输入参数作为列表传入
        input_args=input_args,
        # 输入关键字参数作为字典传入
        input_kwargs=input_kwargs,
        # 从输出中创建 PythonOutArgument 对象列表传入
        output_args=PythonOutArgument.from_outputs(outputs),
        # 张量选项参数作为元组传入
        tensor_options_args=tuple(tensor_options_args),
        # 返回值描述对象传入
        returns=returns,
        # 方法描述信息传入
        method=method,
    )
# 定义一个函数，接受一个返回类型的元组作为参数，并返回字段名列表
def structseq_fieldnames(returns: tuple[Return, ...]) -> list[str]:
    # 如果返回类型的元组长度小于等于1，或者所有返回类型的名称都为None，则返回空列表
    if len(returns) <= 1 or all(r.name is None for r in returns):
        return []
    else:
        # 如果有任何一个返回类型的名称为None，则抛出值错误异常，因为代码生成不支持未命名的字段
        if any(r.name is None for r in returns):
            # 在Windows上构建时，由于某些原因，链接器无法解析`PyStructSequence_UnnamedField`，
            # 导致构建时出错：
            #
            # python_nn_functions.cpp.obj : error LNK2001: unresolved external symbol
            # PyStructSequence_UnnamedField
            #
            # 因此，在这个时间点上，我们不支持未命名字段在structseq中；你必须要么命名所有字段，
            # 要么一个都不命名。
            raise ValueError("Unnamed field is not supported by codegen")

        # 返回所有返回类型的名称组成的列表
        return [str(r.name) for r in returns]


# 定义一个函数，接受一个类型作为参数，并返回其对应的类型字符串
def argument_type_str_pyi(t: Type) -> str:
    add_optional = False
    # 如果类型是OptionalType类型，则提取其元素类型，并设置标志位add_optional为True
    if isinstance(t, OptionalType):
        t = t.elem
        add_optional = True

    # 如果类型是BaseType类型
    if isinstance(t, BaseType):
        # 根据BaseType的名称选择对应的类型字符串
        if t.name in [BaseTy.int, BaseTy.DeviceIndex]:
            ret = "_int"
        elif t.name == BaseTy.SymInt:
            ret = "Union[_int, SymInt]"
        elif t.name == BaseTy.float:
            ret = "_float"
        elif t.name == BaseTy.str:
            ret = "str"
        elif t.name == BaseTy.Scalar:
            ret = "Union[Number, _complex]"
        elif t.name == BaseTy.ScalarType:
            ret = "_dtype"
        elif t.name == BaseTy.bool:
            ret = "_bool"
        elif t.name == BaseTy.QScheme:
            ret = "_qscheme"
        elif t.name == BaseTy.Layout:
            ret = "_layout"
        elif t.name == BaseTy.Device:
            ret = "Optional[DeviceLikeType]"
        elif t.name == BaseTy.MemoryFormat:
            ret = "memory_format"
        elif t.name == BaseTy.Dimname:
            ret = "Union[str, ellipsis, None]"
        elif t.name == BaseTy.Storage:
            ret = "Union[Storage, UntypedStorage]"
        elif t.name in [BaseTy.Tensor, BaseTy.Generator, BaseTy.Stream]:
            # 对于这些Python模式类型名称，与它们的函数模式名称对应
            ret = t.name.name
    # 如果变量 t 是 ListType 类型的实例
    elif isinstance(t, ListType):
        # 如果列表元素的字符串表示是 "int"
        if str(t.elem) == "int":
            # 如果列表有指定的大小 t.size，则返回 Union[_int, _size]，否则返回 _size
            ret = "Union[_int, _size]" if t.size is not None else "_size"
        # 如果列表元素是 tensor-like 类型
        elif t.is_tensor_like():
            # 标记需要添加 Optional 类型的标志
            # TODO: this doesn't seem right...
            # Tensor?[] 目前被翻译为 Optional[Union[Tuple[Tensor, ...], List[Tensor]]]
            # 可能应该翻译为 Union[Tuple[Optional[Tensor], ...], List[Optional[Tensor]]]
            if isinstance(t.elem, OptionalType):
                add_optional = True
            # 根据列表元素类型和是否有指定大小 t.size 返回不同的 Union 类型字符串
            ret = (
                "Union[Tensor, Tuple[Tensor, ...], List[Tensor]]"
                if t.size is not None
                else "Union[Tuple[Tensor, ...], List[Tensor]]"
            )
        # 如果列表元素的字符串表示是 "float"
        elif str(t.elem) == "float":
            # 返回 Sequence[_float] 类型的字符串
            ret = "Sequence[_float]"
        # 如果列表元素的字符串表示是 "SymInt" 并且有指定大小 t.size
        elif str(t.elem) == "SymInt" and t.size is not None:
            # 获取元素类型的字符串表示
            elem = argument_type_str_pyi(t.elem)
            # 返回 Union[elem, Sequence[elem]] 类型的字符串
            ret = f"Union[{elem}, Sequence[{elem}]]"
        else:
            # 获取元素类型的字符串表示
            elem = argument_type_str_pyi(t.elem)
            # 返回 Sequence[elem] 类型的字符串
            ret = f"Sequence[{elem}]"

    else:
        # 如果变量 t 不是已识别的类型，则抛出运行时错误
        raise RuntimeError(f"unrecognized type {repr(t)}")

    # 如果需要添加 Optional 类型
    if add_optional:
        # 将 ret 字符串封装在 Optional[] 中
        ret = "Optional[" + ret + "]"

    # 返回最终推断出的类型字符串
    return ret
def return_type_str_pyi(t: Type) -> str:
    # 如果参数接受 Union，返回类型应该返回具体类型

    if isinstance(t, OptionalType):
        inner = return_type_str_pyi(t.elem)
        return f"Optional[{inner}]"

    if isinstance(t, BaseType):
        if t.name == BaseTy.Device:
            return "_device"
        elif t.name == BaseTy.Dimname:
            ret = "Optional[str]"
        else:
            return argument_type_str_pyi(t)

    if isinstance(t, ListType):
        inner = return_type_str_pyi(t.elem)
        return f"Tuple[{inner}, ...]"

    return argument_type_str_pyi(t)


def returns_structseq_pyi(signature: PythonSignature) -> tuple[str, str] | None:
    # 提取 Python 函数签名中的返回类型信息，生成 structseq 类型的定义

    python_returns = [return_type_str_pyi(r.type) for r in signature.returns.returns]
    structseq_name = signature.name
    field_names = structseq_fieldnames(signature.returns.returns)
    if field_names:
        # 这些类型是 structseq 对象，类似于命名元组 NamedTuple，但构造函数类似于 tuple 的构造函数。
        # 使用 typing.NamedTuple 无法覆盖 __init__。
        seq_type = f"Tuple[{', '.join(python_returns)}]"
        structseq_def_lines = [
            f"class {structseq_name}({seq_type}):",
        ]
        for name, typ in zip(field_names, python_returns):
            structseq_def_lines.extend(
                [
                    "    @property",
                    f"    def {name}(self) -> {typ}: ...",
                ]
            )
        structseq_def_lines.extend(
            [
                f"    def __new__(cls, sequence: {seq_type}): ...",
                f"    n_fields: _int = {len(field_names)}",
                f"    n_sequeunce_fields: _int = {len(field_names)}",
                "    n_unnamed_fields: _int = 0",
                "    def __init_subclass__(cls) -> NoReturn: ...  # 禁止子类化",
                "",  # 添加一个额外的空行
            ]
        )
        structseq_def = "\n".join(structseq_def_lines)
        # 示例:
        # structseq_def = (
        #     "class max(Tuple[Tensor, Tensor]):\n"
        #     "    @property\n"
        #     "    def values(self) -> Tensor: ...\n"
        #     "    @property\n"
        #     "    def indices(self) -> Tensor: ...\n"
        #     "    def __new__(cls, sequence: Tuple[Tensor, Tensor]): ...\n"
        #     "    n_fields: _int = 2",
        #     "    n_sequeunce_fields: _int = 2",
        #     "    n_unnamed_fields: _int = 0",
        #     "    def __init_subclass__(cls) -> NoReturn: ...  # 禁止子类化",
        # )
        return structseq_name, structseq_def
    return None


def returns_str_pyi(signature: PythonSignature) -> str:
    field_names = structseq_fieldnames(signature.returns.returns)
    if field_names:
        return f"torch.return_types.{signature.name}"

    python_returns = [return_type_str_pyi(r.type) for r in signature.returns.returns]
    # 如果 python_returns 列表的长度大于 1，则表示返回类型是一个元组
    if len(python_returns) > 1:
        # 将列表 python_returns 中的元素用逗号连接成字符串，并加上 "Tuple[" 和 "]"，表示返回类型是一个元组
        return "Tuple[" + ", ".join(python_returns) + "]"
    
    # 如果 python_returns 列表的长度等于 1，则返回列表中唯一的元素作为返回类型
    if len(python_returns) == 1:
        return python_returns[0]
    
    # 如果 python_returns 列表为空（长度为 0），则返回 "None"，表示没有明确的返回类型
    return "None"
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                        C++ Function Dispatch
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# This section provides APIs to generate the code that does C++ function
# dispatch. The C++ function call is wrapped by a lambda function.
# For example:
#
#    // aten::selu_(Tensor(a!) self) -> Tensor(a!)
#    auto dispatch_selu_ = [](Tensor self) -> Tensor {
#      pybind11::gil_scoped_release no_gil;
#      return at::selu_(self);
#    };
#
# The lambda function's signature follows the C++ signature in common
# cases, e.g.:
#
#   // aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
#   [](const Tensor & self, const Tensor & other, Scalar alpha) -> Tensor
#
# For out variant the 'out' argument's type is changed from 'Tensor &'
# to 'Tensor'. It's because when calling the lambda it passes in the
# PythonArgParser output '_r.tensor(3)', which is stack allocated object
# and needs to pass by value. Also see comments in 'dispatch_lambda_return_str()'.
#
#   // aten::add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
#   [](Tensor out, const Tensor & self, const Tensor & other, Scalar alpha) -> Tensor
#
# For multi-output case it can keep using reference type because the
# PythonArgParser output has been unpacked to local variables, e.g.:
#
#   // aten::max.names_dim_max(Tensor self, Dimname dim, bool keepdim=False, *,
#   //     Tensor(a!) max, Tensor(b!) max_values) -> (Tensor(a!) values, Tensor(b!) indices)
#   [](Tensor & max, Tensor & max_values, const Tensor & self, Dimname dim, bool keepdim) -> std::tuple<Tensor,Tensor>
#
# For deprecated python signature, it should follow deprecated python arg order.
# TODO: This is to keep same byte-for-byte result as the old codegen - maybe unnecessary?

# 定义函数dispatch_lambda_args，用于生成C++函数调度的lambda参数
def dispatch_lambda_args(
    ps: PythonSignature, f: NativeFunction, symint: bool = True
) -> tuple[DispatchLambdaArgument, ...]:
    # 如果参数ps是PythonSignatureDeprecated类型，则使用其deprecated_schema
    if isinstance(ps, PythonSignatureDeprecated):
        schema = ps.deprecated_schema
    else:
        schema = f.func

    # 开始生成cpp参数列表，lambda签名始终包括'self'
    cpp_args = cpp.arguments(
        arguments=schema.arguments,
        faithful=False,
        symint=symint,
        method=False,
        cpp_no_default_args=f.cpp_no_default_args,
    )
    # 创建一个集合，包含所有输出参数的名称
    out_args: set[str] = {a.name for a in schema.arguments.out}

    # 将cpp参数转换为lambda参数
    # 定义函数dispatch_lambda_arg，接受一个名为cpp_arg的Binding类型参数，返回一个DispatchLambdaArgument类型对象
    def dispatch_lambda_arg(cpp_arg: Binding) -> DispatchLambdaArgument:
        # 获取cpp_arg的type属性值
        type_str = cpp_arg.type
        # 检查cpp_arg的name是否在out_args列表中，确定是否为输出参数
        is_out_arg = cpp_arg.name in out_args
        
        # 如果存在ps.method并且cpp_arg的name为"self"
        if ps.method and cpp_arg.name == "self":
            # 对于方法的'self'参数，可以使用'const Tensor &'，并简单地忽略可变性！
            type_str = "const at::Tensor &"
        else:
            # 对于其他情况，需要避免临时变量的悬空引用（除非它是解包散列输出）
            # 原因在上面的注释和'dispatch_lambda_return_str()'中有解释
            # TODO: 是否可以避免这种特殊处理？
            ensure_temp_safe = len(out_args) <= 1 or not is_out_arg
            if ensure_temp_safe:
                # 如果确保临时变量安全，则根据类型映射字典修改type_str
                type_str = {
                    "at::Tensor &": "at::Tensor",
                }.get(type_str, type_str)
        
        # 返回一个DispatchLambdaArgument对象，包含cpp_arg的name、type_str和is_out_arg属性
        return DispatchLambdaArgument(
            name=cpp_arg.name,
            type_str=type_str,
            is_out_arg=is_out_arg,
        )

    # 使用map函数将dispatch_lambda_arg应用于cpp_args中的每个元素，并将结果转换为元组返回
    return tuple(map(dispatch_lambda_arg, cpp_args))
# 支持的返回类型，用于检查函数返回类型是否在支持的列表中
SUPPORTED_RETURN_TYPES = {
    "at::Tensor",
    "::std::tuple<at::Tensor,at::Tensor>",
    "::std::tuple<at::Tensor,at::Tensor,at::Tensor>",
    "::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor>",
    "::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>",
    "::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>",
    "::std::tuple<at::Tensor,at::Tensor,at::Tensor,int64_t>",
    "::std::tuple<at::Tensor,at::Tensor,double,int64_t>",
    "::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t>",
    "::std::tuple<at::Tensor,at::Tensor,double,at::Tensor,int64_t>",
    "::std::tuple<double,int64_t>",
    "::std::tuple<at::Tensor,::std::vector<at::Tensor>>",
    "::std::vector<at::Tensor>",
    # 用于闪存注意力的前向和反向传播
    "::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,c10::SymInt,c10::SymInt,at::Tensor,at::Tensor,at::Tensor>",
    "at::Scalar",
    "bool",
    "int64_t",
    "void*",
    "void",
    "at::QScheme",
    "double",
    "at::IntArrayRef",
    "at::ScalarType",
    "at::Stream",
}

# 根据给定的 NativeFunction 返回一个描述返回类型的字符串
def dispatch_lambda_return_str(f: NativeFunction) -> str:
    # 生成一个不带类型注释的返回对象元组，以便用于返回类型的生成
    returns_without_annotation = tuple(
        Return(r.name, r.type, None) for r in f.func.returns
    )
    # 调用 C++ 返回类型的生成函数，获取返回类型的 C++ 表示字符串
    return_str = cpp.returns_type(returns_without_annotation, symint=True).cpp_type()
    # 检查返回类型是否在支持的类型列表中，如果不在则抛出异常
    if return_str not in SUPPORTED_RETURN_TYPES:
        raise RuntimeError(f"{f.func.name} returns unsupported type {return_str}")
    return return_str

# 根据给定的 NativeFunction 返回一个描述 C++ 调度目标的字符串
def cpp_dispatch_target(f: NativeFunction) -> str:
    # 检查函数是否包含符号整数重载
    symint = f.func.has_symint()
    # 生成函数的 C++ 名称字符串
    name = cpp.name(f.func, symint_overload=symint)
    # 如果函数是方法，则返回自身方法的调用字符串
    if Variant.method in f.variants:
        return f"self.{name}"
    # 如果 Variant.function 存在于 f.variants 中，则执行以下代码块
    if Variant.function in f.variants:
        # 如果 f 函数具有张量选项或者函数名以 "_like" 结尾，则选择命名空间 "torch"
        if has_tensor_options(f) or f.func.name.name.base.endswith("_like"):
            namespace = "torch"
        else:
            # 否则选择命名空间 "at"
            namespace = "at"
        # 返回命名空间和名称组合成的字符串
        return f"{namespace}::{name}"
    
    # 如果条件不满足，抛出运行时错误，指明无法调度函数或方法
    raise RuntimeError(f"could not dispatch, neither function nor method: {f.func}")
# 定义一个函数 cpp_dispatch_exprs，用于根据给定的 NativeFunction 对象生成 C++ 表达式序列
def cpp_dispatch_exprs(
    f: NativeFunction,
    *,
    python_signature: PythonSignature | None = None,
) -> tuple[str, ...]:
    # 获取 C++ 函数签名的参数绑定列表
    cpp_args: Sequence[Binding] = _cpp_signature(f, method=False).arguments()

    # 初始化表达式元组为空
    exprs: tuple[str, ...] = tuple()

    # 如果没有提供 PythonSignatureDeprecated 类型的 python_signature 参数
    if not isinstance(python_signature, PythonSignatureDeprecated):
        # 默认使用与 C++ 签名一致的表达式
        exprs = tuple(a.name for a in cpp_args)
    else:
        # 对于已弃用的 Python 签名，可能需要填充一些常量
        exprs = tuple(
            filter(
                lambda n: n != "out" or f.func.is_out_fn(),
                python_signature.deprecated_args_exprs,
            )
        )

    # 如果函数变体中包含 Variant.method，移除表达式中的 "self"
    if Variant.method in f.variants:
        exprs = tuple(filter("self".__ne__, exprs))

    # 返回最终生成的表达式元组
    return exprs


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                     Python / C++ Args Binding
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


# 明确列出 PythonArgParser 对各种支持的类型进行解包的方法
# 这种做法可能比必要的更冗长，部分原因是解包方法命名不规则，
# 部分原因是为了模仿旧的代码生成行为，以拒绝旧代码生成器拒绝的意外或不支持的情况。
# 对于某些情况，这种方法故意比必要的更为严格，例如：不接受具有确定大小的 doublelist。
def arg_parser_unpack_method(
    t: Type, default: str | None, default_init: str | None, *, symint: bool = True
) -> str:
    # 检查是否提供了默认初始化值
    has_default_init = default_init is not None
    # 如果提供了默认初始化值，检查类型是否为特定类型，如果不是则引发 RuntimeError
    if has_default_init and str(t) not in (
        "ScalarType?",
        "ScalarType",
        "Device",
        "Device?",
        "Layout",
        "Layout?",
        "bool",
        "bool?",
    ):
        raise RuntimeError(f"type '{t}' does not supported unpacking with default")
    # 检查类型 t 是否是 BaseType 的实例
    if isinstance(t, BaseType):
        # 检查 t.name 是否在预定义的类型列表中
        if t.name in [
            BaseTy.Tensor,
            BaseTy.Stream,
            BaseTy.Storage,
            BaseTy.Scalar,
            BaseTy.Dimname,
        ]:
            # 返回与 t.name 对应的小写字符串，这些字符串与它们的模式名称相对应
            return t.name.name.lower()
        elif t.name == BaseTy.ScalarType:
            # 如果具有默认初始化，则返回 "scalartypeWithDefault"，否则返回 "scalartype"
            return "scalartypeWithDefault" if has_default_init else "scalartype"
        elif t.name == BaseTy.Device:
            # 如果具有默认初始化，则返回 "deviceWithDefault"，否则返回 "device"
            return "deviceWithDefault" if has_default_init else "device"
        elif t.name == BaseTy.DeviceIndex:
            # 返回 "toInt64"
            return "toInt64"
        elif t.name == BaseTy.int:
            # 返回 "toInt64"
            return "toInt64"
        elif t.name == BaseTy.SymInt:
            # 如果 symint 为真，则返回 "toSymInt"，否则返回 "toInt64"
            return "toSymInt" if symint else "toInt64"
        elif t.name == BaseTy.bool:
            # 如果具有默认初始化，则返回 "toBoolWithDefault"，否则返回 "toBool"
            return "toBoolWithDefault" if has_default_init else "toBool"
        elif t.name == BaseTy.float:
            # 返回 "toDouble"
            return "toDouble"
        elif t.name == BaseTy.str:
            # 返回 "stringView"
            return "stringView"
        elif t.name == BaseTy.Layout:
            # 如果具有默认初始化，则返回 "layoutWithDefault"，否则返回 "layout"
            return "layoutWithDefault" if has_default_init else "layout"
        elif t.name == BaseTy.MemoryFormat:
            # 返回 "memoryformat"
            return "memoryformat"

    # 如果 t 是 OptionalType 的实例
    elif isinstance(t, OptionalType):
        if str(t.elem) == "Tensor":
            # 返回 "optionalTensor"
            return "optionalTensor"
        elif str(t.elem) == "Generator":
            # 返回 "generator"
            return "generator"
        elif str(t.elem) == "Dimname[]":
            # 返回 "toDimnameListOptional"
            return "toDimnameListOptional"
        elif not has_default_init and default in (
            None,
            "None",
            "c10::nullopt",
            "::std::nullopt",
            "std::nullopt",
        ):
            # 如果 default 为 None，则将 elem 的解包方法名称添加 "Optional"
            return (
                arg_parser_unpack_method(t.elem, None, None, symint=symint) + "Optional"
            )
        else:
            # 否则，使用默认值加载作为基础类型
            return arg_parser_unpack_method(
                t.elem, default, default_init, symint=symint
            )

    # 如果 t 是 ListType 的实例
    elif isinstance(t, ListType):
        if str(t.elem) == "Tensor":
            # 如果有确定的大小，则返回对应的张量列表格式，否则返回通用格式
            return f"tensorlist_n<{t.size}>" if t.size is not None else "tensorlist"
        elif str(t.elem) == "Tensor?":
            # 返回 "list_of_optional_tensors"
            return "list_of_optional_tensors"
        elif str(t.elem) == "Dimname":
            # 返回 "dimnamelist"
            return "dimnamelist"
        elif str(t.elem) == "int":
            # 返回 "intlist"
            return "intlist"
        elif str(t.elem) == "float":
            # 返回 "doublelist"
            return "doublelist"
        elif str(t.elem) == "SymInt":
            # 如果 symint 为真，则返回 "symintlist"，否则返回 "intlist"
            return "symintlist" if symint else "intlist"
        elif str(t.elem) == "Scalar":
            # 返回 "scalarlist"
            return "scalarlist"

    # 如果未匹配到支持的类型，则引发 RuntimeError
    raise RuntimeError(f"type '{t}' is not supported by PythonArgParser")
# 将 Python 参数解析器的输出转换为右手边（RHS）表达式。
# 例如，对于参数名称 'foo'，参数类型 'bool'，参数索引 = 2，返回 '_r.toBool(2)'。
def arg_parser_output_expr(
    arg_index: int, a: PythonArgument, *, symint: bool = True
) -> PythonArgParserOutputExpr:
    # 检查是否有默认值
    has_default = a.default_init is not None
    # 获取解包方法
    unpack_method = arg_parser_unpack_method(
        t=a.type, default=a.default, default_init=a.default_init, symint=symint
    )
    # 如果有默认值，将其格式化为字符串
    default = f", {a.default_init}" if has_default else ""
    # 构建最终的表达式字符串
    expr = f"_r.{unpack_method}({arg_index}{default})"

    return PythonArgParserOutputExpr(
        name=a.name,
        expr=expr,
        index=arg_index,
        argument=a,
    )


# 返回一个映射，其中键 = arg_name，值 = PythonArgParserOutputExpr。
def arg_parser_output_exprs(
    ps: PythonSignature, f: NativeFunction, *, symint: bool = True
) -> dict[str, PythonArgParserOutputExpr]:
    return {
        e.name: e
        # 遍历参数签名中的每个参数，为每个参数生成 arg_parser_output_expr 对象
        for i, a in enumerate(ps.arguments())
        for e in (arg_parser_output_expr(i, a, symint=symint),)
    }


# tensor 选项字段的参数名称到类型的映射
TENSOR_OPTIONS_FIELDS = {
    "dtype": "ScalarType?",
    "device": "Device?",
    "layout": "Layout?",
    "pin_memory": "bool?",
    "requires_grad": "bool?",
}


# 将 Python 参数解析器的输出与 dispatch lambda 表达式的参数（C++ 参数）绑定。
def dispatch_lambda_exprs(
    ps: PythonSignature, f: NativeFunction, *, symint: bool = True
) -> DispatchLambdaArgumentExprs:
    # 此方法通过生成每个 lambda 参数的 'inits' 和 'lambda_args_exprs'，
    # 以将 'arg_parser_outputs' 和 'lambda_args' 绑定在一起。
    arg_parser_outputs = arg_parser_output_exprs(ps, f, symint=symint)
    lambda_args = dispatch_lambda_args(ps, f, symint=symint)
    inits: list[str] = []  # 初始化 lambda 参数的表达式列表
    lambda_args_exprs: dict[str, str] = {}  # lambda 参数名到表达式的映射

    # 检查函数是否具有 tensor options
    has_toptions = has_tensor_options(f)

    # 1. 为每个 lambda 参数提供特殊的初始化/解包，以提供绑定表达式。
    for a in ps.arguments(skip_tensor_options=True):
        # Iterate over arguments of ps (presumably a PythonSignature object), skipping tensor options
        name = a.name
        # Extract the name of the current argument
        arg_parser_expr = arg_parser_outputs[a.name].expr
        # Retrieve the expression associated with the current argument from arg_parser_outputs

        if has_toptions and name == "self":
            # Check if there are tensor options and if the argument name is "self"
            # Special case handling for "self" argument, likely related to method calls
            inits.extend(
                [
                    f"auto self = {arg_parser_expr};",
                ]
            )
            # Extend 'inits' list with initialization of 'self' variable using 'arg_parser_expr'
            lambda_args_exprs[name] = name
            # Map 'self' to itself in lambda_args_exprs dictionary
        elif (
            isinstance(a, PythonOutArgument)
            and len(a.outputs) > 1
            and f.func.is_out_fn()
        ):
            # Check if argument 'a' is an instance of PythonOutArgument, has multiple outputs, and the function is an output function
            inits.extend(
                [
                    f"auto out = {arg_parser_expr};",
                ]
            )
            # Extend 'inits' list with initialization of 'out' variable using 'arg_parser_expr'
            for i, out_arg in enumerate(a.outputs):
                lambda_args_exprs[out_arg.name] = f"out[{i}]"
                # Map each output argument name to its corresponding index in 'out'
        elif str(a.type) == "Dimname[]?":
            # Check if the argument type is "Dimname[]?" (an optional array of Dimname)
            # [old codegen]
            # Note: Discusses historical context or deprecated code
            # Handle special case for optional<ArrayRef<T>>, explaining conversion complexities
            inits.extend(
                [
                    f"auto __{name} = {arg_parser_expr};",
                    f"::std::optional<DimnameList> {name} = __{name} ? ::std::make_optional(DimnameList(__{name}.value())) : ::std::nullopt;",  # noqa: B950
                ]
            )
            # Extend 'inits' list with handling for optional DimnameList conversion
            lambda_args_exprs[name] = name
            # Map the argument name to itself in lambda_args_exprs dictionary
        else:
            # Default case - direct usage of PythonArgParser output expression
            lambda_args_exprs[name] = arg_parser_expr
            # Map the argument name to its corresponding expression in lambda_args_exprs

    # method's self is passed directly to python binding, rather than parsed
    if ps.method:
        # Check if ps represents a method (likely true)
        lambda_args_exprs["self"] = "self"
        # Map 'self' to itself in lambda_args_exprs for method calls

    # 2. special packing/checking for TensorOptions.
    tensor_options_args_names = [a.name for a in ps.tensor_options_args]
    # Create a list of names from ps.tensor_options_args
    if has_toptions:
        # Check if tensor options are present
        if f.func.is_out_fn():
            # Check if the function is an output function
            raise RuntimeError(f"{f.func}: tensor options with output arg")
            # Raise an error indicating tensor options are incompatible with output arguments
        for a in ps.tensor_options_args:
            # Iterate over tensor options arguments in ps
            if a.name not in TENSOR_OPTIONS_FIELDS:
                # Check if the tensor options field is recognized
                raise RuntimeError(
                    f"{f.func}: unrecognized tensor options field '{a.name}' in python binding arguments"
                )
                # Raise an error for unrecognized tensor options fields
            if str(a.type) != TENSOR_OPTIONS_FIELDS.get(a.name):
                # Check if the tensor options type matches expected type
                raise RuntimeError(
                    f"{f.func}: unrecognized type '{str(a.type)}' for tensor options field '{a.name}'"
                )
                # Raise an error for unrecognized tensor options type
        if not all(a in tensor_options_args_names for a in TENSOR_OPTIONS_FIELDS):
            # Check if all expected tensor options fields are present in tensor_options_args_names
            raise RuntimeError(
                f"{f.func}: incomplete tensor options args: {tensor_options_args_names}"
            )
            # Raise an error indicating incomplete tensor options arguments

        inits.append(
            f"""\
// 创建 TensorOptions 对象并配置其属性，使用来自 arg_parser_outputs 的表达式来填充各个选项
const auto options = TensorOptions()
    .dtype({arg_parser_outputs['dtype'].expr})          // 设置数据类型
    .device({arg_parser_outputs['device'].expr})        // 设置设备类型
    .layout({arg_parser_outputs['layout'].expr})        // 设置布局类型
    .requires_grad({arg_parser_outputs['requires_grad'].expr})  // 设置是否需要梯度
    .pinned_memory({arg_parser_outputs['pin_memory'].expr});  // 设置是否固定内存

// 根据配置的选项，可能初始化设备
torch::utils::maybe_initialize_device(options);

// 将配置好的选项表达式存储到 lambda_args_exprs 中
lambda_args_exprs["options"] = "options";

// 3. 特殊情况 - 在不打包的情况下访问分散的 TensorOptions 字段
// 如果没有使用 TensorOptions，并且存在 tensor_options_args_names，则进行以下处理
if (!has_toptions && tensor_options_args_names) {
    // 如果 tensor_options_args_names 包含 "dtype"
    if ("dtype" in tensor_options_args_names) {
        // 如果函数不是输出参数函数，则抛出运行时错误
        if (!f.func.is_out_fn()) {
            throw RuntimeError(
                f"{f.func}: dtype in tensor_options_args without output arg, {ps} {ps.arguments}"
            );
        }
        // 如果不是所有的参数都在 tensor_options_args_names 中，则抛出错误
        if (!all(a in tensor_options_args_names for a in ("layout", "device"))) {
            throw RuntimeError(
                f"{f.func}: incomplete tensor options for output check"
            );
        }
        
        // 添加检查输出类型匹配的初始化代码
        inits.append(
            f"""\
check_out_type_matches({arg_parser_outputs['out'].expr}, {arg_parser_outputs['dtype'].expr},
                       {arg_parser_outputs['dtype'].is_none_expr}, {arg_parser_outputs['layout'].expr},
                       {arg_parser_outputs['device'].expr}, {arg_parser_outputs['device'].is_none_expr});
"""
        );
    }
    
    // 对于输出张量，设置 requires_grad
    if ("requires_grad" not in tensor_options_args_names) {
        throw RuntimeError(
            f'{f.func}: expected "requires_grad" in tensor_options_args absent, but found [{tensor_options_args_names}]'
        );
    }
}

// 返回一个包含表达式和初始化列表的 DispatchLambdaArgumentExprs 对象
return DispatchLambdaArgumentExprs(
    exprs=tuple(lambda_args_exprs[a.name] for a in lambda_args),
    inits=inits,
);
```