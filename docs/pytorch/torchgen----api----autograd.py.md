# `.\pytorch\torchgen\api\autograd.py`

```
# 引入将来版本的注释特性，允许在类型注释中使用类本身
from __future__ import annotations

# 引入正则表达式模块
import re
# 引入用于数据类的装饰器
from dataclasses import dataclass
# 引入类型提示模块中的cast函数和Sequence泛型
from typing import cast, Sequence

# 引入torchgen库的本地模块
from torchgen import local
# 引入torchgen库的C++ API模块
from torchgen.api import cpp
# 引入torchgen库的类型定义相关模块
from torchgen.api.types import BaseCType, Binding, NamedCType, tensorListT
# 引入torchgen库的模型定义模块
from torchgen.model import (
    BaseTy,
    BaseType,
    FunctionSchema,
    ListType,
    NativeFunction,
    NativeFunctionsViewGroup,
    SchemaKind,
    Type,
)
# 引入torchgen库的工具模块中的IDENT_REGEX正则表达式
from torchgen.utils import IDENT_REGEX


# 表示在反向计算中涉及的已保存属性
# 注意，它可以是输入参数的派生属性，例如：
# 我们可以保存`other.scalar_type()`而不是整个`other`张量。
@dataclass(frozen=True)
class SavedAttribute:
    # NamedCType持有属性的更新名称和cpp类型
    # 对于名称，如果是派生属性，将附加后缀，例如：`other_scalar_type`
    nctype: NamedCType

    # 保存时读取派生属性的表达式，例如：`other.scalar_type()`
    expr: str


# 表示计算一个或多个张量导数的反向公式
@dataclass(frozen=True)
class Derivative:
    # 公式字符串（合法的C++表达式）
    # 注意，对输入参数的表达式已被相应的已保存属性替换。
    # 例如：
    # 原始公式：`mul_tensor_backward(grad, self, other.scalar_type())`
    # 在这里：`mul_tensor_backward(grad, self, other_scalar_type)`
    formula: str

    # 替换输入参数之前的原始公式字符串
    original_formula: str

    # 计算此公式导数的参数名称
    var_names: tuple[str, ...]

    # 被公式引用的已保存输入
    saved_inputs: tuple[SavedAttribute, ...]

    # 被公式引用的已保存输出
    saved_outputs: tuple[SavedAttribute, ...]

    # 公式中按名称引用的梯度
    named_gradients: set[str]


# 表示计算一个张量正向导数的前向公式
@dataclass(frozen=True)
class ForwardDerivative:
    # 公式字符串（合法的C++表达式）
    # 注意，特殊关键词如“linear”或“element_wise”已被自动生成的公式替换。
    formula: str

    # 计算此公式正向导数的输出参数名称
    var_names: tuple[str, ...]

    # 计算此公式正向导数的输出参数类型
    var_types: tuple[Type, ...]

    # 此公式所需的前向导数的输入
    required_inputs_fw_grad: tuple[str, ...] | None

    # 此公式所需的原始值输入
    required_inputs_primal: tuple[str, ...] | None

    # 指定此公式是否需要self的原始值
    # 仅用于原地操作
    required_original_self_value: bool
    # 检查是否指定了此公式在 derivatives.yaml 中，或者我们正在为 inplace 重新使用 out of place 的公式
    is_reusing_outplace_formula: bool
# Represents differentiability info for a NativeFunction.
@dataclass(frozen=True)
class DifferentiabilityInfo:
    # The base name read from derivatives.yaml.
    name: str

    # The matching native function.
    #
    # There can be multiple NativeFunction having the same base name:
    #  - different overloads with different types of input arguments;
    #  - in-place/out/functional variants of the same function;
    #
    # We first use the schema string (under the 'name' key) in derivatives.yaml
    # to find the NativeFunction having the same schema string.
    # Then we find the in-place/out/functional variants of the matching function.
    # Among these variants, we choose the one having the same name as the
    # derivatives.yaml entry. If there is no exact match, then we choose the
    # in-place variant.
    # TODO: maybe the logic to search for all variants is no longer necessary?
    func: NativeFunction

    # The name of the generated autograd function.
    # It's set only if we will calculate a derivative, i.e.
    # 'args_with_derivatives' is not empty.
    op: str | None

    # The derivatives formulae for this function.
    # Note that the length of this sequence is the number of differentiable inputs
    derivatives: Sequence[Derivative]

    # The forward derivatives formulae for this function.
    # Note that the length of this sequence is the number of differentiable outputs
    forward_derivatives: Sequence[ForwardDerivative]

    # The union of 'saved_inputs' of all 'derivatives'.
    all_saved_inputs: Sequence[SavedAttribute]

    # The union of 'saved_outputs' of all 'derivatives'.
    all_saved_outputs: Sequence[SavedAttribute]

    # All named gradients that are available for use, in the same
    # order as in the grads vector.
    available_named_gradients: Sequence[str]

    # The named gradients that are used in any of the derivatives.
    # Invariant: all(name in available_named_gradients for name in used_named_gradients)
    used_named_gradients: set[str]

    # The function's input arguments for which it calculates derivatives.
    # It's the union of 'var_names' of all 'derivatives', sorted by the
    # argument order in the function schema.
    args_with_derivatives: Sequence[Binding]

    # Names of arguments whose derivative formula is 'non_differentiable'.
    non_differentiable_arg_names: Sequence[str]

    # Raw data read from derivatives.yaml.
    output_differentiability: list[bool] | None

    # output_differentiability in derivatives.yaml can be a list of
    # conditions that express if the output is differentiable. In this case,
    # the number of conditions must match the number of outputs
    # (NB: we only support one condition right now).
    # output_differentiability gets populated with True for each condition,
    # while output_differentiability_conditions gets populated with the conditions
    output_differentiability_conditions: list[str] | None

    @property
    # 检查对象是否具有导数信息，即args_with_derivatives列表的长度是否大于0
    def has_derivatives(self) -> bool:
        return len(self.args_with_derivatives) > 0

    # 使用完全相同的导数信息生成一个新的DifferentiabilityInfo对象，
    # 但使用一个新的操作符名称。
    # 这在生成视图操作的“副本”变体时使用，
    # 这些操作可以使用与原始视图操作完全相同的导数公式。
    # 参见注释 [Codegen'd {view}_copy Operators]
    def create_view_copy_from_view_derivative(
        self, g: NativeFunctionsViewGroup
    ) -> DifferentiabilityInfo | None:
        # 如果g.view_copy为None，则返回None
        if g.view_copy is None:
            return None
        # 将g.view_copy赋值给f
        f = g.view_copy

        # 使用点号分割self.name，最多分割2次
        name_split_by_period = self.name.split(".", maxsplit=2)
        # 将操作符的基本名称添加"_copy"（但保持重载名称不变）
        view_copy_name = f"{name_split_by_period[0]}_copy." + ".".join(
            name_split_by_period[1:]
        )
        # 如果self.op为None，则view_copy_op_name为None，否则为self.op+"_copy"
        view_copy_op_name = None if self.op is None else f"{self.op}_copy"

        return DifferentiabilityInfo(
            # 使用"_copy"版本的名称/函数/操作符
            name=view_copy_name,
            func=f,
            op=view_copy_op_name,
            # 但保持所有导数信息不变
            derivatives=self.derivatives,
            forward_derivatives=self.forward_derivatives,
            all_saved_inputs=self.all_saved_inputs,
            all_saved_outputs=self.all_saved_outputs,
            available_named_gradients=self.available_named_gradients,
            used_named_gradients=self.used_named_gradients,
            args_with_derivatives=self.args_with_derivatives,
            non_differentiable_arg_names=self.non_differentiable_arg_names,
            output_differentiability=self.output_differentiability,
            output_differentiability_conditions=self.output_differentiability_conditions,
        )
# 判断给定的 DifferentiabilityInfo 是否为 None，若是则返回 False
def uses_ident(info: DifferentiabilityInfo | None, ident: str) -> bool:
    if info is None:
        return False
    # 遍历 derivatives 列表中的每一个 ForwardDerivative 对象
    for derivative in info.derivatives:
        # 获取当前 derivative 的 formula 属性
        formula = derivative.formula
        # 使用 IDENT_REGEX.format(ident) 进行正则表达式搜索，判断 formula 是否包含指定的 ident
        if re.search(IDENT_REGEX.format(ident), formula):
            return True
    # 若未找到匹配的 ident，则返回 False
    return False


# 判断给定的 DifferentiabilityInfo 是否使用了 "retain_variables"
def uses_retain_variables(info: DifferentiabilityInfo | None) -> bool:
    return uses_ident(info, "retain_variables")


# 判断给定的 DifferentiabilityInfo 是否使用了 "grad"
def uses_single_grad(info: DifferentiabilityInfo | None) -> bool:
    return uses_ident(info, "grad")


# 表示一个可微分的参数（Argument）。
# 它与普通的 Argument 类型有何不同？
# - 它处理的是可微分的参数，仅在自动求导代码生成的上下文中使用；
# - 可以表示 SelfArgument 或常规 Argument，但不包括 TensorOptionsArgument；
@dataclass(frozen=True)
class DifferentiableInput:
    name: str
    type: Type

    # TODO: 仅为了与旧代码兼容保持字节一致性，应删除。
    cpp_type: str


# 表示一个可微分的返回值（Return）。
# 它与普通的 Return 类型有何不同？
# - Return 中的 name 是可选的。在这里，它总是使用相同的 cpp.return_names() 方法填充。
#   TODO: 一些 cpp 命名逻辑（例如解决名称冲突）可能与此处无关？
# - 它处理的是可微分的返回值，符合 derivatives.yaml 中定义的 output_differentiability 字段（如果指定），
#   仅在自动求导代码生成的上下文中使用；
@dataclass(frozen=True)
class DifferentiableOutput:
    name: str
    type: Type

    # TODO: 仅为了与旧代码兼容保持字节一致性，应删除。
    cpp_type: str


@dataclass(frozen=True)
class NativeFunctionWithDifferentiabilityInfo:
    func: NativeFunction
    info: dict[str, DifferentiabilityInfo] | None
    fw_derivatives: dict[str, Sequence[ForwardDerivative]] | None


# TODO: 更新以下注释，因为它已经过时。
def dispatch_strategy(fn: NativeFunctionWithDifferentiabilityInfo) -> str:
    """How are we going to call the underlying implementation of a
    """
    # 如果函数是抽象的或者至少有一个分发键的不同iability信息中具有导数，
    # dispatch_strategy() 用于保护在 VariableType 和 ADInplaceOrViewType 中生成 fn 的操作。
    # 只要任何分发键定义了导数，我们就希望生成这些函数。
    if fn.func.is_abstract or (
        fn.info is not None and any(info.has_derivatives for info in fn.info.values())
    ):
        # 如果函数是抽象的（未在 at::Type 上实现），我们必须在派生类型上使用未打包的张量来调用实现。
        
        # 如果函数已经指定了导数并且是具体的，我们可以调用任一实现。我们更倾向于在派生类型上使用未打包的张量调用实现，
        # 因为在某些情况下性能更好：对其他 ATen 函数的任何内部调用不会跟踪历史。
        
        # 如果函数具有分发到类型的参数（即是工厂函数），我们更倾向于调用派生类型的实现，因为这样更高效，
        # 并且确保工厂函数返回版本为 0 的张量（可能并非绝对必要，但有助于使版本简单易懂）。
        
        return "use_derived"
    else:
        # 如果函数是具体的（不需要在 derivatives.yaml 中声明），并且我们没有重写它，
        # 我们假设它实际上是由不同iable 函数实现的。（这种假设可能不成立，但您会看到 gradcheck 失败。）
        return "use_type"
    ```
# 判断给定的函数是否是一个 foreach 函数
def is_foreach_func(f: NativeFunction) -> bool:
    return f.func.name.name.base.startswith("_foreach_")


# `_foreach_with_inplace_ref` 是一个集合，包含一些 foreach 函数的名称，
# 这些函数引用了 inplace 的 `torch` 函数，其 schema kind 是 functional 的，
# 用于它们的反向导数（将来可能还包括前向导数），例如 `_foreach_zero_`。
_foreach_with_inplace_ref = {"_foreach_zero_"}

# `_foreach_with_tensor_overload` 是一个集合，包含一些 foreach 函数的名称，
# 这些函数支持对 Tensor 的重载，例如 `_foreach_add.Tensor`、`_foreach_mul.Tensor` 等。
_foreach_with_tensor_overload = {
    "_foreach_add.Tensor",
    "_foreach_mul.Tensor",
    "_foreach_div.Tensor",
}

# `_skip_argument_len_check` 是一个集合，包含一些 foreach 函数的名称，
# 这些函数不支持 `alpha` 关键字参数，而非 foreach 版本支持。
# 这些函数通常进行参数长度的检查，例如 `_foreach_add.Scalar`、`_foreach_sub.ScalarList` 等。
_skip_argument_len_check = {
    "_foreach_add.Scalar",
    "_foreach_add_.Scalar",
    "_foreach_add.ScalarList",
    "_foreach_add_.ScalarList",
    "_foreach_sub.Scalar",
    "_foreach_sub_.Scalar",
    "_foreach_sub.ScalarList",
    "_foreach_sub_.ScalarList",
}


# 检查 `function_schema` 是否是一个本地的、非 foreach 的函数，
# 而 `f` 是一个 foreach 函数的引用，用于生成导数。
def is_reference_for_foreach(
    f: NativeFunction,
    function_schema: FunctionSchema,
) -> bool:
    return (
        # 检查 `f` 的函数名是否与 `function_schema` 的函数名匹配，
        # 而且 `function_schema` 不是 inplace 的，或者 `f` 在 `_foreach_with_inplace_ref` 中。
        f.func.name.name.base.split("_foreach_")[-1] == function_schema.name.name.base
        and (
            not function_schema.name.name.inplace
            or str(f.func.name) in _foreach_with_inplace_ref
        )
        and (
            # 检查 `f` 的参数长度是否与 `function_schema` 的参数长度匹配，
            # 或者 `f` 的函数名在 `_skip_argument_len_check` 中。
            str(f.func.name) in _skip_argument_len_check
            or len(f.func.arguments.flat_non_out)
            == len(function_schema.arguments.flat_non_out)
        )
        and all(
            # 检查 `f` 的参数类型是否与 `function_schema` 的参数类型匹配，
            # 或者参数类型是 `arg.type` 或者 `arg.type.elem`。
            ref_arg.type in (arg.type, getattr(arg.type, "elem", None))
            for arg, ref_arg in zip(
                f.func.arguments.flat_non_out,
                function_schema.arguments.flat_non_out,
            )
        )
    )


# TODO(crcrpar): 最好避免硬编码 "Default"。
def gen_foreach_derivativeinfo(
    foreach_function: NativeFunction,
    functional_info_by_signature: dict[
        FunctionSchema, dict[str, DifferentiabilityInfo]
    ],
    non_functional_info_by_signature: dict[
        FunctionSchema, dict[str, DifferentiabilityInfo]
    ],
    dispatch_key: str = "Default",
) -> tuple[DifferentiabilityInfo | None, bool]:
    """为 out-place foreach 函数生成 DifferentiabilityInfo，返回 in-place 函数的现有信息。

    第二个返回值指示是否在此函数中生成了信息。
    """
    # ref_diff_info 初始为 None，用于存储找到的 DifferentiabilityInfo
    ref_diff_info: DifferentiabilityInfo | None = None

    # 遍历 functional_info_by_signature 中的函数签名及其不同导数信息
    for function_schema, diff_info in functional_info_by_signature.items():
        # 如果不是用于 foreach 函数生成导数的引用，则继续下一个循环
        if not is_reference_for_foreach(foreach_function, function_schema):
            continue
        # 获取 dispatch_key 对应的导数信息
        ref_diff_info = diff_info[dispatch_key]
        # 如果找到了有效的 ref_diff_info，则退出循环
        if ref_diff_info is not None:
            break
    # note(crcrpar): 看起来 `zero` 的信息在 functional_info_by_signature 中不可用，
    # 而 `zero_` 的信息在 non_functional_info_by_signature 中可用。
    # 如果 ref_diff_info 为 None，并且 foreach_function 的类型为 inplace，且其名称在 _foreach_with_inplace_ref 中
    if (
        ref_diff_info is None
        and foreach_function.func.kind() == SchemaKind.inplace
        and str(foreach_function.func.name) in _foreach_with_inplace_ref
    ):
        # 遍历 non_functional_info_by_signature.items() 中的 function_schema 和 diff_info
        for function_schema, diff_info in non_functional_info_by_signature.items():
            # 如果不是 foreach_function 的参考函数，则继续下一轮循环
            if not is_reference_for_foreach(foreach_function, function_schema):
                continue
            # 获取 ref_diff_info，并检查其是否不为 None，如果是，则跳出循环
            ref_diff_info = diff_info[dispatch_key]
            if ref_diff_info is not None:
                break
    # 如果 ref_diff_info 仍为 None，则返回 None 和 False
    if ref_diff_info is None:
        return None, False

    # 如果 foreach_function 的类型为 inplace，则返回 ref_diff_info 和 False
    if foreach_function.func.kind() == SchemaKind.inplace:
        return ref_diff_info, False

    # 创建两个空字典 map_refarg2foreacharg 和 map_name2arg
    map_refarg2foreacharg, map_name2arg = {}, {}
    # 遍历 foreach_function.func.arguments.flat_non_out 和 function_schema.arguments.flat_non_out 的元素
    for i, (arg, ref_arg) in enumerate(
        zip(
            foreach_function.func.arguments.flat_non_out,
            function_schema.arguments.flat_non_out,
        )
    ):
        # 将 ref_arg.name 映射到 arg.name 中，构建 map_refarg2foreacharg 字典
        map_refarg2foreacharg[ref_arg.name] = arg.name
        # 将 arg.name 映射到 arg 中，构建 map_name2arg 字典
        map_name2arg[arg.name] = arg

    # 创建空列表 all_saved_inputs, all_saved_outputs 和 all_var_names
    all_saved_inputs, all_saved_outputs, all_var_names = [], [], []
    # 创建空列表 modified_derivative_formulas
    modified_derivative_formulas = []
    for i, derivative in enumerate(ref_diff_info.derivatives):
        # 遍历 ref_diff_info.derivatives 中的每个元素，使用索引 i 和 derivative
        modified_formula = derivative.formula.replace("grad", "grads[i]").replace(
            "result", "result[i]"
        )
        # 根据 derivative.formula 替换字符串中的 "grad" 和 "result"，生成 modified_formula
        
        saved_inputs, saved_outputs = [], []
        # 初始化空列表 saved_inputs 和 saved_outputs

        # note(crcrpar): This context seems necessary to call `cpp.argument_type`
        # 注释：这个上下文似乎是调用 `cpp.argument_type` 必要的
        
        with local.parametrize(
            use_const_ref_for_mutable_tensors=foreach_function.use_const_ref_for_mutable_tensors,
            use_ilistref_for_tensor_lists=foreach_function.part_of_structured_group,
        ):
            # 使用 local.parametrize 上下文管理器，设置参数 use_const_ref_for_mutable_tensors 和 use_ilistref_for_tensor_lists

            for ref_input in derivative.saved_inputs:
                # 遍历 derivative.saved_inputs 中的每个元素，使用 ref_input
                ref_input_jit_name = ref_input.expr.split(".")[0]
                # 从 ref_input.expr 中取出第一个 "." 之前的部分作为 ref_input_jit_name
                mapped_name = map_refarg2foreacharg[ref_input_jit_name]
                # 在 map_refarg2foreacharg 中查找 ref_input_jit_name 对应的映射名称
                
                if isinstance(map_name2arg[mapped_name].type, ListType):
                    # 如果 map_name2arg[mapped_name].type 是 ListType 类型
                    mapped_expr = mapped_name + "[i]"
                else:
                    mapped_expr = mapped_name
                # 根据条件选择 mapped_expr 的值
                
                new_expr = ref_input.expr.replace(ref_input_jit_name, mapped_expr)
                # 使用 mapped_expr 替换 ref_input.expr 中的 ref_input_jit_name，生成 new_expr
                
                modified_formula = modified_formula.replace(
                    cast(str, ref_input.nctype.name), new_expr
                )
                # 替换 modified_formula 中的 ref_input.nctype.name 为 new_expr
                
                nctype = cpp.argument_type(map_name2arg[mapped_name], binds=mapped_name)
                # 调用 cpp.argument_type 函数，使用 map_name2arg[mapped_name] 获取类型信息
                
                canonical_nctype = NamedCType(
                    nctype.name, nctype.type.remove_const_ref()
                )
                # 创建 NamedCType 对象 canonical_nctype，使用 nctype.name 和 nctype.type.remove_const_ref() 初始化
                
                saved_inputs.append(
                    SavedAttribute(nctype=canonical_nctype, expr=mapped_name)
                )
                # 将新创建的 SavedAttribute 对象添加到 saved_inputs 列表中

            for ref_output in derivative.saved_outputs:
                # 遍历 derivative.saved_outputs 中的每个元素，使用 ref_output
                if ref_output.nctype.name == "result":
                    # 如果 ref_output.nctype.name 是 "result"
                    saved_outputs.append(
                        SavedAttribute(
                            nctype=NamedCType(
                                name="result", type=BaseCType(tensorListT)
                            ),
                            expr="result",
                        )
                    )
                    # 创建 NamedCType 为 "result" 的 SavedAttribute 对象，添加到 saved_outputs 列表中
                else:
                    raise RuntimeError("")
                # 如果 ref_output.nctype.name 不是 "result"，则抛出 RuntimeError 异常

        var_names = [map_refarg2foreacharg[var] for var in derivative.var_names]
        # 创建变量名列表 var_names，根据 derivative.var_names 中的每个元素，在 map_refarg2foreacharg 中查找对应的映射名称
        all_var_names.extend(var_names)
        # 将 var_names 中的元素添加到 all_var_names 列表末尾
        all_saved_inputs.extend(saved_inputs)
        # 将 saved_inputs 中的元素添加到 all_saved_inputs 列表末尾
        all_saved_outputs.extend(saved_outputs)
        # 将 saved_outputs 中的元素添加到 all_saved_outputs 列表末尾
        
        modified_derivative = Derivative(
            formula=modified_formula,
            original_formula=derivative.formula,
            var_names=tuple(var_names),
            saved_inputs=tuple(saved_inputs),
            saved_outputs=tuple(saved_outputs),
            named_gradients=set(),
        )
        # 创建 Derivative 对象 modified_derivative，使用给定的参数进行初始化
        
        modified_derivative_formulas.append(modified_derivative)
        # 将 modified_derivative 添加到 modified_derivative_formulas 列表末尾

    with local.parametrize(
        use_const_ref_for_mutable_tensors=foreach_function.use_const_ref_for_mutable_tensors,
        use_ilistref_for_tensor_lists=foreach_function.part_of_structured_group,
    ):
        # 使用 local.parametrize 上下文管理器，再次设置参数 use_const_ref_for_mutable_tensors 和 use_ilistref_for_tensor_lists
    ):
        # 为每个非输出参数生成具有推导信息的绑定对象列表
        args_with_derivatives = [
            Binding(
                name=arg.name,
                nctype=cpp.argument_type(arg, binds=arg.name),
                argument=arg,
                default=None,
            )
            for arg in foreach_function.func.arguments.flat_non_out
            if arg.name in all_var_names
        ]

    # 初始化空的前向导数列表
    forward_derivatives: list[ForwardDerivative] = []
    # 初始化前向导数对象
    fw_derivative: ForwardDerivative
    # 返回元组，包含函数不同iability的信息和True标志
    return (
        DifferentiabilityInfo(
            name=foreach_function.func.name.name.base,
            func=foreach_function,
            op=f"Foreach{ref_diff_info.op}{foreach_function.func.name.overload_name}",
            derivatives=modified_derivative_formulas,
            forward_derivatives=forward_derivatives,
            all_saved_inputs=tuple(set(all_saved_inputs)),
            all_saved_outputs=tuple(set(all_saved_outputs)),
            available_named_gradients=(),
            used_named_gradients=set(),
            args_with_derivatives=args_with_derivatives,
            non_differentiable_arg_names=[],
            output_differentiability=None,
            output_differentiability_conditions=None,
        ),
        True,
    )
    def match_differentiability_info(
        native_functions: list[NativeFunction],
        differentiability_infos: dict[FunctionSchema, dict[str, DifferentiabilityInfo]],
    ) -> list[NativeFunctionWithDifferentiabilityInfo]:
        """Sets the "derivative" key on declarations to matching autograd function
        In-place functions will use the out-of-place derivative definition if there
        is no in-place specific derivative.
        """

        # 根据功能类型，生成去除默认值的函数签名到信息字典的映射
        functional_info_by_signature = {
            schema.signature(strip_default=True): info_dict
            for schema, info_dict in differentiability_infos.items()
            if schema.kind() == SchemaKind.functional
        }

        # 根据非功能类型，生成去除默认值的函数签名到信息字典的映射
        non_functional_info_by_signature = {
            schema.signature(strip_default=True): info_dict
            for schema, info_dict in differentiability_infos.items()
            if schema.kind() != SchemaKind.functional
        }

        def find_info(
            f: NativeFunction,
        ) -> tuple[dict[str, DifferentiabilityInfo] | None, bool]:
            # 如果函数有 "generated" 标签且是 out 类型的函数，则不进行匹配信息
            if "generated" in f.tags and f.func.kind() == SchemaKind.out:
                return None, False

            # (1) 检查是否有精确匹配
            if f.func in differentiability_infos:
                return differentiability_infos[f.func], True

            # (2) 如果没有精确匹配，检查是否有对应的 out-of-place 变体
            # 比如 mul() 对应 mul_() 或 mul_out()
            # 注意：对于 in-place foreach 函数，使用现有的原生函数而不是 out-place 变体
            f_sig = f.func.signature(strip_default=True)
            if f_sig in functional_info_by_signature and not is_foreach_func(f):
                return functional_info_by_signature[f_sig], False

            # (3) 一些操作符对可变变体有显式定义的导数，但生成的 out-of-place 变体可能没有相应的导数公式
            # 对于生成的 out-of-place 变体，如果可变变体的公式存在，则使用它
            if "generated" in f.tags and f_sig in non_functional_info_by_signature:
                info_dict = non_functional_info_by_signature[f_sig]
                # 参考 https://github.com/pytorch/pytorch/pull/76320/files#r874816389
                assert not any(
                    any("self" in str(inpt.nctype.name) for inpt in info.all_saved_inputs)
                    for info in info_dict.values()
                ), "Check for 'self' in inputs should not be present."
        # 尝试将可变运算符的导数公式转换为其功能变体的导数信息
        # （"{str(f.func)}" 自动使用功能变体，但目前不支持（需要在代码生成中修复公式）。
        # 返回空字典和 False 表示失败。
        """
        Attempted to convert a derivative formula for a mutable operator
        to be used by automatically by its functional variant ("{str(f.func)}").
        this is not currently supported (we'd need to fix up the formula in the codegen)."""
        return info_dict, False

    # (4) 如果在 `derivatives.yaml` 中未定义 foreach 函数的导数信息，则生成其导数信息
    if is_foreach_func(f):
        assert f.func not in differentiability_infos
        diff_info, is_generated = gen_foreach_derivativeinfo(
            f,
            functional_info_by_signature,
            non_functional_info_by_signature,
        )
        # 如果未生成导数信息，则返回空和 False 表示失败
        if diff_info is None:
            return None, False
        # 创建包含默认键的导数信息字典
        diff_info_dict = {"Default": diff_info}
        # 如果成功生成了导数信息，则更新相关信息字典
        if is_generated:
            differentiability_infos[f.func] = diff_info_dict
            functional_info_by_signature[f.func] = diff_info_dict
        return diff_info_dict, is_generated

    # 对于其他情况，返回空和 False 表示未找到导数信息
    return None, False

# 创建一个空的列表，用于存储具有不同iable 信息的本地函数
result: list[NativeFunctionWithDifferentiabilityInfo] = []
return result


def is_differentiable(
    name: str, type: Type, info: DifferentiabilityInfo | None
) -> bool:
    # 判断类型是否类似于张量，且不可微参数列表中不包含该名称
    return type.is_tensor_like() and (
        info is None or name not in info.non_differentiable_arg_names
    )


def gen_differentiable_outputs(
    fn: NativeFunctionWithDifferentiabilityInfo, key: str = "Default"
) -> list[DifferentiableOutput]:
    f = fn.func
    # 获取函数的不同iable 信息，如果没有则为 None
    info = fn.info[key] if fn.info else None
    # 创建不同iable 输出列表，每个输出包含名称、类型和对应的 C++ 类型
    outputs: list[DifferentiableOutput] = [
        DifferentiableOutput(
            name=name,
            type=ret.type,
            cpp_type=cpp.return_type(ret, symint=True).cpp_type(),
        )
        for name, ret in zip(cpp.return_names(f), f.func.returns)
    ]
    # 获取输出的不同iable 属性，如果没有则为 None
    output_differentiability = info.output_differentiability if info else None
    # 如果有输出的不同iable 属性
    if output_differentiability is not None:
        # 检查输出的数量和不同iable 属性的长度是否一致
        if len(output_differentiability) != len(outputs):
            raise RuntimeError(
                f"The length of output_differentiability ({len(output_differentiability)}), "
                f"does not match the number of outputs ({len(outputs)})."
            )
        # 创建一个空的不同iable 输出列表
        differentiable_outputs: list[DifferentiableOutput] = []
        # 如果输出标记为不可不同iable 且函数为 inplace 操作，则引发运行时错误
        if False in output_differentiability and f.func.kind() == SchemaKind.inplace:
            raise RuntimeError(
                "output_differentiability=False for inplace operation (version_counter won't get updated)"
            )
        # 遍历输出的不同iable 属性和输出列表，将可不同iable 的输出添加到结果列表中
        for differentiable, output in zip(output_differentiability, outputs):
            if differentiable:
                differentiable_outputs.append(output)
        return differentiable_outputs
    # 如果没有输出的不同iable 属性，则根据 is_differentiable 函数判断每个输出的可不同iable 性
    candidate_differentiable_outputs = list(
        filter(lambda r: is_differentiable(r.name, r.type, info), outputs)
    )
    # 如果使用单个梯度，则返回第一个可不同iable 的输出
    if uses_single_grad(info):
        return candidate_differentiable_outputs[:1]
    # 如果前面的条件不满足，则返回候选的可微分输出
    else:
        return candidate_differentiable_outputs
```