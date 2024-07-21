# `.\pytorch\tools\autograd\gen_inplace_or_view_type.py`

```
# 生成 ADInplaceOrViewType.h/cpp 文件
#
# 注意：如果对 ADInplaceOrView 代码生成进行任何更改，请同时检查
# 是否需要更新 torch/csrc/autograd/autograd_not_implemented_fallback.cpp 文件
# 预期回退应该与此代码生成同步，因此我们应该保持两者的一致性。

from __future__ import annotations

from torchgen.api import cpp
from torchgen.api.autograd import (
    dispatch_strategy,  # 导入 dispatch_strategy
    gen_differentiable_outputs,  # 导入 gen_differentiable_outputs
    NativeFunctionWithDifferentiabilityInfo,  # 导入 NativeFunctionWithDifferentiabilityInfo 类
)
from torchgen.api.types import (
    BaseCType,  # 导入 BaseCType 类
    Binding,  # 导入 Binding 类
    boolT,  # 导入 boolT 类
    ConstRefCType,  # 导入 ConstRefCType 类
    CType,  # 导入 CType 类
    DispatcherSignature,  # 导入 DispatcherSignature 类
    intArrayRefT,  # 导入 intArrayRefT 类
    longT,  # 导入 longT 类
    OptionalCType,  # 导入 OptionalCType 类
    symIntArrayRefT,  # 导入 symIntArrayRefT 类
    SymIntT,  # 导入 SymIntT 类
    tensorT,  # 导入 tensorT 类，参见注释 [Nested Arg Types]
)
from torchgen.code_template import CodeTemplate  # 导入 CodeTemplate 类
from torchgen.context import with_native_function  # 导入 with_native_function 函数
from torchgen.model import (
    NativeFunction,  # 导入 NativeFunction 类
    SchemaKind,  # 导入 SchemaKind 类
    SelfArgument,  # 导入 SelfArgument 类
    TensorOptionsArguments,  # 导入 TensorOptionsArguments 类
    Type,  # 导入 Type 类
)
from torchgen.utils import FileManager  # 导入 FileManager 类

from .context import with_native_function_with_differentiability_info  # 从当前目录导入 with_native_function_with_differentiability_info 函数
from .gen_trace_type import (
    get_return_value,  # 导入 get_return_value 函数
    MANUAL_AUTOGRAD,  # 导入 MANUAL_AUTOGRAD 常量
    tie_return_values,  # 导入 tie_return_values 函数
    type_wrapper_name,  # 导入 type_wrapper_name 函数
)


# 查看 variable.h 文件中的 NOTE [ Autograd View Variables ] 获取详细信息。
# 如果更新了 VIEW_FUNCTIONS 或 RETURNS_VIEWS_OF_INPUT 列表，
# 必须同时在 docs/source/tensor_view.rst 中更新公共视图操作列表。
# 注意并非所有 ATen 函数都公开到公共，例如 alias 和 sparse_coo_tensor_with_dims_and_tensors。
#
# 一个映射：函数名 => 所有输出都是其视图的参数名称

VIEW_FUNCTIONS_WITH_METADATA_CHANGE = [
    "view_as_complex",  # 将 "view_as_complex" 添加到 VIEW_FUNCTIONS_WITH_METADATA_CHANGE 列表
    "view_as_real",  # 将 "view_as_real" 添加到 VIEW_FUNCTIONS_WITH_METADATA_CHANGE 列表
    "_conj",  # 将 "_conj" 添加到 VIEW_FUNCTIONS_WITH_METADATA_CHANGE 列表
    "_neg_view",  # 将 "_neg_view" 添加到 VIEW_FUNCTIONS_WITH_METADATA_CHANGE 列表
    "_nested_get_values",  # 将 "_nested_get_values" 添加到 VIEW_FUNCTIONS_WITH_METADATA_CHANGE 列表
    "_nested_view_from_buffer",  # 将 "_nested_view_from_buffer" 添加到 VIEW_FUNCTIONS_WITH_METADATA_CHANGE 列表
    "_nested_view_from_jagged",  # 将 "_nested_view_from_jagged" 添加到 VIEW_FUNCTIONS_WITH_METADATA_CHANGE 列表
]

# 视图函数映射：函数名 => 所有输出都是其视图的参数名称

VIEW_FUNCTIONS = {
    "numpy_T": "self",  # 将 "numpy_T" 映射到 "self"
    "alias": "self",  # 将 "alias" 映射到 "self"
    "as_strided": "self",  # 将 "as_strided" 映射到 "self"
    "diagonal": "self",  # 将 "diagonal" 映射到 "self"
    "expand": "self",  # 将 "expand" 映射到 "self"
    "permute": "self",  # 将 "permute" 映射到 "self"
    "select": "self",  # 将 "select" 映射到 "self"
    "slice": "self",  # 将 "slice" 映射到 "self"
    "slice_inverse": "self",  # 将 "slice_inverse" 映射到 "self"
    "split": "self",  # 将 "split" 映射到 "self"
    "split_with_sizes": "self",  # 将 "split_with_sizes" 映射到 "self"
    "squeeze": "self",  # 将 "squeeze" 映射到 "self"
    "t": "self",  # 将 "t" 映射到 "self"
    "transpose": "self",  # 将 "transpose" 映射到 "self"
    "unfold": "self",  # 将 "unfold" 映射到 "self"
    "unsqueeze": "self",  # 将 "unsqueeze" 映射到 "self"
    "flatten": "self",  # 将 "flatten" 映射到 "self"
    "view": "self",  # 将 "view" 映射到 "self"
    "unbind": "self",  # 将 "unbind" 映射到 "self"
    "_indices": "self",  # 将 "_indices" 映射到 "self"
    "_values": "self",  # 将 "_values" 映射到 "self"
    "indices": "self",  # 将 "indices" 映射到 "self"
    "values": "self",  # 将 "values" 映射到 "self"
    "crow_indices": "self",  # 将 "crow_indices" 映射到 "self"
    "col_indices": "self",  # 将 "col_indices" 映射到 "self"
    "ccol_indices": "self",  # 将 "ccol_indices" 映射到 "self"
    "row_indices": "self",  # 将 "row_indices" 映射到 "self"
    # sparse_coo_tensor_with_dims_and_tensors 的输出实际上应该是
    # indices 和 values 的视图，但我们只支持将视图作为单个变量的一部分，
    # 而 indices 本身是离散的。
    # FIXME: 在构造时克隆 indices。
    "sparse_coo_tensor_with_dims_and_tensors": "values",  # 将 "sparse_coo_tensor_with_dims_and_tensors" 映射到 "values"
    "_reshape_alias": "self",  # 将 "_reshape_alias" 映射到 "self"
    "_test_autograd_multiple_dispatch_view": "self",  # 将 "_test_autograd_multiple_dispatch_view" 映射到 "self"
}

# 对 VIEW_FUNCTIONS_WITH_METADATA_CHANGE 中的每个键进行映射更新
for key in VIEW_FUNCTIONS_WITH_METADATA_CHANGE:
    VIEW_FUNCTIONS[key] = "self"
# note: some VIEW_FUNCTIONS are just compositions of the view functions above
# this list contains both the root view functions and any that are purely composed
# of viewing functions, and is used by the JIT to determine when an operator
# may return a view of its inputs; however they may sometimes return a copy.
# (e.g. `contiguous`)
RETURNS_VIEWS_OF_INPUT = set(VIEW_FUNCTIONS.keys()).union(
    {
        "chunk",
        "detach",
        "contiguous",
        "reshape",
        "reshape_as",
        "expand_as",
        "view_as",
        "real",
        "imag",
        "narrow",
        "movedim",
        "tensor_split",
        "swapdims",
        "swapaxes",
        "mT",
        "mH",
        "adjoint",
        "matrix_H",
    }
)

# These are the functions we consider views for the purposes of validating
# StorageImpl and TensorImpl in gen_variable_type.
# `_unsafe_view` is not included in VIEW_FUNCTIONS above because it is not a
# view for the purposes of ADInplaceOrView kernel, we do not want to call as_view
# See NOTE [Unsafe View] for more info.
ALL_VIEW_FUNCTIONS = {
    **VIEW_FUNCTIONS,
    "_unsafe_view": "self",
}

ARRAYREF_TO_VEC = CodeTemplate(
    """\
auto ${vec} = ${arg}.vec();
"""
)

OPTIONAL_TO_VAL = CodeTemplate(
    """\
auto ${val} = ${arg}.value_or(${default});
"""
)

CALL_DISPATCH = CodeTemplate(
    """\
at::_ops::${unambiguous_name}::call(${unpacked_args})"""
)

REVERSE_VIEW_DISPATCH = CodeTemplate(
    """\
${reverse_name}(${unpacked_args})"""
)

MULTI_OUTPUT_VIEW_ITERATION = CodeTemplate(
    """\
for (auto ${view_idx} : c10::irange(${var}.size())) {
  ${body}
}
"""
)

SETUP_REPLAY_VIEW_IF_NOT_SUPPORT_AS_STRIDED_OR_VIEW_WITH_METADATA_CHANGE = CodeTemplate(
    """\
std::unique_ptr<torch::autograd::ViewFunc> func(nullptr);
std::function<at::Tensor(const at::Tensor&)> rev_func=nullptr;
if (${is_view_with_metadata_change} ||
    !self.unsafeGetTensorImpl()->support_as_strided() ||
    self.unsafeGetTensorImpl()->is_python_dispatch() ||
    c10::AutogradState::get_tls_state().get_view_replay_enabled()) {
  ${replay_view_func}
  ${reverse_replay_view_func}
}
"""
)

REPLAY_VIEW_FUNC = CodeTemplate(
    """\
func = std::make_unique<${view_func_name}>(${view_func_args});
"""
)

REVERSE_REPLAY_VIEW_LAMBDA_FUNC = CodeTemplate(
    """\
rev_func = [=](const at::Tensor& ${input_view}) {
  return ${reverse_replay_view_call};
};
"""
)

METHOD_DEFINITION = CodeTemplate(
    """\
${return_type} ${type_wrapper_name}(${formals}) {
  ${type_definition_body}
}
"""
)

WRAPPER_REGISTRATION = CodeTemplate(
    """\
m.impl("${unqual_operator_name_with_overload}",
       TORCH_FN(${class_type}::${type_wrapper_name})
);
"""
)

AUTOGRAD_NOT_IMPLEMENTED_REGISTRATION = CodeTemplate(
    """\
m.impl("${unqual_operator_name_with_overload}", torch::autograd::autogradNotImplementedFallback());
"""
)

INPLACE_REDISPATCH = CodeTemplate(
    """\
// Placeholder for inplace redispatch
"""
)
{
  at::AutoDispatchBelowADInplaceOrView guard;
  // 创建自动分发下的 ADInplaceOrView 守卫对象，确保当前操作在 ADInplaceOrView 下进行
  at::_ops::${unambiguous_name}::redispatch(${unpacked_args});
  // 重新分发给 at::_ops::${unambiguous_name} 命名空间下的 redispatch 函数，传入解包后的参数
}
"""
)

ASSIGN_RETURN_VALUE = CodeTemplate(
    """\
${return_values} = ${rhs_value};
"""
)

VIEW_REDISPATCH = CodeTemplate(
    """\
${assign_return_values} ([&]() {
  at::AutoDispatchBelowADInplaceOrView guard;
  // 创建自动分发下的 ADInplaceOrView 守卫对象，确保当前操作在 ADInplaceOrView 下进行
  return at::_ops::${unambiguous_name}::redispatch(${unpacked_args});
  // 重新分发给 at::_ops::${unambiguous_name} 命名空间下的 redispatch 函数，传入解包后的参数
})();
"""
)

TMP_VAR = "_tmp"


// FIXME: Ideally these functions should be methods on Type class, but we have a
//        comment in codegen/model.py there saying these concepts are not well defined.
//        Thus we put a version that commonly used by autograd codegen here.
// 判断是否为张量类型
def is_tensor_type(t: Type) -> bool:
    // TODO: Should handle optional here?
    return t.is_tensor_like() and t.is_list_like() is None


// 判断是否为张量列表类型
def is_tensor_list_type(t: Type) -> bool:
    // TODO: Should handle optional here?
    return t.is_tensor_like() and t.is_list_like() is not None


UNPACK_TENSOR = CodeTemplate(
    """\
auto${ref} ${arg_name}_ = unpack${suffix}(${arg_name}, "${arg_name}", ${arg_pos});"""
)


// 获取解包后的参数名
def unpacked_name(arg_name: str) -> str:
    return arg_name + "_"


// e.g. select.int -> select_copy_int_inverse()
// 生成逆视图函数名
def inverse_view_name(f: NativeFunction) -> str:
    copy_variant = f"{f.root_name}_copy"
    overload = f"{f.func.name.overload_name}"
    if overload != "":
        overload = "_" + overload
    return f"{copy_variant}{overload}_inverse"


// 提取绑定信息
def extract_bindings(f: NativeFunction) -> list[Binding]:
    return [
        r
        for a in f.func.schema_order_arguments()
        for r in cpp.argument(
            a,
            method=False,
            symint=True,
            cpp_no_default_args=set(),
            faithful=False,
            has_tensor_options=False,
        )
    ]


// 解包参数
@with_native_function
def unpack_args(f: NativeFunction) -> tuple[list[str], list[Binding]]:
    body: list[str] = []
    unpacked_bindings: list[Binding] = []
    # 使用 extract_bindings 函数从 f 中提取绑定列表，并使用 enumerate() 枚举每个绑定
    for i, binding in enumerate(extract_bindings(f)):
        # 断言绑定的参数不是 SelfArgument 类型
        assert not isinstance(binding.argument, SelfArgument)
        
        # 如果绑定的参数是 TensorOptionsArguments 类型，则抛出 RuntimeError
        if isinstance(binding.argument, TensorOptionsArguments):
            raise RuntimeError("VariableKernel shouldn't take TensorOptions")
        
        # 检查参数是否可为空
        is_nullable = binding.argument.type.is_nullable()
        
        # 如果参数不是张量样式的，或者可为空，则将该绑定添加到 unpacked_bindings 中并继续下一个绑定
        if not binding.argument.type.is_tensor_like() or is_nullable:
            unpacked_bindings.append(binding)
            continue
        
        # 检查参数是否为张量列表类型
        is_tensor_list = is_tensor_list_type(binding.argument.type)
        
        # 判断是否为非空且非张量列表的情况
        ref = (not is_nullable) and not is_tensor_list
        
        # 如果参数可为空且不是张量列表，则使用后缀 "_opt"；否则不使用后缀
        suffix = "_opt" if is_nullable and not is_tensor_list else ""
        
        # 使用 UNPACK_TENSOR 模板替换，将解包张量的操作添加到 body 列表中
        body.append(
            UNPACK_TENSOR.substitute(
                arg_name=binding.name,
                arg_pos=i,
                suffix=suffix,
                ref="&" if ref else "",  # 如果 ref 为真，则添加取地址符 "&"
            )
        )
        
        # 将解包后的绑定信息添加到 unpacked_bindings 列表中
        unpacked_bindings.append(
            Binding(
                name=unpacked_name(binding.name),
                nctype=binding.nctype,
                argument=binding.argument,
                default=binding.default,
            )
        )

    # 返回解包后的操作列表 body 和更新后的绑定列表 unpacked_bindings
    return body, unpacked_bindings
# 根据给定的原生函数对象 `f`，获取其函数名的基本名称，并返回该名称作为字符串。
def get_base_name(f: NativeFunction) -> str:
    return f.func.name.name.base  # TODO: 应该使用 str(f.func.name.name) 代替 base 名称？


# 根据给定的原生函数对象 `f`，获取与其基本名称相关联的视图信息，返回视图信息字符串或 None。
def get_view_info(f: NativeFunction) -> str | None:
    # 获取函数 `f` 的基本名称
    base_name = get_base_name(f)
    # 从 VIEW_FUNCTIONS 字典中获取基本名称对应的视图信息，若不存在则为 None
    view_info = VIEW_FUNCTIONS.get(base_name, None)
    # 若未找到视图信息并且基本名称存在于 RETURNS_VIEWS_OF_INPUT 中，则视图信息为 "self"
    if view_info is None and base_name in RETURNS_VIEWS_OF_INPUT:
        view_info = "self"
    return view_info


# 生成一个额外的 lambda 函数，用于在不支持 as_strided 的情况下恢复反向视图。
# 查看 "View + Inplace update for base tensor" 和 "View + Inplace update for view tensor" 的详细说明。
def emit_view_func(
    f: NativeFunction, bindings: list[Binding], view_idx: str | None = None
) -> str:
    """
    生成一个额外的 lambda 函数，用于在不支持 as_strided 的情况下恢复反向视图。
    查看 "View + Inplace update for base tensor" 和 "View + Inplace update for view tensor" 的详细说明。
    """
    # TODO: 如果我们去除反向视图函数或使其实例化，应该清理此逻辑。
    # 输入基本名称
    input_base = "input_base"
    # 重播视图函数为空字符串
    replay_view_func = ""
    # 更新后的参数列表
    updated_args: list[str] = []
    # 已知视图参数的简单类型列表
    known_view_arg_simple_types: list[CType] = [
        BaseCType(longT),
        OptionalCType(BaseCType(longT)),
        BaseCType(SymIntT),
        OptionalCType(BaseCType(SymIntT)),
        BaseCType(boolT),
        BaseCType(intArrayRefT),
        BaseCType(symIntArrayRefT),
        ConstRefCType(BaseCType(tensorT)),
        ConstRefCType(OptionalCType(BaseCType(tensorT))),
    ]
    # 遍历参数绑定列表中的每一个绑定对象
    for binding in bindings:
        # 提取参数名和参数类型
        arg, arg_type = binding.name, binding.nctype.type
        # 如果参数名是 "self"，直接将 input_base 添加到更新后的参数列表中并继续下一个循环
        if arg == "self":
            updated_args.append(input_base)
            continue
        # 如果参数类型不在已知的简单视图参数类型列表中，抛出类型错误异常
        if arg_type not in known_view_arg_simple_types:
            # 准备错误消息，指出添加了一个未知类型的参数，并建议更新列表和相关测试
            known_types_str = ", ".join([str(t) for t in known_view_arg_simple_types])
            raise TypeError(
                f"You are adding an {arg_type} {arg} argument to op {cpp.name(f.func)} in addition to known types: "
                f"{known_types_str}. Please update the list or materialize it so that it can be closed "
                "over by value, also add a test in pytorch/xla/test/test_operations.py where this code "
                "is exercised."
            )
        # 如果参数类型是 intArrayRefT 或 symIntArrayRefT，需要将其转换为值类型的向量
        if arg_type == BaseCType(intArrayRefT) or arg_type == BaseCType(symIntArrayRefT):
            arg_vec = arg + "_vec"
            # 在回放视图函数中添加将数组引用转换为向量的操作
            replay_view_func += ARRAYREF_TO_VEC.substitute(arg=arg, vec=arg_vec)
            updated_args.append(arg_vec)
        # 如果参数类型是 OptionalCType(BaseCType(longT))，需要将其转换为值类型的 int64_t
        elif arg_type == OptionalCType(BaseCType(longT)):
            arg_value = arg + "_val"
            # 在回放视图函数中添加将可选的 int64_t 转换为 int64_t 的操作
            replay_view_func += OPTIONAL_TO_VAL.substitute(
                arg=arg, val=arg_value, default="0"
            )
            updated_args.append(arg_value)
        # 如果参数类型是 ConstRefCType(BaseCType(tensorT)) 或 ConstRefCType(OptionalCType(BaseCType(tensorT)))，表示闭合了一个 tensor 类型的参数
        elif arg_type == ConstRefCType(BaseCType(tensorT)) or arg_type == ConstRefCType(OptionalCType(BaseCType(tensorT))):
            # 注意：闭合一个 tensor 类型的参数。如果用户修改此 tensor，这将是静默错误的。正确的做法是存储版本计数并进行写时复制。
            updated_args.append(arg)
        else:
            # 对于其他类型的参数，直接将参数名加入更新后的参数列表
            updated_args.append(arg)

    # 导入视图函数生成器模块，获取视图函数的名称列表
    from .gen_view_funcs import view_func_name

    # 生成视图函数的参数列表，不包括 "self" 参数
    view_func_args = [b.name for b in bindings if b.name != "self"]
    # 如果有视图索引，将其作为字符串添加到参数列表中
    if view_idx is not None:
        view_func_args.append(f"{view_idx}")
    
    # 在回放视图函数中添加生成回放视图函数的代码，包括视图函数的名称和参数列表
    replay_view_func += REPLAY_VIEW_FUNC.substitute(
        view_func_name=view_func_name(f, include_namespace=True),
        view_func_args=view_func_args,
    )

    # 定义输入视图的变量名
    input_view = "input_view"
    # 准备反向解包参数列表，包括 "self"、输入视图、返回模式等
    reverse_unpacked_args = [
        "self",
        f"{input_view}",
        # inverse_return_mode=
        "at::functionalization::InverseReturnMode::AlwaysView",
        *(() if view_idx is None else (f"{view_idx}",)),
        # skip input_base arg
        *updated_args[1:],  # 添加更新后的参数列表（跳过第一个元素 input_base）
    ]

    # 导入反向视图调度函数生成器模块，获取反向视图函数的名称
    from torchgen.api.functionalization import reverse_name

    # 生成反向视图调度函数的调用代码，包括反向视图函数的名称和解包参数列表
    reverse_replay_view_call = REVERSE_VIEW_DISPATCH.substitute(
        reverse_name=reverse_name(f, include_namespace=True),
        unpacked_args=reverse_unpacked_args,
    )
    
    # 生成反向回放视图函数的 Lambda 函数代码，包括输入视图和反向视图调用代码
    reverse_replay_view_func = REVERSE_REPLAY_VIEW_LAMBDA_FUNC.substitute(
        input_view=input_view, reverse_replay_view_call=reverse_replay_view_call
    )
    # 检查函数名是否在具有元数据更改的视图函数列表中，返回布尔值字符串
    is_view_with_metadata_change = (
        "true" if cpp.name(f.func) in VIEW_FUNCTIONS_WITH_METADATA_CHANGE else "false"
    )

    # 使用模板替换字符串，将视图函数的元数据更改情况和相关函数作为参数插入
    return SETUP_REPLAY_VIEW_IF_NOT_SUPPORT_AS_STRIDED_OR_VIEW_WITH_METADATA_CHANGE.substitute(
        is_view_with_metadata_change=is_view_with_metadata_change,
        replay_view_func=replay_view_func,
        reverse_replay_view_func=reverse_replay_view_func,
    )
def emit_view_body(
    fn: NativeFunctionWithDifferentiabilityInfo, var: str
) -> tuple[str, str]:
    # 函数定义，用于生成视图函数体的代码，返回两个字符串
    # 参考 variable.h 中的 NOTE [ Autograd View Variables ] 获取更多细节说明
    f = fn.func
    # 获取函数的基本名称
    base_name = get_base_name(f)
    # 获取函数的视图信息
    view_info = get_view_info(f)
    # 初始化调用字符串为空
    call = ""
    # 生成具有不同可微输出的对象列表
    differentiable_outputs = gen_differentiable_outputs(fn)
    # 不同可微输出变量的名称集合
    differentiable_output_vars = {r.name for r in differentiable_outputs}
    if not isinstance(view_info, str):
        # 如果视图信息不是字符串，则抛出类型错误
        raise TypeError(
            f"The view info should be a string for {base_name}, but it is: {view_info}"
        )
    if len(differentiable_output_vars) == 0:
        # 如果没有可微输出（例如 SparseTensors 的 .indices()）
        rhs_value = (
            f"as_view({view_info}, {var}, "
            f"/* is_bw_differentiable */ false, /* is_fw_differentiable */ false)"
        )
    elif len(differentiable_output_vars) == 1:
        # 单个可微输出（张量或张量列表）
        return_info = differentiable_outputs[0]
        # 我们仅支持简单的张量或张量列表作为返回视图的函数
        if not is_tensor_type(return_info.type) and not is_tensor_list_type(
            return_info.type
        ):
            raise RuntimeError(
                f"{base_name} that return differentiable views can only return Tensor or Tensor[]"
            )

        # 查看注释 [ 视图 + 原地操作检测 ]
        def get_creation_meta_in_mode(original: str) -> str:
            creation_meta_with_grad_mode = f"(at::GradMode::is_enabled() ? {original} : CreationMeta::NO_GRAD_MODE)"
            return f"InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : {creation_meta_with_grad_mode}"

        # 只允许在返回单个张量时重建历史记录
        # 如果我们处于无梯度块中，则发出警告
        # 查看注释 [ 视图 + 原地操作检测 ] 以获取更多有关此逻辑的详细信息
        if is_tensor_list_type(return_info.type):
            creation_meta = get_creation_meta_in_mode("CreationMeta::MULTI_OUTPUT_NODE")
            view_idx = "view_idx"
            view_func = emit_view_func(
                f, extract_bindings(f), view_idx=view_idx
            ).strip()
            as_view_call = (
                f"as_view(/* base */ {view_info}, /* output */ {var}[{view_idx}], "
                "/* is_bw_differentiable */ true, /* is_fw_differentiable */ true, "
                "/* view_func */ std::move(func), /* rev_view_func */ rev_func, "
                f"/* creation_meta */ {creation_meta});"
            )
            call += MULTI_OUTPUT_VIEW_ITERATION.substitute(
                var=var, view_idx=view_idx, body=f"{view_func}\n{as_view_call}"
            )
            rhs_value = f"std::move({var})"
        else:
            call += emit_view_func(f, extract_bindings(f), view_idx=None)
            creation_meta = get_creation_meta_in_mode("CreationMeta::DEFAULT")
            rhs_value = (
                f"as_view(/* base */ {view_info}, /* output */ {var}, /* is_bw_differentiable */ true, "
                "/* is_fw_differentiable */ true, "
                f"/* view_func */ std::move(func), /* rev_view_func */ rev_func, /* creation_meta */ {creation_meta})"
            )
    else:
        # 目前不需要支持此功能，因此保持简单。
        raise RuntimeError(
            "Function that return multiple differentiable output "
            "when at least one of them is view is not supported."
        )
    return call, rhs_value
# 判断函数 f 是否修改其参数，返回布尔值
def modifies_arguments(f: NativeFunction) -> bool:
    return f.func.kind() in [SchemaKind.inplace, SchemaKind.out]


# 根据函数 fn 生成 inplace 或 view 操作的函数体
@with_native_function_with_differentiability_info
def emit_inplace_or_view_body(fn: NativeFunctionWithDifferentiabilityInfo) -> list[str]:
    f = fn.func
    inplace_view_body: list[str] = []

    # 从函数签名生成调度器签名对象
    dispatcher_sig = DispatcherSignature.from_schema(f.func)
    # 获取调度器表达式列表
    dispatcher_exprs = dispatcher_sig.exprs()

    # 将调度键直接通过内核传递以提升性能
    # 参见注释 [Plumbing Keys Through The Dispatcher] 获取详细信息
    dispatch_key_set = "ks & c10::after_ADInplaceOrView_keyset"
    # 生成重新调度的参数列表字符串
    redispatch_args = ", ".join([dispatch_key_set] + [a.expr for a in dispatcher_exprs])

    # 如果函数修改其参数，则为 inplace 操作
    if modifies_arguments(f):
        # 添加 inplace 重调度代码
        inplace_view_body.append(
            INPLACE_REDISPATCH.substitute(
                unambiguous_name=f.func.name.unambiguous_name(),
                unpacked_args=redispatch_args,
            )
        )
        # 对返回值进行版本增量更新
        for r in cpp.return_names(f):
            inplace_view_body.append(f"increment_version({r});")
    else:
        # 否则为 view 操作，确保获取视图信息不为空
        assert get_view_info(f) is not None
        # 添加 view 重调度代码
        inplace_view_body.append(
            VIEW_REDISPATCH.substitute(
                assign_return_values="auto " + TMP_VAR + " = ",
                unambiguous_name=f.func.name.unambiguous_name(),
                unpacked_args=redispatch_args,
            )
        )
        # 生成 view 操作的调用及其右手边值
        call, rhs_value = emit_view_body(fn, TMP_VAR)
        inplace_view_body.append(call)
        assert rhs_value is not None
        # 添加赋值返回值的代码
        inplace_view_body.append(
            ASSIGN_RETURN_VALUE.substitute(
                return_values=tie_return_values(f), rhs_value=rhs_value
            )
        )
    # 如果函数有返回值，则添加返回结果的代码
    if f.func.returns:
        inplace_view_body.append(f"return {get_return_value(f)};")
    return inplace_view_body


# 生成函数形参列表字符串
@with_native_function
def gen_formals(f: NativeFunction) -> str:
    return ", ".join(
        # 生成自动求导内核形参列表，直接通过内核传递调度键以提升性能
        # 参见注释 [Plumbing Keys Through The Dispatcher] 获取详细信息
        ["c10::DispatchKeySet ks"]
        + [
            f'{cpp.argument_type(a, binds="__placeholder__", symint=True).cpp_type()} {a.name}'
            for a in f.func.schema_order_arguments()
        ]
    )


# 生成 inplace 或 view 方法的定义字符串
@with_native_function_with_differentiability_info
def inplace_or_view_method_definition(
    fn: NativeFunctionWithDifferentiabilityInfo,
) -> str | None:
    f = fn.func
    # 如果函数的视图信息为空，并且以下条件之一成立：
    # 1. 函数修改其输入但不返回它们，因此无法支持自动求导。
    #    参见 https://github.com/pytorch/pytorch/issues/53796
    # 2. 函数没有返回值
    if get_view_info(f) is None and (
        not modifies_arguments(f)  # 函数修改了其输入
        or len(f.func.returns) == 0  # 函数没有返回值
    ):
        return None  # 返回空值

    # 使用 METHOD_DEFINITION 模板替换以下内容：
    # - 返回类型：根据函数的返回类型生成 C++ 的类型表示
    # - 类型封装名称：生成函数类型的包装器名称
    # - 形参：生成函数的形式参数列表
    # - 类型定义体：生成内联或视图函数体的定义
    return METHOD_DEFINITION.substitute(
        return_type=cpp.returns_type(f.func.returns, symint=True).cpp_type(),
        type_wrapper_name=type_wrapper_name(f),
        formals=gen_formals(f),
        type_definition_body=emit_inplace_or_view_body(fn),
    )
# 使用装饰器将函数注册为具有不同可区分性信息的本地函数
@with_native_function_with_differentiability_info
def inplace_or_view_method_registration(
    fn: NativeFunctionWithDifferentiabilityInfo,
) -> str | None:
    # 提取函数对象
    f = fn.func
    # 如果该函数不是视图函数且不修改参数，或者返回值为空，则返回None
    if get_view_info(f) is None and (
        not modifies_arguments(f) or len(f.func.returns) == 0
    ):
        return None
    # 使用字符串模板替换器生成包装器注册信息
    return WRAPPER_REGISTRATION.substitute(
        unqual_operator_name_with_overload=f.func.name,
        type_wrapper_name=type_wrapper_name(f),
        class_type="ADInplaceOrView",
    )


def use_derived(fn: NativeFunctionWithDifferentiabilityInfo) -> bool:
    # 提取函数对象
    f = fn.func
    # 获取函数的C++名称
    name = cpp.name(f.func)
    # 如果函数名称不在手动自动微分的集合中，并且调度策略为"use_derived"，则返回True
    return name not in MANUAL_AUTOGRAD and dispatch_strategy(fn) == "use_derived"


def gen_inplace_or_view_type_env(
    fn: NativeFunctionWithDifferentiabilityInfo,
) -> dict[str, list[str]]:
    # 生成内联或视图方法的定义
    definition = inplace_or_view_method_definition(fn)
    # 生成内联或视图方法的注册信息
    registration = inplace_or_view_method_registration(fn)

    # 返回包含生成结果的字典
    return {
        "ops_headers": (
            [f"#include <ATen/ops/{fn.func.root_name}_ops.h>"]
            if definition is not None
            else []
        ),
        "inplace_or_view_method_definitions": [definition]
        if definition is not None
        else [],
        "inplace_or_view_wrapper_registrations": [registration]
        if registration is not None
        else [],
    }


def gen_inplace_or_view_type(
    out: str,
    native_yaml_path: str,
    tags_yaml_path: str,
    fns_with_infos: list[NativeFunctionWithDifferentiabilityInfo],
    template_path: str,
) -> None:
    # 注意：参见VariableType.cpp顶部的“分片文件”注释，关于生成文件的分片处理。
    # 定义分片数量
    num_shards = 2

    # 创建文件管理器实例
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    # 写入分片文件
    fm.write_sharded(
        "ADInplaceOrViewType.cpp",
        [fn for fn in fns_with_infos if use_derived(fn)],
        key_fn=lambda fn: fn.func.root_name,
        base_env={
            "generated_comment": "@"
            + f"generated from {fm.template_dir_for_comments()}/ADInplaceOrViewType.cpp",
        },
        env_callable=gen_inplace_or_view_type_env,
        num_shards=2,
        sharded_keys={
            "ops_headers",
            "inplace_or_view_method_definitions",
            "inplace_or_view_wrapper_registrations",
        },
    )
```