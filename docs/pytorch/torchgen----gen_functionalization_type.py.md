# `.\pytorch\torchgen\gen_functionalization_type.py`

```py
# 从未来版本导入注释，用于支持类型注释
from __future__ import annotations

# 导入 dataclass 模块，用于定义不可变数据类
from dataclasses import dataclass

# 导入 Callable 和 TYPE_CHECKING 类型注释
from typing import Callable, TYPE_CHECKING

# 导入 torchgen 的相关模块和函数
from torchgen.api import cpp, dispatcher
from torchgen.api.translate import translate
from torchgen.api.types import (
    BaseCType,
    Binding,
    CType,
    DispatcherSignature,
    FunctionalizationLambda,
    iTensorListRefT,
    NativeSignature,
    OptionalCType,
    optionalSymIntArrayRefT,
    symIntArrayRefT,
    SymIntT,
    tensorListT,
    tensorT,
    VectorCType,
    ViewInverseSignature,
)

# 导入 torchgen 的上下文相关模块
from torchgen.context import (
    method_with_native_function,
    native_function_manager,
    with_native_function,
    with_native_function_and,
)

# 导入 torchgen 的模型相关模块和类
from torchgen.model import (
    Argument,
    BackendIndex,
    BaseTy,
    BaseType,
    FunctionSchema,
    ListType,
    NativeFunction,
    NativeFunctionsGroup,
    NativeFunctionsViewGroup,
    Return,
    SchemaKind,
    SelfArgument,
    TensorOptionsArguments,
)

# 导入 torchgen 的本地函数生成相关模块
from torchgen.native_function_generation import (
    INPLACE_OPS_THAT_DONT_GET_GROUPED_PROPERLY,
    MUTABLE_OPS_THAT_CANNOT_GET_AN_OUT_VARIANT,
    OUT_OPS_THAT_DONT_GET_GROUPED_PROPERLY,
)

# 导入数据类的字符串表示方法
from torchgen.utils import dataclass_repr

# 如果 TYPE_CHECKING 为真，则导入选择性构建器模块
if TYPE_CHECKING:
    from torchgen.selective_build.selector import SelectiveBuilder


# 注释：[Mutable Ops Not Using Functionalization]
# Ops in this list currently do not work with functionalization and should be fixed.
# 列出不支持功能化的可变操作的名称列表，应该进行修复。
MUTABLE_OPS_NOT_USING_FUNCTIONALIZATION = (
    OUT_OPS_THAT_DONT_GET_GROUPED_PROPERLY
    + MUTABLE_OPS_THAT_CANNOT_GET_AN_OUT_VARIANT
    + INPLACE_OPS_THAT_DONT_GET_GROUPED_PROPERLY
    + [
        # It will be BC-breaking, but we should fix their schemas.
        # should be inplace?
        "record_stream",
        # See Note [resize_ in Functionalization]
        "resize_",
        "resize_as_",
        # This function is used as for testing purposes only.
        "_fill_mem_eff_dropout_mask_",
    ]
)

# 注释：
# 这个文件包含与功能化传递相关的代码生成。
# 它包括以下内容：
# - gen_functionalization_definition
#     为功能化传递生成调度器内核定义。
# - gen_functionalization_registration
#     为功能化传递生成调度器内核注册。
# - gen_functionalization_view_inverse_declaration
#     为功能化传递生成声明，对于每个在功能化中需要的视图操作，我们手动实现它们的定义。
# - gen_composite_view_copy_kernel
#     为所有视图复制操作生成 view_copy() 组合内核。
# 生成 {view}_copy 本地函数的默认复合 C++ 内核主体。
# 参见注释 [view_copy NativeFunctions]
@dataclass(frozen=True)
class GenCompositeViewCopyKernel:
    # 后端索引，用于定义后端实现
    backend_index: BackendIndex

    # 使用本地函数的方法装饰器，包装生成的方法
    @method_with_native_function
    def __call__(self, g: NativeFunctionsViewGroup) -> str | None:
        # 如果 g.view_copy 为 None，则返回 None
        if g.view_copy is None:
            return None
        # 如果 g.view_copy 的函数名不符合标准命名规则 <op>_copy
        elif g.view_copy.func.name.name.base != f"{g.view.func.name.name}_copy":
            # 如果 view_copy 的命名不符合标准命名约定 <op>_copy，
            # 假设它已经存在并且不需要生成。
            # 例如：slice_inverse() 的复制变体命名为 slice_scatter()
            # 而不是 slice_inverse_copy()
            return None

        # 获取 g.view_copy 的元数据
        metadata = self.backend_index.get_kernel(g.view_copy)
        # 断言确保元数据不为 None
        assert metadata is not None

        # 当普通的视图调用会失败时，我们可以通过使用 reshape() 来使 view_copy 在更多情况下工作。
        # 这也使得 LTC 更高效，因为它们不需要在图中包含 clone() 调用（通常由 reshape() 需要）。
        if str(g.view_copy.func.name) == "view_copy":
            # 断言确保元数据的内核类型为 view_copy_symint
            assert metadata.kernel == "view_copy_symint"
            return """\
# 定义一个函数 view_copy_symint，接受一个常量引用 self 和一个 SymIntArrayRef 类型的 size 参数，返回一个 at::Tensor 对象
at::Tensor view_copy_symint(const at::Tensor & self, at::SymIntArrayRef size) {
  # 使用 infer_size_dv 函数推断出符号维度向量 shape
  c10::SymDimVector shape = infer_size_dv(size, self.sym_numel());
  # 如果无法计算出 self 的步长，调用 self 的 reshape_symint 方法返回
  if (!at::detail::computeStride(self.sym_sizes(), self.sym_strides(), shape).has_value()) {
    return self.reshape_symint(size);
  } else {
    # 否则调用 at::_ops::view::call 方法，生成一个视图 output
    auto output = at::_ops::view::call(self, size);
    # 克隆 output，并指定内存格式为 Contiguous，然后返回
    return output.clone(/*memory_format=*/at::MemoryFormat::Contiguous);
  }
}

# 生成一个 NativeSignature 对象 view_copy_sig，指定 symint 为 metadata.supports_symint() 的结果
view_copy_sig = NativeSignature(
    g.view_copy.func, symint=metadata.supports_symint()
)

# 生成一个 DispatcherSignature 对象 view_sig，传入 g.view.func 函数签名
view_sig = DispatcherSignature(g.view.func)

# 获取 g.view.func 函数的非歧义名字，并以字符串形式存储在 view_api_name 中
view_api_name = g.view.func.name.unambiguous_name()

# 使用 translate 函数将 view_copy_sig.arguments() 和 view_sig.arguments() 翻译为表达式列表，并以逗号分隔合并为字符串 exprs
exprs = ", ".join(
    [e.expr for e in translate(view_copy_sig.arguments(), view_sig.arguments())]
)

# 断言 g.view.func 的返回类型列表长度为 1，并且返回类型为 BaseType(BaseTy.Tensor) 或 ListType(BaseType(BaseTy.Tensor), None) 之一
assert len(g.view.func.returns) == 1
assert g.view.func.returns[0].type == BaseType(
    BaseTy.Tensor
) or g.view.func.returns[0].type == ListType(BaseType(BaseTy.Tensor), None)

# 如果返回类型是 BaseType(BaseTy.Tensor)，返回克隆的输出对象，并指定内存格式为 Contiguous
if g.view.func.returns[0].type == BaseType(BaseTy.Tensor):
    return_cloned_output = """\
return output.clone(/*memory_format=*/at::MemoryFormat::Contiguous);"""
else:
    # 如果返回类型是列表，则需要克隆列表中的每个 Tensor
    return_cloned_output = f"""\
{view_copy_sig.returns_type().cpp_type()} out_clone;
for (const auto i : c10::irange(output.size())) {{
  out_clone.push_back(output[i].clone(/*memory_format=*/at::MemoryFormat::Contiguous));
}}
return out_clone;"""

# 默认生成的复合内核，用于 {view}_copy() 操作符，仅克隆输入 Tensor，并在克隆上运行底层视图操作
return f"""
{view_copy_sig.defn(name=metadata.kernel)} {{
  auto output = at::_ops::{view_api_name}::call({exprs});
  {return_cloned_output}
}}
    # 判断变量 a 是否是 SelfArgument 类的实例，如果是则返回 True
    # 或者，如果变量 a 是 Argument 类的实例，并且其类型是类似于张量的（tensor-like），也返回 True
    return isinstance(a, SelfArgument) or (
        isinstance(a, Argument) and a.type.is_tensor_like()
    )
# We need to wrap / unwrap various arguments from the op in the functionalization kernels.
# Some op schemas include non-owning types though (like TensorList),
# and when we unwrap them we expect to get out an owning type!.
# We also return a lambda that tells you how to convert the non-owning type argument into the owning type.
def get_owning_type(t: CType) -> tuple[CType, Callable[[str], str]]:
    if t == BaseCType(tensorListT):
        # Convert TensorList type to VectorCType of tensorT and provide lambda for conversion
        return VectorCType(BaseCType(tensorT)), lambda x: f"{x}.vec()"
    if t == BaseCType(iTensorListRefT):
        # Convert iTensorListRefT type to VectorCType of tensorT and provide lambda for conversion
        return VectorCType(BaseCType(tensorT)), lambda x: f"{{{x}.begin(), {x}.end()}}"
    # There are technically other non-owning types out there (like IntArrayRef),
    # but functionalization only actually cares about the ones involving tensors.
    # Return the original type and an identity lambda for conversion
    return t, lambda x: x


# unwraps all tensor-like arguments, returning:
# (1) a string containing all of the logic that does the unwrapping
# (2) a context, to be used by translate(), with all of the relevant bindings.
def unwrap_tensor_args(
    sig: DispatcherSignature, *, is_view_op: bool
) -> tuple[str, list[Binding]]:
    context: list[Binding] = []
    unwrapped_tensor_args: list[str] = []
    for arg in sig.arguments():
        if is_tensor_like(arg.argument):
            # Generate code to unwrap tensor-like arguments
            unwrapped_name = f"{arg.name}_"
            # Determine whether to sync input tensors before functionalization
            maybe_sync_input = (
                "" if is_view_op else f"at::functionalization::impl::sync({arg.name});"
            )
            # Get owning type and conversion lambda for the argument
            unwrapped_type, conversion_fn = get_owning_type(
                arg.nctype.remove_const_ref().type
            )
            # Generate code to conditionally unwrap functional tensors or apply conversion function
            unwrapped_tensor_args.append(
                f"""
      {unwrapped_type.cpp_type()} {unwrapped_name};
      if (at::functionalization::impl::isFunctionalTensor({arg.name})) {{
        {maybe_sync_input}
        {unwrapped_name} = at::functionalization::impl::from_functional_tensor({arg.name});
      }} else {{
        {unwrapped_name} = {conversion_fn(arg.name)};
      }}"""
            )
            # Append the argument with the unwrapped name to the context
            context.append(arg.with_name(unwrapped_name))
        else:
            # For non-tensor inputs, directly add them to the context
            context.append(arg)
    # Join all unwrapped tensor arguments into a single string
    unwrap_tensor_args_str = "\n      ".join(unwrapped_tensor_args)
    return unwrap_tensor_args_str, context


# converts all tensor-like arguments to meta tensors, which are used to compute stride info. Returns:
# (1) a string containing all of the logic that does the conversions.
# (2) a context, to be used by translate(), with all of the relevant bindings.
# 将函数签名转换为元数据张量的字符串和上下文绑定列表
def convert_to_meta_tensors(sig: DispatcherSignature) -> tuple[str, list[Binding]]:
    # 初始化上下文为空列表
    context: list[Binding] = []
    # 初始化未包装张量参数的列表为空
    unwrapped_tensor_args: list[str] = []
    # 遍历函数签名中的每个参数
    for arg in sig.arguments():
        # 如果参数是张量样式的数据结构
        if is_tensor_like(arg.argument):
            # 获取参数的名称
            a_ = arg.name
            # 创建未包装的张量参数名称，格式为原参数名后加'_meta'
            unwrapped_name = f"{arg.name}_meta"
            # 生成将参数转换为元数据的代码，并添加到未包装张量参数列表中
            unwrapped_tensor_args.append(f"auto {unwrapped_name} = to_meta({a_});")
            # 将具有新名称的参数绑定添加到上下文中
            context.append(arg.with_name(unwrapped_name))
        else:
            # 对于非张量输入参数，直接添加到上下文中
            context.append(arg)
    # 将生成的未包装张量参数的代码连接成字符串，每行缩进4个空格
    unwrap_tensor_args_str = "\n        ".join(unwrapped_tensor_args)
    # 返回未包装的张量参数代码字符串和上下文绑定列表
    return unwrap_tensor_args_str, context


# 检查视图操作符的属性是否符合预期
def assert_view_op_properties(func: FunctionSchema) -> None:
    # 判断参数是否具有别名语义（注解）
    def is_alias(a: Argument) -> bool:
        return a.annotation is not None

    # 获取函数参数中不是输出的所有参数
    args = func.arguments.flat_non_out
    # 第一个参数应为张量，并具有别名语义
    assert len(args) > 0 and args[0].type == BaseType(
        BaseTy.Tensor
    ), f"""In the functionalization codegen, we expect the first argument of every view operator to be a tensor,
but found an argument of type {str(args[0].type)} for operator: {str(func.name)}."""
    # 没有其他参数具有别名语义
    assert is_alias(args[0]) and not any(
        is_alias(a) for a in args[1:]
    ), """In the functionalization codegen, we expect the first argument of every view operator to alias the output.
View operators with multiple aliasing inputs aren't supported yet. Found an operator that doesn't satisfy this constraint"""


# 用于检查表达式是否具有符号值的一行表达式
def emit_expr_has_symbolic_values(expr: str, type: CType) -> str:
    # 如果类型是符号整型
    if type == BaseCType(SymIntT):
        return f"{expr}.is_symbolic()"

    # 如果类型是可选类型
    if isinstance(type, OptionalCType):
        innerexpr = f"(*{expr})"
        return f"{expr}.has_value() ? {emit_expr_has_symbolic_values(innerexpr, type.elem)} : false"

    # 如果类型是可选符号整型数组引用
    if type == BaseCType(optionalSymIntArrayRefT):
        return emit_expr_has_symbolic_values(
            expr, OptionalCType(BaseCType(symIntArrayRefT))
        )

    # 如果类型是符号整型数组引用或者向量类型的符号整型
    if type in (BaseCType(symIntArrayRefT), VectorCType(BaseCType(SymIntT))):
        argname = "arg"
        lambda_check = emit_expr_has_symbolic_values(argname, BaseCType(SymIntT))
        return (
            "std::any_of("
            f"{expr}.begin(), {expr}.end(), "
            f"[=](auto& {argname}) {{ return {lambda_check}; }})"
        )
    # 抛出值错误异常，说明不支持对符号值进行检查
    raise ValueError(
        "unsupported type for has_symbolic_values check. "
        "It should be a SymInt or a collection of those. "
        f"Got: {type.cpp_type()}"
    )
# 为 ViewMeta 构造函数中的 emit_has_symbolic_inputs 函数，检测 SymInt 参数是否包含符号值
def emit_has_symbolic_inputs(sig: DispatcherSignature) -> tuple[str, str]:
    # 设定名称为 "has_symbolic_inputs"
    name = "has_symbolic_inputs"
    # 生成检测语句列表，检查每个参数绑定是否为 Argument 类型且具有符号整数属性
    statements = [
        # 根据绑定的名称和类型发射表达式来检测符号值
        f"{name} = {name} | ({emit_expr_has_symbolic_values(binding.name, binding.nctype.type)});"
        for binding in sig.arguments()
        if (
            isinstance(binding.argument, Argument)
            and binding.argument.type.is_symint_like()
        )
    ]
    # 将所有检测语句连接成一个多行字符串作为函数体
    body = "\n      ".join(statements)
    # 返回函数名称和函数体的元组
    return (
        name,
        f"""
      bool {name} = false;
      {body}""",
    )


# 为 emit_view_functionalization_body 函数生成 Functionalization kernel，用于处理视图操作
def emit_view_functionalization_body(
    g: NativeFunctionsViewGroup, *, view_inplace: bool
) -> str:
    # 如果是 inplace 视图操作
    if view_inplace:
        # 该操作同时是 inplace 和视图操作
        # 参见注释 [Functionalization Pass - Inplace View Ops] 获取详细信息
        # 目前，我让视图元调用视图的非原位变体，以避免定义额外的 ~20 个 inplace {view}_inverse_ 函数。
        # 大多数视图操作没有 NativeFunctionGroup 的两者，因为我们不为视图操作定义 out= 变体。
        # 我假设每个 inplace-view 操作都有一个对应的非原位视图操作，名称相同但末尾的下划线被移除。
        # 目前在 gen.py 的解析时进行断言 (见 error_check_native_functions)。
        assert g.view_inplace is not None
        # 使用 inplace 视图操作函数
        f = g.view_inplace
    else:
        # 使用普通视图操作函数
        f = g.view

    # 断言视图复制函数不为空
    assert g.view_copy is not None
        with native_function_manager(f):
            # 从函数签名创建调度器签名对象
            call_sig = DispatcherSignature.from_schema(g.view_copy.func)

            # 获取需要调用的视图复制操作名称
            api_name = g.view_copy.func.name.unambiguous_name()
            
            # 如果传入的张量不是功能性张量，则需要进行空操作（no-op），即重新调度到原始操作
            noop_api_name = f.func.name.unambiguous_name()

            # 从函数签名创建调度器签名对象
            dispatcher_sig = DispatcherSignature.from_schema(f.func)
            
            # 确保视图操作的属性正确
            assert_view_op_properties(f.func)
            
            # 获取调度器签名中的第一个参数名称（视图张量的名称）
            view_tensor_name = dispatcher_sig.arguments()[0].name

            # 获取返回类型，去除常量引用并转换为 C++ 类型
            return_type = dispatcher_sig.returns_type().remove_const_ref().cpp_type()

            # 解包张量参数并生成对应的字符串表示形式
            unwrap_tensor_args_str, unwrapped_args_ctx = unwrap_tensor_args(
                dispatcher_sig, is_view_op=True
            )

            # 获取视图操作的重定向参数列表
            view_redispatch_args = [
                e.expr
                for e in translate(unwrapped_args_ctx, call_sig.arguments(), method=False)
            ]

            # 根据函数创建功能化 Lambda 表达式对象（正向）
            forward_lambda = FunctionalizationLambda.from_func(g, is_reverse=False)
            
            # 根据函数创建功能化 Lambda 表达式对象（反向）
            reverse_lambda = FunctionalizationLambda.from_func(g, is_reverse=True)

            # Meta API 调用应使用相同的参数，但首先将所有张量转换为元张量
            meta_conversion_str, meta_call_ctx = convert_to_meta_tensors(dispatcher_sig)
            meta_call_args = [
                e.expr for e in translate(meta_call_ctx, call_sig.arguments(), method=False)
            ]

            # 生成检查是否存在符号输入的变量名和检查语句
            symbolic_inputs_varname, symbolic_inputs_check = emit_has_symbolic_inputs(call_sig)

            # 如果函数带有 "inplace_view" 标签
            if "inplace_view" in f.tags:
                # 查看注释中的 [功能化传递 - 原位视图操作] 的注释，以获取更多详细信息
                return f"""
    {
      // 调度函数签名定义，使用给定的参数名生成包装器名称
      dispatcher_sig.defn(name=wrapper_name(f.func), is_redispatching_fn=True)
    } {{
      // 如果视图张量不是 FunctionalTensorWrapper，则不进行功能化处理
      if (!at::functionalization::impl::isFunctionalTensor({view_tensor_name})) {{
        // 功能化是可重入的，但如果未传递 FunctionalTensorWrapper，则将不执行任何操作。
        {unwrap_tensor_args_str}
        // 设置自动调度跳过功能化
        at::AutoDispatchSkipFunctionalize guard;
        // 调用对应的 noop_api_name 操作，并传递视图重分派参数
        return at::_ops::{noop_api_name}::call({', '.join(view_redispatch_args)});
      }}
    
      // 获取功能化重新应用视图
      auto reapply_views = at::functionalization::impl::getFunctionalizationReapplyViewsTLS();
      // 根据 reapply_views 设置逆返回模式
      auto inverse_return_mode = (
          reapply_views ? at::functionalization::InverseReturnMode::ViewOrScatterInverse
                        : at::functionalization::InverseReturnMode::NeverView
      );
    
      // 检查符号输入
      {symbolic_inputs_check}
    
      // 创建视图元数据对象 view_meta
      at::functionalization::ViewMeta view_meta = at::functionalization::ViewMeta(
        // 声明前向 Lambda 函数
        {forward_lambda.decl()} {{
          if (reapply_views) {{
            return {forward_lambda.inner_call(reapply_views=True)}
          }} else {{
            return {forward_lambda.inner_call(reapply_views=False)}
          }}
        }},
        // 声明反向 Lambda 函数
        {reverse_lambda.decl()} {{
          return {reverse_lambda.inner_call()}
        }},
        // 是否具有符号输入
        /*has_symbolic_inputs=*/{symbolic_inputs_varname}
      );
    
      // 计算参考元数据的标志
      auto compute_reference_meta =
        {view_tensor_name}.key_set().has_backend(c10::BackendComponent::XLABit) ||
        {view_tensor_name}.key_set().has_backend(c10::BackendComponent::LazyBit);
    
      // 定义返回类型的 reference_tensor_output 变量
      {return_type} reference_tensor_output;
    
      // 如果需要计算参考元数据，则执行以下操作
      if (compute_reference_meta) {{
        // 执行元数据转换
        {meta_conversion_str}
        // 设置自动调度跳过功能化
        at::AutoDispatchSkipFunctionalize func_guard;
        // 排除元调度关键字保护
        c10::impl::ExcludeDispatchKeyGuard guard(exclude_keys_for_meta_dispatch);
        // 调用对应的 noop_api_name 操作，并传递元调用参数
        reference_tensor_output = at::_ops::{noop_api_name}::call({', '.join(meta_call_args)});
      }}
    
      // 将上述视图元数据添加到当前张量中，并从基本张量上重播它们，修改当前 FunctionalTensorWrapper 的大小/步幅信息。
      // 因此，我们需要确保在此之前运行参考形状函数（否则将使用错误的大小/步幅运行参考函数）。
      at::functionalization::impl::mutate_view_meta({view_tensor_name}, view_meta);
    
      // 参考  Note [Propagating strides in the functionalization pass]
      // XLA/LTC 在功能化传递中不正确地实现了传播步幅的逻辑，因此我们需要在这里依赖一个参考实现
      // （而不是依赖前向 Lambda 的输出具有正确的步幅信息）。
      if (compute_reference_meta) {{
        // 设置大小、步幅和偏移量
        at::functionalization::impl::set_sizes_strides_offset({view_tensor_name}, reference_tensor_output);
      }}
    
      // 返回视图张量
      return {view_tensor_name};
    }}
"""
    # 根据给定的函数对象和变量名，可能创建输出字符串
    def maybe_create_output(f: NativeFunction, var_name: str) -> str:
        # 如果函数的返回类型为空列表，则返回空字符串
        if len(f.func.returns) == 0:
            return ""
        # 根据函数的返回类型生成对应的 C++ 类型字符串，移除常量引用
        return_type = dispatcher.returns_type(f.func.returns).remove_const_ref().cpp_type()
"""
    # 返回一个以指定类型和变量名格式化后的字符串，以 "=" 结尾
    return f"{return_type} {var_name} = "
# 给定一个 NativeFunction 和一个变量名，该变量名对应于在函数上重新分派后的输出，
# 返回两个名称列表，包括：
# - 原始（可变）输入的返回名称列表
# - 内部重新分派函数的（不可变）输出的返回名称列表
def get_mutable_redispatch_return_names(
    f: NativeFunction, inner_return_var: str
) -> tuple[list[str], list[str]]:
    # 存储已别名化的返回值名称
    aliased_returns = []
    # 存储未别名化的返回值名称
    non_aliased_returns = []
    # 遍历函数 f 的别名化返回值名称列表
    for i, name in enumerate(f.func.aliased_return_names()):
        # 如果返回值有别名，将其添加到 aliased_returns 中
        if name is not None:
            aliased_returns.append(name)
        # 否则根据情况生成未别名化的返回值名称并添加到 non_aliased_returns 中
        else:
            non_aliased_returns.append(
                inner_return_var
                if len(f.func.returns) == 1
                else f"std::get<{i}>({inner_return_var})"
            )
    # 返回别名化和未别名化的返回值名称列表
    return aliased_returns, non_aliased_returns


# 当函数化“无操作”并在可变操作符上重新分派时，需确保：
#  - 对于新创建的输出，返回重新分派的结果（无需包装输出）
#  - 对于与输入别名的输出，直接返回输入（因为其中一些可能已被包装）
def return_from_mutable_noop_redispatch(
    f: NativeFunction, inner_return_var: str
) -> str:
    # 调用 get_mutable_redispatch_return_names 函数获取别名化和未别名化的返回值名称列表
    aliased, non_aliased = get_mutable_redispatch_return_names(f, inner_return_var)
    # 返回函数化的返回字符串，将别名化和未别名化的返回值名称列表连接为字符串返回
    return return_str(f.func.returns, aliased + non_aliased)


# 对于给定的 NativeFunction 和 functional_op，以及内部返回变量名 inner_return_var，
# 封装和传播突变并返回字符串
def wrap_propagate_mutations_and_return(
    f: NativeFunction, functional_op: NativeFunction, inner_return_var: str
) -> str:
    # 获取 f 函数的可变参数名称列表
    mutable_arg_names = f.func.arguments.mutable_arg_names()
    # 获取 f 函数的别名化和未别名化的返回值名称列表
    (
        aliased_outer_rets,
        non_aliased_outer_rets,
    ) = get_mutable_redispatch_return_names(f, inner_return_var)
    # 获取 functional_op 函数的别名化和未别名化的返回值名称列表
    _, non_aliased_inner_rets = get_mutable_redispatch_return_names(
        functional_op, inner_return_var
    )
    # 断言：外部函数可能包含别名化和未别名化的输出，但我们转换的内部 functional_op 函数只应包含未别名化的输出
    assert len(mutable_arg_names) + len(non_aliased_outer_rets) == len(
        non_aliased_inner_rets
    )

    # 首先，将内部调用生成的所有新创建的输出包装为 functional tensors
    updates = []
    non_aliased_wrapped_ret_names = []
    # 遍历非别名化的内部返回值列表中与外部返回值对应的部分
    for i, inner_ret in enumerate(
        non_aliased_inner_rets[: len(non_aliased_outer_rets)]
    ):
        # 构造新的返回值名称
        ret_name = f"output_{i}"
        # 将生成的 functional tensor 添加到更新列表中
        updates.append(
            f"""\
  auto output_{i} = at::functionalization::impl::to_functional_tensor({inner_ret});"""
        )
        # 将新创建的返回值名称添加到列表中
        non_aliased_wrapped_ret_names.append(ret_name)

    # 接下来，处理内部调用中对应于可变输入的突变输出，并传播这些突变
    # 使用 zip 函数将 mutable_arg_names 和 non_aliased_inner_rets 对应起来
    for outer_arg, inner_ret in zip(
        mutable_arg_names, non_aliased_inner_rets[len(non_aliased_outer_rets) :]
    ):
        # 构建更新操作的字符串，包括数据传输、替换、提交更新和同步步骤
        updates.append(
            f"""\
  at::functionalization::impl::propagate_xla_data({outer_arg}, {inner_ret});
  at::functionalization::impl::replace_({outer_arg}, {inner_ret});
  at::functionalization::impl::commit_update({outer_arg});
  at::functionalization::impl::sync({outer_arg});"""
        )

    # 最终返回：
    # - 任何既是可变参数也有返回值的内容
    # - 用于包装内部调用输出的任何不可变返回值
    returns_str = return_str(
        f.func.returns, aliased_outer_rets + non_aliased_wrapped_ret_names
    )
    # 将更新操作转换为字符串形式
    updates_str = "\n".join(updates)
    # 返回函数的最终字符串表示，包含更新操作和返回值描述
    return f"""\
{updates_str}
    {returns_str}"""

# 生成功能化内核代码块，用于：
# - 变异操作（就地操作和out=操作）
@with_native_function_and
def emit_inplace_functionalization_body(
    f: NativeFunction, g: NativeFunctionsGroup
) -> str:
    # 确保是变异操作
    assert modifies_arguments(f)

    # 从函数签名创建调度器签名
    dispatcher_sig = DispatcherSignature.from_schema(f.func)

    # 解包张量参数，获取解包后的参数上下文
    unwrap_tensor_args_str, unwrapped_args_ctx = unwrap_tensor_args(
        dispatcher_sig, is_view_op=False
    )

    # 获取所有变异张量名称
    mutated_names = [
        a.name
        for a in f.func.arguments.flat_all
        if a.type.is_tensor_like() and a.annotation is not None
    ]
    # 获取所有非变异张量名称
    non_mutated_names = [
        a.name
        for a in f.func.arguments.flat_all
        if a.type.is_tensor_like() and a.annotation is None
    ]
    # 获取所有非变异张量名称（只限于BaseTy.Tensor类型）
    non_mutated_tensor_names = [
        a.name
        for a in f.func.arguments.flat_all
        if a.type == BaseType(BaseTy.Tensor) and a.annotation is None
    ]

    # 检查所有可变输入是否是功能张量，以便参与功能化
    check_all_mutated_args_are_functional = " && ".join(
        ["true"]
        + [
            f"at::functionalization::impl::isFunctionalTensor({a})"
            for a in mutated_names
        ]
    )

    # 检查任意非变异输入是否是功能张量
    check_any_non_mutated_args_are_functional = " || ".join(
        ["false"]
        + [
            f"at::functionalization::impl::isFunctionalTensor({a})"
            for a in non_mutated_names
        ]
    )

    # 检查任意非变异张量是否是XLA张量
    check_any_non_mutated_tensors_are_xla = " || ".join(
        ["false"]
        + [
            f"{a}.device().type() == c10::DeviceType::XLA"
            for a in non_mutated_tensor_names
        ]
    )

    # 如果不进行功能化，则用于不进行功能化并重新调度到就地操作的情况
    # 情况1：我们遇到一个就地操作，没有对应的非就地操作
    # 情况2：我们遇到就地操作，但是我们的输入不是功能张量（此时内核不执行任何操作）
    inplace_exprs = [
        e.expr
        for e in translate(unwrapped_args_ctx, dispatcher_sig.arguments(), method=False)
    ]

    # 调用out-of-place操作的变体
    return_type = (
        dispatcher.returns_type(g.functional.func.returns).remove_const_ref().cpp_type()
    )
    functional_sig = DispatcherSignature.from_schema(g.functional.func)
    functional_exprs = [
        e.expr
        for e in translate(unwrapped_args_ctx, functional_sig.arguments(), method=False)
    ]

    # 如果函数是out函数，则处理可变输入后处理
    mutable_input_post_processing = "\n".join(
        [
            f"""
      at::functionalization::impl::replace_(
        {a.name}, {'std::get<' + str(i) + '>(tmp_output)' if len(f.func.returns) > 1 else 'tmp_output'});
      at::functionalization::impl::commit_update({a.name});"""
            for (i, a) in enumerate(f.func.arguments.out)
            if a.annotation and a.annotation.is_write and a.type.is_tensor_like()
        ]
    )
    else:
        # 构建可变输入后处理的字符串，将多行字符串组合成一个字符串
        mutable_input_post_processing = "\n".join(
            [
                f"""
      at::functionalization::impl::replace_({a.name}, tmp_output);
      at::functionalization::impl::commit_update({a.name});"""
                for a in f.func.arguments.flat_all
                # 遍历所有扁平化的函数参数，筛选出具有注释、写操作且类型类似张量的参数
                if a.annotation and a.annotation.is_write and a.type.is_tensor_like()
            ]
        )

    # 将调度器签名转换为元数据张量的字符串表示和元调用上下文
    meta_conversion_str, meta_call_ctx = convert_to_meta_tensors(dispatcher_sig)

    # 检查是否有任何存储类型的参数，用于判断是否有需要进行处理的存储参数
    any_storage_args = any(
        a.type == BaseType(BaseTy.Storage) for a in f.func.arguments.flat_all
    )

    # 返回函数的字符串表示，包括构建的可变输入后处理字符串
    return f"""
    {
      // 调度器签名定义，创建调度函数名并标记为重新调度函数
      {dispatcher_sig.defn(name=wrapper_name(f.func), is_redispatching_fn=True)} {{
        // 如果没有存储参数并且函数是原地操作，则执行以下操作
        if ({str(not any_storage_args and f.func.kind() == SchemaKind.inplace).lower()}) {{
          // 在将可变操作转换为其功能性变体之前，通过原始操作运行元张量
          // 这将帮助我们捕捉适用于原地操作但不适用于其功能性变体的形状错误
          // （今天我们只能对原地操作执行此操作，因为它们技术上都支持元张量）
          {meta_conversion_str}
          // 自动分发跳过功能化
          at::AutoDispatchSkipFunctionalize func_guard;
          // 排除调度关键字保护
          c10::impl::ExcludeDispatchKeyGuard guard(exclude_keys_for_meta_dispatch);
          // 调用原地操作的函数
          at::_ops::{f.func.name.unambiguous_name()}::call({', '.join(a.name for a in meta_call_ctx)});
        }}
        // 解包张量参数字符串
        {unwrap_tensor_args_str}
        // 如果不是所有变异参数都是功能张量
        if (!({check_all_mutated_args_are_functional})) {{
          // 如果存在任何XLA张量，我们希望禁用此检查
          // cpu_tensor.copy_(xla_tensor) 是有效代码
          if (!({check_any_non_mutated_tensors_are_xla}) && ({check_any_non_mutated_args_are_functional})) {{
            // 情况1：尝试使用功能张量来变异非功能张量是一个错误
            TORCH_INTERNAL_ASSERT(false,
              "mutating a non-functional tensor with a functional tensor is not allowed.",
              " Please ensure that all of your inputs are wrapped inside of a functionalize() call.");
          }} else {{
            // 情况2：参数不是功能张量，因此我们无操作并重新调度
            // 自动分发跳过功能化
            at::AutoDispatchSkipFunctionalize guard;
            // 调用原地操作的函数
            {maybe_create_output(f, 'tmp_output')}at::_ops::{f.func.name.unambiguous_name()}::call({', '.join(inplace_exprs)});
            // 从可变无操作重新调度返回
            {return_from_mutable_noop_redispatch(f, 'tmp_output')}
          }}
        }} else {{
          // 返回类型临时输出
          {return_type} tmp_output;
          {{
            // 自动分发跳过功能化
            at::AutoDispatchSkipFunctionalize guard;
            // 调用功能表达式的函数
            tmp_output = at::_ops::{g.functional.func.name.unambiguous_name()}::call({', '.join(functional_exprs)});
          }}
          // 包装传播变异并返回
          {wrap_propagate_mutations_and_return(f, g.functional, 'tmp_output')}
        }}
      }}"""
    }
# 生成 RegisterFunctionalization.cpp 文件的函数声明
# 这些文件提供运行 functionalization pass 所需的核心代码，可以选择性地应用于每个后端（例如 XLA 或 Vulkan），
# 或作为可组合的转换（functorch 中的 functionalize()）。

# 查看功能化 pass 的注释 [Functionalization Pass: View Inverses]。
def gen_functionalization_view_inverse_declaration(
    selector: SelectiveBuilder, g: NativeFunctionsViewGroup
) -> str | None:
    # 对于每个（非复合）视图操作，需要一个相应的“反视图”函数声明。
    # 这里生成声明，以便在添加新视图操作时获得明确的编译器错误。
    @with_native_function
    def emit_decl_helper(g: NativeFunctionsViewGroup) -> str | None:
        # 如果视图操作具有复合隐式自动微分内核，则返回 None。
        if g.view.has_composite_implicit_autograd_kernel:
            return None
        # 创建反视图函数的签名对象
        view_inverse_sig = ViewInverseSignature(g)
        # 返回反视图函数的声明
        return view_inverse_sig.decl()

    return emit_decl_helper(g)


def gen_functionalization_registration(
    selector: SelectiveBuilder,
    g: NativeFunction | NativeFunctionsGroup | NativeFunctionsViewGroup,
    composite_implicit_autograd_index: BackendIndex,
) -> list[str]:
    @with_native_function
    def emit_registration_helper(f: NativeFunction) -> str:
        # 断言函数不具有复合隐式自动微分内核。
        assert not f.has_composite_implicit_autograd_kernel
        # 构建函数的注册字符串，形如 "m.impl("func_name", functionalization::wrapper_name(f.func))"
        registration_str = f"TORCH_FN(functionalization::{wrapper_name(f.func)})"
        return f'm.impl("{f.func.name}", {registration_str});'

    # 如果不包含所有操作符，则返回空列表，不生成内核。
    if not selector.include_all_operators:
        return []

    if isinstance(g, NativeFunctionsViewGroup):
        # functionalization 需要为视图和视图内部操作注册内核
        # 查看注释 [Functionalization <> torch.Tensor constructor]
        if str(g.view.func.name) == "lift_fresh":
            return []
        view_str = []
        # 如果视图操作没有复合隐式自动微分内核，则注册视图操作
        if not g.view.has_composite_implicit_autograd_kernel:
            view_str.append(emit_registration_helper(g.view))
        # 如果存在视图内部操作且没有复合隐式自动微分内核，则注册视图内部操作
        if (
            g.view_inplace is not None
            and not g.view_inplace.has_composite_implicit_autograd_kernel
        ):
            assert g.view_inplace.is_view_op
            view_str.append(emit_registration_helper(g.view_inplace))
        return view_str

    elif isinstance(g, NativeFunctionsGroup):
        # 获取手写的功能化内核
        if g.inplace is not None and str(g.inplace.func.name) == "set_.source_Tensor":
            fns = []
        else:
            fns = list(g.functions())
    else:
        # 如果函数名称在 MUTABLE_OPS_NOT_USING_FUNCTIONALIZATION 列表中，则返回空列表，不生成内核。
        if str(g.func.name) in MUTABLE_OPS_NOT_USING_FUNCTIONALIZATION:
            return []
        fns = [g]

    registrations = []
    for f in fns:
        # 对于每个函数对象 f 在 fns 列表中
        if f.has_composite_implicit_autograd_kernel:
            # 如果 f 具有复合隐式自动求导内核，则跳过当前循环
            continue
        if str(f.func.name) == "lift":
            # 如果函数名为 "lift"
            # 参见注释 [Functionalization <> torch.Tensor constructor]
            return []
        if str(f.func.name) == "resize_":
            # 如果函数名为 "resize_"
            # 参见注释 [resize_ in Functionalization]
            return []
        if str(f.func.name.name) != "set_":
            # 如果函数名不是 "set_"
            # 断言 f 不是视图操作
            assert not f.is_view_op
        # functionalization 需要为原地操作生成并注册内核。
        # 我们还需要直接注册复合隐式自动求导内核，
        # 以便在 functionalization 之前正确分解它们。
        if modifies_arguments(f):
            # 如果修改了参数
            registrations.append(emit_registration_helper(f))
    # 返回注册的列表
    return registrations
def gen_functionalization_definition(
    selector: SelectiveBuilder,
    # 注意：理想情况下，此代码不应该需要查看 NativeFunction，
    # 而是只需操作分组后的 NativeFunctions。
    # 目前唯一的原因是我们需要生成直接分发注册，
    # 用于 CompositeImplicitAutograd 运算符，它们可能未分组。
    g: NativeFunction | NativeFunctionsGroup | NativeFunctionsViewGroup,
) -> list[str]:
    # 在移动端构建时不生成内核
    if not selector.include_all_operators:
        return []

    if isinstance(g, NativeFunctionsViewGroup):
        # 情况1：为功能化处理生成 view -> view_copy 内核
        view_defs = []
        if not g.composite:
            # 不变量：NativeFunctionsViewGroup 一定有 view_copy 运算符
            assert g.view_copy is not None, dataclass_repr(g, indent=1)
            view_defs.append(emit_view_functionalization_body(g, view_inplace=False))
            if g.view_inplace is not None:
                view_defs.append(emit_view_functionalization_body(g, view_inplace=True))
        return view_defs
    elif isinstance(g, NativeFunction):
        # 不变量：所有需要在功能化中处理的可变运算符都应该已经正确分组。
        # TODO: 以下所有操作都有“问题”的模式，阻止它们被功能化。
        # 而不是绞尽脑汁让一切正常运行，我认为我们应该：
        # (1) 修复它们的模式（破坏向后兼容）
        # (2) 手动编写它们的功能化内核
        if (
            str(g.func.name) not in MUTABLE_OPS_NOT_USING_FUNCTIONALIZATION
            and str(g.func.name.name) not in MUTABLE_OPS_NOT_USING_FUNCTIONALIZATION
        ):
            assert g.has_composite_implicit_autograd_kernel or not modifies_arguments(g)
        return []
    else:
        # 情况2：为功能化处理生成 inplace -> out-of-place 内核
        mutation_defs = []
        mutation_defs.append(emit_inplace_functionalization_body(g.out, g))
        if g.inplace is not None:
            mutation_defs.append(emit_inplace_functionalization_body(g.inplace, g))
        if g.mutable is not None:
            mutation_defs.append(emit_inplace_functionalization_body(g.mutable, g))
        return mutation_defs
    return []
```