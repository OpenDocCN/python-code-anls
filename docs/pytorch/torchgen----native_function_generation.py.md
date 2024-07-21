# `.\pytorch\torchgen\native_function_generation.py`

```
# 引入未来版本的注释类型声明，用于在代码中指定类型的注释
from __future__ import annotations

# 引入默认字典集合，用于处理默认值的集合类型数据
from collections import defaultdict

# 引入类型提示模块中的序列类型
from typing import Sequence

# 从 torchgen.api 中引入 dispatcher 模块
import torchgen.api.dispatcher as dispatcher

# 从 torchgen.api.translate 中引入 translate 函数
from torchgen.api.translate import translate

# 从 torchgen.api.types 中引入 Binding, DispatcherSignature, Expr 类型
from torchgen.api.types import Binding, DispatcherSignature, Expr

# 从 torchgen.context 中引入 with_native_function 函数
from torchgen.context import with_native_function

# 从 torchgen.model 中引入多个类和常量
from torchgen.model import (
    Annotation,
    Argument,
    BackendIndex,
    BackendMetadata,
    BaseOperatorName,
    BaseTy,
    BaseType,
    DEFAULT_KERNEL_NAMESPACE,
    DeviceCheckType,
    DispatchKey,
    FunctionSchema,
    NativeFunction,
    NativeFunctionsGroup,
    OperatorName,
    Return,
    SchemaKind,
    Variant,
)

# 从 torchgen.utils 中引入 concatMap 函数
from torchgen.utils import concatMap

# 定义列表：包含不正确分组的输出操作的名称，注释标明其为未正确分组的操作
OUT_OPS_THAT_DONT_GET_GROUPED_PROPERLY = [
    # 具有功能变体，但当前标记为私有。
    # 此函数也应标记为私有（*_backward 操作不会向 Python 公开）。
    "adaptive_avg_pool3d_backward.grad_input",
    # 具有功能变体 _slow_conv2d_backward.output_mask，但未正确分组。
    # 可能我们可以使用 convolution_backward 来替代此操作？
    "_slow_conv2d_backward.grad_input",
]

# 定义列表：包含无法获取输出变体的可变操作的名称，注释标明其为无法获取输出变体的操作
MUTABLE_OPS_THAT_CANNOT_GET_AN_OUT_VARIANT = [
    # 应该有 out=？ 参数
    "_cummax_helper",
    # 应该有 out=？ 参数
    "_cummin_helper",
]

# 定义列表：包含无法获取输出变体的功能操作的名称，注释标明其为无法获取输出变体的操作
FUNCTIONAL_OPS_THAT_CANNOT_GET_AN_OUT_VARIANT = [
    "_assert_async",  # 没有返回值
    "_assert_async.msg",  # 没有返回值
    "_cslt_sparse_mm_search",  # 返回一个整数
    "_assert_scalar",  # 没有返回值
    "_dimI",  # 返回一个整数
    "_dimV",  # 返回一个整数
    "_has_same_storage_numel",  # 返回一个布尔值
    "_linalg_check_errors",  # 没有返回值
    "_local_scalar_dense",  # 返回一个标量
    "_nested_tensor_from_mask_left_aligned",  # 返回一个布尔值
    "_nnz",  # 返回一个整数
    "_use_cudnn_ctc_loss",  # 返回一个布尔值
    "_use_cudnn_ctc_loss.Tensor",  # 返回一个布尔值
    "_validate_compressed_sparse_indices",  # 没有返回值
    "allclose",  # 返回一个布尔值
    "dense_dim",  # 返回一个整数
    "equal",  # 返回一个布尔值
    "is_coalesced",  # 返回一个布尔值
    "is_pinned",  # 返回一个布尔值
    "is_same_size",  # 返回一个布尔值
    "is_set_to",  # 返回一个布尔值
    "q_per_channel_axis",  # 返回一个整数
    "q_scale",  # 返回一个浮点数
    "q_zero_point",  # 返回一个整数
    "qscheme",  # 返回一个 QScheme
    "record_stream",  # 没有返回值
    "sparse_dim",  # 返回一个整数
    "sym_constrain_range",  # 没有返回值
    "sym_constrain_range_for_size",  # 没有返回值
    "_nested_tensor_storage_offsets",  # 返回一个整数向量
    "_chunk_grad_outputs_efficient_attention",  # 返回一个布尔值
    "_fused_sdp_choice",  # 返回一个整数
    "_print",  # 没有返回值
    "_sink_tokens",  # 没有返回值
]
    "_nested_get_ragged_idx",  # 返回一个整数
# 一些无法正确分组的原地操作的列表
# 例如 polygamma 和 polygamma.out 都存在，但具有不同的前自变量（而 polygamma_ 没有）
# 我们应该修复此模式，以便可以正确分组，或者允许代码生成为此操作生成新的 functional/out= NativeFunctions
# （这将需要更改其重载名称以防止重载歧义）。
INPLACE_OPS_THAT_DONT_GET_GROUPED_PROPERLY = [
    "polygamma_"
]

# 将“相似” NativeFunctions 分组在一起的函数
# 例如 add.Tensor、add_.Tensor、add.out
# “相似”的 NativeFunctions 都应具有相同的 `signature()`，但具有不同的 SchemaKinds。
def pre_group_native_functions(
    native_functions: Sequence[NativeFunction],
) -> dict[FunctionSchema, dict[SchemaKind, NativeFunction]]:
    # 用于预先分组的原生函数字典，默认字典内嵌套字典
    pre_grouped_native_functions: dict[
        FunctionSchema, dict[SchemaKind, NativeFunction]
    ] = defaultdict(dict)
    # 遍历每个 NativeFunction
    for f in native_functions:
        # 获取已分组的函数字典
        d = pre_grouped_native_functions[f.func.signature()]
        # 确保同一 SchemaKind 的函数不会重复添加
        assert f.func.kind() not in d
        # 将函数添加到对应的 SchemaKind 中
        d[f.func.kind()] = f
    return pre_grouped_native_functions


# 给定基础函数重载名称，返回预期的 out= 变体重载名称
def get_expected_out_variant_overload_name(overload_name: str | None) -> str:
    return "out" if not overload_name else f"{overload_name}_out"


# 辅助函数：根据原地函数的 FunctionSchema，生成其对应的 out= 变体
# 示例：_add_relu_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)
# 变为：_add_relu.Scalar_out(Tensor self, Scalar other, Scalar alpha=1, *, Tensor(a!) out)
def self_to_out_signature(func: FunctionSchema) -> FunctionSchema:
    # 从原地模式生成一个 out= 模式
    assert func.kind() == SchemaKind.inplace
    assert func.arguments.self_arg is not None
    # 新的 out= 模式包括：
    # - 一个新的 out 参数，类型与 "func" 相同（但带有可变注释）
    # - 返回值（如果有）现在别名为 out= 参数，而不是 "func"
    # - 一个 "out" 重载名称
    return FunctionSchema(
        name=func.name.remove_inplace().with_overload(
            get_expected_out_variant_overload_name(func.name.overload_name)
        ),
        arguments=func.arguments.remove_self_annotation().with_out_args(
            [
                Argument(
                    name="out",
                    type=func.arguments.self_arg.argument.type,
                    default=None,
                    annotation=func.arguments.self_arg.argument.annotation,
                )
            ]
        ),
        returns=func.returns,
    )
#   _to_copy._out(Tensor self, *, bool non_blocking=False, MemoryFormat? memory_format=None,
#       Tensor(a!) out) -> Tensor(a!)
def functional_to_out_signature(func: FunctionSchema) -> FunctionSchema:
    # 从功能模式的函数架构生成一个 out= 的架构
    assert func.kind() == SchemaKind.functional

    new_returns, new_out_args = generate_out_args_from_schema(func)
    # 新的 out= 架构包括：
    # - 一个或多个带有相同类型的返回值的新 out 参数（但带有可变注释）
    # - 返回值现在与 out= 参数别名
    # - "_out" 重载名称
    return FunctionSchema(
        name=func.name.with_overload(
            get_expected_out_variant_overload_name(func.name.overload_name)
        ),
        arguments=func.arguments.signature().with_out_args(
            new_out_args,
        ),
        returns=tuple(new_returns),
    )


# Helper function: given a function schema, generate corresponding out arguments, also the updated return annotations.
def generate_out_args_from_schema(
    func: FunctionSchema,
) -> tuple[list[Return], list[Argument]]:
    # 更多的是一个健全性检查 - 我们对架构的现有限制应强制执行，即可变架构类型永远不会返回它们的可变参数。
    assert not any(
        r.annotation is not None and r.annotation.is_write for r in func.returns
    )

    tensorlike_rets = [r for r in func.returns if r.type.is_tensor_like()]
    assert len(tensorlike_rets) > 0

    used_annotations = concatMap(
        lambda a: [] if a.annotation is None else a.annotation.alias_set,
        func.arguments.flat_all,
    )
    valid_annotations = [
        x for x in "abcdefghijklmnopqrstuvwxyz" if x not in used_annotations
    ]

    all_rets_are_tensors = all(r.type == BaseType(BaseTy.Tensor) for r in func.returns)

    new_out_args: list[Argument] = []
    # 新返回值的最终结果是：
    # - 如果每个返回值都是普通张量，则新返回值 == 旧返回值，但添加了 out= 别名注释。
    # - 否则，out 参数不会出现在返回值中（如果有的话，我们只剩下非张量样式的返回值）。
    new_returns: list[Return] = []
    # 遍历函数的返回值列表，并用索引 i 和值 r 迭代其中的每一个返回值
    for i, r in enumerate(func.returns):
        # 检查返回值类型是否类似张量（tensor-like）
        if r.type.is_tensor_like():
            # 创建一个新的输出参数对象
            new_out = Argument(
                # 如果函数只有一个返回值，命名为 "out"，否则命名为 "out{i}"，其中 i 是索引
                name="out" if len(func.returns) == 1 else f"out{i}",
                # 设置新输出参数的类型为 r 的类型
                type=r.type,
                # 默认值设为 None
                default=None,
                # 解析有效注释字符串，并设置为新输出参数的注释
                annotation=Annotation.parse(f"{valid_annotations[i]}!"),
            )
            # 将新输出参数添加到新输出参数列表中
            new_out_args.append(new_out)
            
            # 如果所有的返回值都是张量，根据约定处理 out= 模式的返回值
            if all_rets_are_tensors:
                # 创建一个新的返回值对象，名称为 None，类型和注释与新输出参数相同
                new_ret = Return(
                    name=None,
                    type=new_out.type,
                    annotation=new_out.annotation
                )
                # 将新的返回值对象添加到新的返回值列表中
                new_returns.append(new_ret)
        else:
            # 如果返回值不是张量，直接将其添加到新的返回值列表中
            new_returns.append(r)
    
    # 返回处理后的新的返回值列表和新的输出参数列表
    return new_returns, new_out_args
# Helper function: given a mutable FunctionSchema, generate its corresponding out= variant
# Example before:
#   _fused_moving_avg_obs_fq_helper(Tensor self, Tensor observer_on, Tensor fake_quant_on, Tensor(a!) running_min, Tensor(b!) running_max, Tensor(c!) scale, Tensor(d!) zero_point, float averaging_const, int quant_min, int quant_max, int ch_axis, bool per_row_fake_quant=False, bool symmetric_quant=False) -> (Tensor output, Tensor mask)  # noqa: B950
# Example after:
#   _fused_moving_avg_obs_fq_helper._out(Tensor self, Tensor observer_on, Tensor fake_quant_on, Tensor(a!) running_min, Tensor(b!) running_max, Tensor(c!) scale, Tensor(d!) zero_point, float averaging_const, int quant_min, int quant_max, int ch_axis, bool per_row_fake_quant=False, bool symmetric_quant=False, *, Tensor(e!) out0, Tensor(f!) out1) -> (Tensor(e!), Tensor(f!))  # noqa: B950
def mutable_to_out_signature(func: FunctionSchema) -> FunctionSchema:
    # Generating an out= schema from a mutable schema.
    assert func.kind() == SchemaKind.mutable
    # Assert that the input function schema is mutable.

    # The new out= schema has:
    # - Any non-aliased tensor-like returns are converted to mutable, aliased out= arguments
    #   (if the argument is a tensor then we also return it for method chaining,
    #   otherwise we return nothing)
    # - an "out" overload name
    #
    # Note that:
    # (1) This also means that we can *only* generate an out= variant from a mutable schema
    #     if the mutable schema has at least one tensor-like non-aliasing return.
    # (2) The generated out= variant still has mutable positional arguments,
    #     but if necessary we could probably add another out= variant that also
    #     functionalizes the mutable arguments (a functional_out variant)

    # Generate the new returns and out= arguments from the input function schema.
    new_returns, new_out_args = generate_out_args_from_schema(func)

    # Return a new FunctionSchema instance with updated name, arguments, and returns.
    return FunctionSchema(
        name=func.name.remove_inplace().with_overload(
            get_expected_out_variant_overload_name(func.name.overload_name)
        ),
        arguments=func.arguments.with_out_args(new_out_args),
        returns=tuple(new_returns),
    )


# This function, given function of one SchemaKind, as well as a target SchemaKind,
# generates a new NativeFunction with the same properties, but using the target SchemaKind.
# We only actually generate functions for either functional or out= SchemaKinds.
# This function returns a tuple, with:
# - The generated NativeFunction
# - a dictionary of `BackendIndex` objects, describing which dispatch keys
#   we will generate kernels for, for the new NativeFunction.
#   Details are in the function, but we only generate composite kernels (in some cases) today.
def generate_function(
    f: NativeFunction, k: SchemaKind
) -> tuple[NativeFunction, dict[DispatchKey, dict[OperatorName, BackendMetadata]]]:
    from torchgen.api import cpp
    # Import cpp module from torchgen.api

    # Function implementation goes here, generating a new NativeFunction
    # with the specified SchemaKind and returning relevant metadata.
    # 如果操作类型为 SchemaKind.functional
    if k == SchemaKind.functional:
        # 断言：函数对象的类型不是 SchemaKind.functional
        assert f.func.kind() != SchemaKind.functional
        
        # 对于新的 "functional" NativeFunction，执行以下操作：
        # - 任何可变参数已经被转换为（不可变的）返回值。
        #   （如果一个可变参数不是返回值，它会被转换为返回值）
        # - 如果此操作有可变变体，则将 "_functional" 添加到基本名称的末尾。
        #   参见注释 [Overload Ambiguity With Functional Variants]
        # 在 signature() 中，默认的分组逻辑已经包含了这些操作，
        # 所以我们可以利用它（但是我们仍然需要保留返回值的名称）
        func = f.func.signature(keep_return_names=True).with_name(
            OperatorName(
                name=BaseOperatorName(
                    base=f.func.name.name.base,
                    inplace=False,
                    dunder_method=f.func.name.name.dunder_method,
                    # 参见注释 [Overload Ambiguity With Functional Variants]
                    functional_overload=f.func.kind() == SchemaKind.mutable,
                ),
                overload_name=f.func.name.overload_name,
            )
        )
    
    # 如果操作类型为 SchemaKind.out
    elif k == SchemaKind.out:
        # 我们主要生成 out= 操作是为了方便将 NativeFunction 分组成组，
        # 但是至少在今天，实际上没有好理由来真正使用它们。
        # 我们会为它们生成一个分派器条目，但不会为它们注册任何内核。
        if f.func.kind() == SchemaKind.inplace:
            func = self_to_out_signature(f.func)
        elif f.func.kind() == SchemaKind.mutable:
            func = mutable_to_out_signature(f.func)
        elif f.func.kind() == SchemaKind.functional:
            func = functional_to_out_signature(f.func)
        else:
            raise AssertionError(
                "我们只从 inplace、mutable 或 functional 变体生成 out= 函数"
            )
    
    # 如果既不是 functional 也不是 out=，则抛出错误
    else:
        raise AssertionError(
            "我们目前只生成 functional 或 out= NativeFunctions"
        )
    
    # 为 out= 生成的内核命名约定为：<op_name>_<overload_name>。
    # 这样做是为了消除具有相同名称但不同重载名称的运算符的歧义，
    # 例如 `randn.names_out` 和 `randn.generator_with_names_out`。
    kernel_name = (
        func.name.unambiguous_name()
        if func.kind() == SchemaKind.out
        else cpp.name(func)
    )
    
    # 如果函数对象具有 symint 特性，则在内核名称末尾添加 "_symint"
    if f.func.has_symint():
        kernel_name += "_symint"
    
    # 构建后端元数据字典，用于 DispatchKey.CompositeExplicitAutograd
    backend_metadata = {
        DispatchKey.CompositeExplicitAutograd: {
            func.name: BackendMetadata(
                kernel=kernel_name,
                structured=False,
                cpp_namespace=DEFAULT_KERNEL_NAMESPACE,
            )
        }
    }
    
    # 设置标签，包括 "generated"，以及从 f.tags 中的特定标签
    tags = {"generated"} | set(
        f.tags & {"nondeterministic_seeded", "view_copy", "pt2_compliant_tag"}
    )
    # 返回一个元组，包含以下两个元素：
    return (
        # 创建一个 NativeFunction 对象，使用给定的函数（func）和其他配置参数
        NativeFunction(
            func=func,
            use_const_ref_for_mutable_tensors=f.use_const_ref_for_mutable_tensors,
            # 这些生成的函数不打算对用户友好，不生成方法。
            variants={Variant.function},
            structured=False,
            structured_delegate=None,
            structured_inherits=None,
            precomputed=None,
            autogen=[],
            ufunc_inner_loop={},
            manual_kernel_registration=False,
            manual_cpp_binding=False,
            python_module=None,
            category_override=None,
            device_guard=False,
            device_check=DeviceCheckType.NoCheck,
            loc=f.loc,
            cpp_no_default_args=set(),
            is_abstract=f.is_abstract,
            has_composite_implicit_autograd_kernel=False,
            has_composite_implicit_autograd_nested_tensor_kernel=False,
            has_composite_explicit_autograd_kernel=True,
            has_composite_explicit_autograd_non_functional_kernel=False,
            # 每个生成的 NativeFunction 都会有一个 "generated" 标签，方便区分
            # 哪些 NativeFunction 对象并非直接来自 native_functions.yaml 文件。
            tags=tags,
            namespace=f.namespace,
        ),
        # 第二个元素是 backend_metadata 变量的值
        backend_metadata,
    )
# 这个函数负责添加生成的 NativeFunctions，这些函数在代码生成中并未明确出现。
# 你可以使用 torchgen 包检查完整的 NativeFunctions 列表，
# 方法是运行 torchgen.parse_native_yaml("aten/src/ATen/native/native_functions.yaml", "aten/src/ATen/native/tags.yaml")
# （也许我们应该为此提供一个友好的 API）

# 注意：这个函数会修改它的两个输入参数，
# 将新的 NativeFunctions / BackendMetadata 添加到它们中间
def add_generated_native_functions(
    rs: list[NativeFunction],
    indices: dict[DispatchKey, dict[OperatorName, BackendMetadata]],
) -> None:
    # 生成新的 NativeFunctions 的主要代码
    # 首先按照 schema 类型对 NativeFunctions 进行分组，
    # 然后检测缺失的函数并生成它们。
    pre_grouped_native_functions = pre_group_native_functions(rs)

    # 为缺少的操作生成一个 out= 变体
    if gets_out_variant:
        fn, metadata = generate_function(base_fn, SchemaKind.out)
        d[SchemaKind.out] = fn
        BackendIndex.grow_index(indices, metadata)
        rs.append(fn)

    # 生成一个 functional 变体，但前提是操作符已经有了 out= 变体
    # （Functional 变体只有在我们可以对这些变体进行分组时才有用，
    # 而这只有在它们有 out= 变体时才能做到）
    if not has_functional and (has_out or gets_out_variant):
        fn, metadata = generate_function(base_fn, SchemaKind.functional)
        d[SchemaKind.functional] = fn
        BackendIndex.grow_index(indices, metadata)
        rs.append(fn)


# 给定一个函数和对应于该函数输出的变量名，
# 收集所有未别名化的单独返回值
def gather_nonaliased_inner_rets(func: FunctionSchema, out_var: str) -> list[str]:
    aliased_rets = func.aliased_return_names()
    non_aliased_names = []
    is_out_var_a_tuple = len(func.returns) > 1
    for i, r in enumerate(aliased_rets):
        if r is None:
            non_aliased_names.append(
                f"std::get<{i}>({out_var})" if is_out_var_a_tuple else out_var
            )
    return non_aliased_names


# 给定一个函数，并且给出对应于该函数输出的变量的名称，
# 收集所有未别名化的内部返回值
def gather_nonaliased_inner_rets(func: FunctionSchema, out_var: str) -> list[str]:
    aliased_rets = func.aliased_return_names()
    non_aliased_names = []
    is_out_var_a_tuple = len(func.returns) > 1
    for i, r in enumerate(aliased_rets):
        if r is None:
            non_aliased_names.append(
                f"std::get<{i}>({out_var})" if is_out_var_a_tuple else out_var
            )
    return non_aliased_names


# 给定返回值元组和名称列表，生成返回语句的字符串表示
def return_str(rets: tuple[Return, ...], names: list[str]) -> str:
    assert len(rets) == len(names)
    if len(rets) == 0:
        return ""
    elif len(rets) == 1:
        return f"return {names[0]};"
    else:
        return f"return {dispatcher.returns_type(rets).cpp_type()}({', '.join(names)});"


# 生成功能性内核，以其 inplace.mutable 对应项表示。
# 我们只针对“生成的”NativeFunctions执行此操作
@with_native_function
# 为给定的 NativeFunctionsGroup 生成复合函数核心代码。返回生成的函数字符串或空（如果不适用）。
def gen_composite_functional_kernel(g: NativeFunctionsGroup) -> str | None:
    # 我们仅为代码生成的 NativeFunctions 生成这些核心代码
    if "generated" not in g.functional.tags:
        return None
    
    # 对于生成的操作，始终以非生成操作为基础编写核心代码。
    if g.inplace is not None and "generated" not in g.inplace.tags:
        target_f = g.inplace
    elif g.mutable is not None and "generated" not in g.mutable.tags:
        target_f = g.mutable
    else:
        # 我们应该保证有一个有效的 inplace/mutable 变体可供调用。
        # 参见注释: [Mutable Ops Not Using Functionalization]
        raise AssertionError(str(g.functional.func))

    # 创建调度器签名对象，用于处理函数签名
    sig = DispatcherSignature(g.functional.func)
    target_sig = DispatcherSignature(target_f.func)

    # 上下文用于存储生成代码中的绑定和表达式
    context: list[Binding | Expr] = []
    # 用于存储克隆的可变输入参数的代码
    clone_mutable_inputs = []
    # 存储已克隆返回值名称的列表
    cloned_return_names = []

    # 不能直接将所有参数从 functional 操作传递到 mutating 操作。
    # 需要检查 mutating 操作的哪些输入是可变的，并首先对这些输入进行克隆。
    for a_curr, a_tgt in zip(
        dispatcher.jit_arguments(g.functional.func),
        dispatcher.jit_arguments(target_f.func),
    ):
        if a_tgt.annotation is not None and a_tgt.annotation.is_write:
            clone_mutable_inputs.append(
                f"auto {a_curr.name}_clone = clone_arg({a_curr.name});"
            )
            context.append(
                Expr(
                    expr=f"{a_curr.name}_clone",
                    type=dispatcher.argument_type(a_curr, binds=a_curr.name),
                )
            )
            # 不变量: 在内部可变操作中，可变参数始终作为 functional 操作的返回值。
            cloned_return_names.append(f"{a_curr.name}_clone")
        else:
            context.append(dispatcher.argument(a_curr))
    
    # 将上下文转换为目标函数的参数表达式字符串
    exprs = ", ".join([e.expr for e in translate(context, target_sig.arguments())])

    # 输出名称
    out_name = "output"
    # 如果目标函数有返回值，则准备赋值操作
    maybe_assign = f"auto {out_name} = " if len(target_f.func.returns) > 0 else ""
    # 获取不带别名的内部返回值名称列表
    inner_return_names = gather_nonaliased_inner_rets(target_f.func, out_name)
    # 生成返回字符串
    ret_str = return_str(
        g.functional.func.returns, inner_return_names + cloned_return_names
    )

    # 生成可变输入参数克隆的代码字符串
    clone_mutable_inputs_str = "\n".join(clone_mutable_inputs)
    
    # 构建最终生成的函数字符串
    return f"""
{sig.defn(name=sig.name() + ("_symint" if g.out.func.has_symint() else ""))} {{
  {clone_mutable_inputs_str}
  {maybe_assign}at::_ops::{target_f.func.name.unambiguous_name()}::call({exprs});
  {ret_str}
}}
"""


# 为 functional 的操作生成 out= 内核，基于它们的 functional 对应操作。
# 我们仅对 "generated" 的 NativeFunctions 这样做。
@with_native_function
def gen_composite_out_kernel(g: NativeFunctionsGroup) -> str | None:
    # 我们仅为代码生成的 NativeFunctions 生成这些核心代码
    if "generated" not in g.out.tags:
        return None
    # 创建一个调度签名对象，用于分析 g.out.func 的签名信息
    sig = DispatcherSignature(g.out.func)
    # 创建一个目标调度签名对象，用于分析 g.functional.func 的签名信息
    target_sig = DispatcherSignature(g.functional.func)

    # 将 g.out.func 和 g.functional.func 的参数转换为表达式字符串列表
    exprs = ", ".join(
        [e.expr for e in translate(sig.arguments(), target_sig.arguments())]
    )

    # 初始化一个空列表，用于存储生成的代码片段
    copy_outs = []
    # 设置输出名称
    out_name = "tmp_output"
    # 遍历 g.out.func 的输出参数列表
    for i, out_arg in enumerate(g.out.func.arguments.out):
        # 根据返回值数量决定返回值的命名方式
        functional_return_name = (
            out_name
            if len(g.functional.func.returns) == 1
            else f"std::get<{i}>({out_name})"
        )
        # 生成调整输出大小和复制参数的代码片段，并添加到列表中
        copy_outs.append(
            f"""\
  resize_out_helper({out_arg.name}, {functional_return_name});
  copy_arg({out_arg.name}, {functional_return_name});"""
        )

    # 初始化一个空列表，用于存储最终的返回值列表
    rets = []
    # 遍历 g.out.func 的别名返回值列表
    for i, ret_name in enumerate(g.out.func.aliased_return_names()):
        # 如果别名不为空，则直接将其添加到返回值列表中
        if ret_name is not None:
            rets.append(ret_name)
        else:
            # 否则根据返回值数量决定返回值的命名方式，并添加到返回值列表中
            functional_return_name = (
                out_name
                if len(g.functional.func.returns) == 1
                else f"std::get<{i}>({out_name})"
            )
            rets.append(functional_return_name)

    # 将复制输出参数的代码片段列表连接成字符串，用于后续返回
    copy_outs_str = "\n".join(copy_outs)

    # 返回一个生成的字符串，作为内核函数的名称，需遵循 generate_function() 中定义的命名约定
    return f"""
{
  # 构造函数签名，基于输出函数名和是否有符号整数进行调整
  sig.defn(name=g.out.func.name.unambiguous_name() + ("_symint" if g.out.func.has_symint() else ""))
} {{
  # 调用特定功能模块中的函数，并将结果赋给变量 {out_name}
  auto {out_name} = at::_ops::{g.functional.func.name.unambiguous_name()}::call({exprs});
  # 复制输出变量的字符串表示，可能为空
  {copy_outs_str}
  # 生成返回语句，基于输出函数的返回类型和返回值列表
  {return_str(g.out.func.returns, rets)}
}}
```