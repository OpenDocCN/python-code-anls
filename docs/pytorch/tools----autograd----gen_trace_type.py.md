# `.\pytorch\tools\autograd\gen_trace_type.py`

```
# Note [Manual Backend kernels]
# 对于这些操作，我们希望手动注册到分发键 Backend，并跳过所有键的代码生成注册，以前是 Backend 的。
# 对于代码生成，这意味着：
#   - 下面的操作必须与 native_functions.yaml 中 manual_kernel_registration=True 的操作匹配，
#     在那里我们跳过了代码生成的后端内核
#   - 所有下面列出的操作都是 MANUAL_AUTOGRAD 的一部分，用于跳过代码生成的 Autograd 内核注册
#   - 所有下面列出的操作都是 MANUAL_TRACER 的一部分，用于跳过代码生成的 Tracer 内核注册
# 注：我们仍然注册到分发键 Profiler 以供这些操作使用，现在保持不变。
# 您可以在 torch/csrc/autograd/VariableTypeManual.cpp 中找到手动注册。

MANUAL_BACKEND = {
    "options",
    "data",
    "set_data",
    "is_leaf",
    "output_nr",
    "_version",
    "retain_grad",
    "_backward",
    "requires_grad_",
}

# 对于这些操作，我们希望跳过代码生成的 Autograd 和 Tracer 键注册。
# 您可以在 torch/csrc/autograd/VariableTypeManual.cpp 中找到手动注册。
MANUAL_AUTOGRAD_AND_TRACER = {
    "resize_",
    "resize_as_",
    "detach",
    "detach_",
    "copy_",
    "_fw_primal",
    "_make_dual",
}

# 目前 MANUAL_AUTOGRAD 和 MANUAL_TRACER 共享相同的操作集：
#   union(MANUAL_BACKEND, MANUAL_AUTOGRAD_AND_TRACER)
# 您可以在 torch/csrc/autograd/VariableTypeManual.cpp 中找到手动注册。

MANUAL_AUTOGRAD = MANUAL_TRACER = MANUAL_BACKEND | MANUAL_AUTOGRAD_AND_TRACER

# 这些函数我们不希望用于跟踪记录，因为我们始终希望跟踪它们的组成部分。
# 这是临时的方法，代替了正确的作用域，在那里后续编译步骤可以按需请求展开。
# 只有具体的 ATen 方法可以通过这种方式禁用；否则将不起作用。
DONT_RECORD_TRACE = {
    "convolution",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "lstm_cell",
    "gru_cell",
    "rnn_tanh_cell",
    "rnn_relu_cell",
    # FIXME: figure out a better way when we support sparse tensors in jit
    "_coalesced",
}


def should_trace(f: NativeFunction) -> bool:
    # 目前涉及 Storage 或 Type 的操作无法跟踪
    if any(
        str(arg.type) in {"Storage", "Type", "ConstQuantizerPtr"}
        for arg in f.func.schema_order_arguments()
    ):
        return False
    # 我们无法跟踪没有任何 Tensor 或 TensorList 返回值的函数
    if not any(r.type.is_tensor_like() for r in f.func.returns):
        return False
    # 检查函数对象的名称是否不在 DONT_RECORD_TRACE 列表中
    return f.func.name.name.base not in DONT_RECORD_TRACE
# 定义一个代码模板，用于生成条件语句的字符串表示
SELECT = CodeTemplate(
    """\
if (${cond}) {
  ${true}
} else {
  ${false}
}
"""
)

# 定义一个代码模板，用于生成操作名称的字符串表示
OP_NAME = CodeTemplate(
    """\
op_name = c10::Symbol::fromQualString("aten::${trace_name}");
"""
)

# 这些函数在追踪中有其名称更改的记录
RENAME_TRACE = {
    "zero": "zeros_like",  # 将 aten::zero_ 替换为 aten::zeros_like
    "fill": "full_like",   # 将 aten::fill_ 替换为 aten::full_like
}

# 格式化操作名称的函数，返回操作名称的字符串表示
def format_trace_op_name(f: NativeFunction) -> str:
    # TODO: byte-for-byte compatible with old codegen behavior - should clean up
    if (
        f.func.kind() in (SchemaKind.functional, SchemaKind.out)
        or f.func.name.name.dunder_method
    ):
        # *_out 函数的特殊情况：JIT 中的原位和非原位操作使用相同的名称重载
        trace_name = str(f.func.name.name)
        trace_name = RENAME_TRACE.get(trace_name, trace_name)
        return OP_NAME.substitute(trace_name=trace_name)

    # 否则，这是一个原位操作，需要生成原位和非原位版本
    outplace_trace_name = f.func.name.name.base
    inplace_trace_name = cpp.name(f.func)
    outplace_trace_name = RENAME_TRACE.get(outplace_trace_name, outplace_trace_name)
    inplace_trace_name = RENAME_TRACE.get(inplace_trace_name, inplace_trace_name)

    return SELECT.substitute(
        cond="tracer_state->force_outplace",
        true=OP_NAME.substitute(trace_name=outplace_trace_name),
        false=OP_NAME.substitute(trace_name=inplace_trace_name),
    )

# 添加追踪输入的代码模板，生成节点的输入描述字符串表示
ADD_TRACE_INPUT = CodeTemplate("""jit::tracer::addInputs(node, "${name}", ${input});""")

# 格式化追踪输入的函数，返回输入的描述字符串表示
def format_trace_inputs(f: NativeFunction) -> str:
    # 内部函数：根据参数类型分发生成追踪输入的代码
    def dispatch_trace_input(arg: Argument | TensorOptionsArguments) -> Sequence[str]:
        if isinstance(arg, TensorOptionsArguments):
            name = "options"
            return [
                ADD_TRACE_INPUT.substitute(
                    name=name, input="c10::optTypeMetaToScalarType(options.dtype_opt())"
                ),
                ADD_TRACE_INPUT.substitute(name=name, input="options.layout()"),
                ADD_TRACE_INPUT.substitute(name=name, input="options.device()"),
                ADD_TRACE_INPUT.substitute(name=name, input="options.pinned_memory()"),
            ]
        else:
            name = arg.name
            if str(arg.type) == "Tensor?[]":
                return [f'jit::tracer::addInputs(node, "{name}", {name});']
            else:
                return [ADD_TRACE_INPUT.substitute(name=name, input=name)]

    # 获取函数的参数列表
    args: list[Argument | TensorOptionsArguments] = list(
        f.func.schema_order_arguments()
    )
    # 如果函数 f 是一个 *_out 函数：
    if f.func.is_out_fn():
        # *_out 函数将结果作为单独的参数，但我们不直接跟踪该参数。
        # 相反，我们跟踪其 TensorOptions。
        
        # 需要从要跟踪的参数列表中移除 out 参数。
        num_out_args = len(f.func.arguments.out)
        args = args[:-num_out_args]

    # 生成追踪输入的迭代器，其中 args 是函数的输入参数列表
    trace_inputs = itertools.chain.from_iterable(
        dispatch_trace_input(arg) for arg in args
    )

    # 如果函数 f 是一个 *_out 函数：
    if f.func.is_out_fn():
        # 对于 *_out 函数，针对结果参数的处理与 inplace/outplace 不同。

        # 如果是 inplace 操作：
        inplace = [
            ADD_TRACE_INPUT.substitute(
                name=f.func.arguments.out[i].name, input=f.func.arguments.out[i].name
            )
            for i in range(num_out_args)
        ]

        # 如果是 outplace 操作：
        # 除非函数是一个工厂方法，否则不执行任何操作。
        # 工厂方法比较特殊，因为其 out-of-place 重载需要额外的 TensorOptions 参数，
        # 这在 _out 函数中是缺少的。
        has_tensor_return = any(r.type.is_tensor_like() for r in f.func.returns)
        has_tensor_input_arg = any(
            a.type.is_tensor_like() for a in f.func.arguments.flat_non_out
        )
        is_factory_method = f.category_override == "factory" or (
            has_tensor_return and not has_tensor_input_arg
        )

        # HACK: 保留旧的代码生成行为 - 旧的代码生成为整个操作族设置了 `is_factory_method` 标志，
        # 如果其中任何一个是工厂方法。对于大多数情况下，整个操作族的所有操作确实都是工厂方法 -
        # 'normal' 是唯一的例外。因此，我们在这里特别处理它，以避免克隆旧逻辑。
        if f.func.name.name.base == "normal":
            is_factory_method = True

        # 如果是工厂方法：
        if is_factory_method:
            outplace = [
                ADD_TRACE_INPUT.substitute(
                    name="out",
                    input="c10::optTypeMetaToScalarType(out.options().dtype_opt())",
                ),
                ADD_TRACE_INPUT.substitute(name="out", input="out.options().layout()"),
                ADD_TRACE_INPUT.substitute(name="out", input="out.options().device()"),
                ADD_TRACE_INPUT.substitute(
                    name="out", input="out.options().pinned_memory()"
                ),
            ]
        else:
            outplace = []

        # 将生成的 inplace 或 outplace 的追踪输入添加到 trace_inputs 中。
        trace_inputs = itertools.chain(
            trace_inputs,
            [
                SELECT.substitute(
                    cond="tracer_state->force_outplace",
                    true="\n".join(outplace),
                    false="\n".join(inplace),
                )
            ],
        )

    # 将所有追踪输入连接成一个字符串并返回。
    return "\n".join(trace_inputs)
# 定义了一个用于重命名追踪参数的映射表，根据函数名将其替换为对应的追踪参数设置
RENAME_TRACE_ADD_ARGS = {
    "fill": """\
    jit::tracer::addInputs(node, "options", c10::optional<ScalarType>());
    jit::tracer::addInputs(node, "options", layout_or_default(c10::nullopt));
    jit::tracer::addInputs(node, "options", device_or_default(c10::nullopt));
    jit::tracer::addInputs(node, "options", pinned_memory_or_default(c10::nullopt));
    c10::optional<MemoryFormat> memory_format = c10::MemoryFormat::Preserve;
    jit::tracer::addInputs(node, "memory_format", memory_format);
""",
    "zero": """\
    jit::tracer::addInputs(node, "options", c10::optional<ScalarType>());
    jit::tracer::addInputs(node, "options", layout_or_default(c10::nullopt));
    jit::tracer::addInputs(node, "options", device_or_default(c10::nullopt));
    jit::tracer::addInputs(node, "options", pinned_memory_or_default(c10::nullopt));
    c10::optional<MemoryFormat> memory_format = c10::MemoryFormat::Preserve;
    jit::tracer::addInputs(node, "memory_format", memory_format);
""",
}

# 定义了一个模板，用于生成确保函数在执行非原位操作时是唯一的标记
INPLACE_GUARD = CodeTemplate(
    """\
jit::tracer::ensureUniqueIfOutOfPlaced("${name}", ${mutable_input});
"""
)

# 定义了一个模板，用于在记录追踪前准备追踪节点的代码
PRE_RECORD_TRACE = CodeTemplate(
    """\
torch::jit::Node* node = nullptr;
std::shared_ptr<jit::tracer::TracingState> tracer_state;
if (jit::tracer::isTracing()) {
  tracer_state = jit::tracer::getTracingState();
  at::Symbol op_name;
  ${set_op_name}
  node = tracer_state->createNode(op_name, /*num_outputs=*/0);
  jit::tracer::recordSourceLocation(node);
  ${add_trace_inputs}
  tracer_state->insertNode(node);
  ${inplace_guard}
  jit::tracer::setTracingState(nullptr);
}
"""
)

def format_prerecord_trace(f: NativeFunction) -> str:
    if not should_trace(f):
        return ""

    # TODO: 清理旧代码生成行为
    # 检查函数是否是就地操作，并且不是双下划线方法（如 __add__）
    is_inplace = (
        f.func.kind() in (SchemaKind.inplace, SchemaKind.out)
        and not f.func.name.name.dunder_method
    )
    # 如果是就地操作，获取相应函数名的追踪参数设置，否则为空字符串
    add_args = (
        RENAME_TRACE_ADD_ARGS.get(f.func.name.name.base, "") if is_inplace else ""
    )
    # 如果有额外的参数需要添加到输入中，根据条件构造额外的输入字符串
    additional_inputs = (
        SELECT.substitute(
            cond="tracer_state->force_outplace",  # 条件语句，用于选择是否强制使用outplace方式
            true=add_args,  # 如果条件为真，则使用add_args作为额外输入
            false="",  # 如果条件为假，则额外输入为空字符串
        )
        if add_args  # 如果add_args为真，则执行条件表达式
        else ""  # 如果add_args为假，则直接返回空字符串
    )

    # 构造预记录跟踪信息的字符串，包括操作名称、跟踪输入和额外的输入
    return PRE_RECORD_TRACE.substitute(
        set_op_name=format_trace_op_name(f),  # 设置操作名称为格式化后的跟踪操作名称
        add_trace_inputs=format_trace_inputs(f) + additional_inputs,  # 设置跟踪输入为格式化后的跟踪输入加上额外的输入
        inplace_guard=INPLACE_GUARD.substitute(
            name=cpp.name(f.func),  # 如果是inplace操作，设置inplace保护名称为C++名称格式化后的函数名称
            mutable_input=(f.func.arguments.out[0].name  # 如果函数有输出参数，设置可变输入为输出的第一个参数名称
                           if f.func.arguments.out
                           else "self"),  # 如果没有输出参数，则设置可变输入为"self"
        )
        if is_inplace  # 如果是inplace操作，则执行inplace保护的条件语句
        else "",  # 如果不是inplace操作，则返回空字符串作为inplace保护
    )
# 定义代码模板，用于生成跟踪记录的代码块
POST_RECORD_TRACE = CodeTemplate(
    """\
if (tracer_state) {
  jit::tracer::setTracingState(std::move(tracer_state));
  ${add_trace_outputs}
}
"""
)


def format_postrecord_trace(f: NativeFunction) -> str:
    # 如果不需要跟踪该函数，则返回空字符串
    if not should_trace(f):
        return ""

    # 对于 outplacing 操作，需要特殊处理 *_out 重载，将输出参数移动到返回值中
    if f.func.is_out_fn():
        # 获取所有输出参数的名称列表
        output_names_outplace = [arg.name for arg in f.func.arguments.out]
        output_names_inplace = cpp.return_names(f)

        # 代码大小优化：通常情况下两个变体的返回值是相同的
        if output_names_outplace == output_names_inplace:
            # 生成将输出参数添加到跟踪记录的代码块
            outputs = [
                f"jit::tracer::addOutput(node, {n});" for n in output_names_outplace
            ]
            return POST_RECORD_TRACE.substitute(add_trace_outputs=outputs)

        # 生成根据条件选择不同输出参数添加到跟踪记录的代码块
        selection = SELECT.substitute(
            cond="force_outplace",
            true="\n".join(
                f"jit::tracer::addOutput(node, {n});" for n in output_names_outplace
            ),
            false="\n".join(
                f"jit::tracer::addOutput(node, {n});" for n in output_names_inplace
            ),
        )
        return POST_RECORD_TRACE.substitute(add_trace_outputs=selection)
    else:
        # 对于非 outplacing 操作，获取返回值的名称列表
        output_names = cpp.return_names(f)
        # 生成将输出参数添加到跟踪记录的代码块
        outputs = [f"jit::tracer::addOutput(node, {n});" for n in output_names]
        return POST_RECORD_TRACE.substitute(add_trace_outputs=outputs)


def tie_return_values(f: NativeFunction) -> str:
    # 如果函数只有一个返回值，则返回该返回值的名称或默认名称 "result"
    if len(f.func.returns) == 1:
        return f'auto {f.func.returns[0].name or "result"}'
    # 否则，返回所有返回值名称的列表
    names = cpp.return_names(f)
    return f'auto [{", ".join(names)}]'


def get_return_value(f: NativeFunction) -> str:
    # 获取所有返回值的名称列表
    names = cpp.return_names(f)
    # 如果函数只有一个返回值，则返回该返回值的名称
    if len(f.func.returns) == 1:
        return names[0]
    # 如果函数的返回类型为 out 类型，则返回所有返回值名称的元组
    if f.func.kind() == SchemaKind.out:
        return f'std::forward_as_tuple({", ".join(names)})'
    else:
        # 否则，生成移动所有返回值的代码块
        moved = ", ".join(f"std::move({name})" for name in names)
        return f"std::make_tuple({moved})"


# 定义代码模板，用于生成跟踪调度的代码块
TRACE_DISPATCH = CodeTemplate(
    """\
${assign_return_values}at::_ops::${unambiguous_name}::redispatch(${unpacked_args});"""
)


def emit_trace_body(f: NativeFunction) -> list[str]:
    # 初始化跟踪体列表
    trace_body: list[str] = []

    # 向跟踪体列表添加预记录跟踪的代码块
    trace_body.append(format_prerecord_trace(f))

    # 获取调度器签名和表达式列表
    dispatcher_sig = DispatcherSignature.from_schema(f.func)
    dispatcher_exprs = dispatcher_sig.exprs()

    # 生成将跟踪调度参数解包后传递给 redispatch 函数的代码块
    dispatch_key_set = "ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer)"
    redispatch_args = ", ".join([dispatch_key_set] + [a.expr for a in dispatcher_exprs])
    # 根据函数的类型和是否有返回值，决定赋值给assign_return_values的字符串
    assign_return_values = (
        f"{tie_return_values(f)} = "
        if f.func.kind() in [SchemaKind.functional, SchemaKind.mutable]
        and f.func.returns
        else ""
    )

    # 注意，这里调用了慢速的手动CPP绑定操作的分派变体。
    # 我们可能可以更努力地确保调用快速变体，但性能收益可能很小。
    trace_body.append(
        TRACE_DISPATCH.substitute(
            assign_return_values=assign_return_values,  # 插入assign_return_values变量的值
            unambiguous_name=f.func.name.unambiguous_name(),  # 使用函数的不含歧义的名称
            unpacked_args=redispatch_args,  # 插入redispatch_args变量的值
        )
    )

    # 向追踪体(trace_body)添加格式化后的记录后追踪
    trace_body.append(format_postrecord_trace(f))

    # 如果函数有返回值，则向追踪体添加返回值的语句
    if f.func.returns:
        trace_body.append(f"return {get_return_value(f)};")

    # 返回填充完整的追踪体(trace_body)
    return trace_body
# 定义一个代码模板，用于生成函数的方法定义，包括返回类型、类型封装名和形式参数
METHOD_DEFINITION = CodeTemplate(
    """\
${return_type} ${type_wrapper_name}(${formals}) {
  ${type_definition_body}
}
"""
)


# 根据给定的 NativeFunction 对象生成类型封装名
def type_wrapper_name(f: NativeFunction, key: str = "Default") -> str:
    if f.func.name.overload_name:
        # 如果存在重载名称，则生成带有重载名称的函数名
        name = f"{cpp.name(f.func)}_{f.func.name.overload_name}"
    else:
        # 否则直接使用函数名生成类型封装名
        name = cpp.name(f.func)

    # 如果 key 不是 "Default"，则在函数名后追加 key
    if key != "Default":
        name = name + f"_{key}"
    return name


# 根据 NativeFunction 对象生成方法定义的字符串表示
@with_native_function
def method_definition(f: NativeFunction) -> str:
    assert cpp.name(f.func) not in MANUAL_TRACER

    # 构建形式参数列表
    formals = ", ".join(
        # 生成的代码跟踪内核直接通过该内核传递和重新计算调度键，以提高性能。
        # 详见“注释 [Plumbing Keys Through The Dispatcher]”。
        ["c10::DispatchKeySet ks"]
        + [
            f'{cpp.argument_type(a, binds="__placeholder__", symint=True).cpp_type()} {a.name}'
            for a in f.func.schema_order_arguments()
        ]
    )

    # 使用 METHOD_DEFINITION 模板替换并生成方法定义字符串
    return METHOD_DEFINITION.substitute(
        return_type=cpp.returns_type(f.func.returns, symint=True).cpp_type(),
        type_wrapper_name=type_wrapper_name(f),
        formals=formals,
        type_definition_body=emit_trace_body(f),  # 调用 emit_trace_body 生成类型定义的主体部分
    )


# 定义一个代码模板，用于生成函数的方法注册代码
WRAPPER_REGISTRATION = CodeTemplate(
    """\
m.impl("${name}",
       TORCH_FN(${class_type}::${type_wrapper_name})
);
"""
)


# 根据给定的 NativeFunction 对象生成方法注册的字符串表示
@with_native_function
def method_registration(f: NativeFunction) -> str:
    assert cpp.name(f.func) not in MANUAL_TRACER

    # 使用 WRAPPER_REGISTRATION 模板替换并生成方法注册字符串
    return WRAPPER_REGISTRATION.substitute(
        name=f.func.name,
        type_wrapper_name=type_wrapper_name(f),
        class_type="TraceType",  # 类型为 TraceType
    )


# 生成跟踪类型函数的实现
def gen_trace_type_func(fn: NativeFunction) -> dict[str, list[str]]:
    return {
        "ops_headers": [f"#include <ATen/ops/{fn.root_name}_ops.h>"],  # 包含相关操作的头文件路径
        "trace_method_definitions": [method_definition(fn)],  # 生成方法定义的列表
        "trace_wrapper_registrations": [method_registration(fn)],  # 生成方法注册的列表
    }


# 生成跟踪类型
def gen_trace_type(
    out: str, native_functions: list[NativeFunction], template_path: str
) -> None:
    # 注意：请参阅 VariableType.cpp 顶部的“注释 [Sharded File]”以了解生成文件的分片情况。
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)  # 创建文件管理器实例
    fm.write_sharded(
        "TraceType.cpp",  # 将内容写入文件名为 "TraceType.cpp" 的文件中
        [fn for fn in native_functions if cpp.name(fn.func) not in MANUAL_TRACER],  # 从 native_functions 中筛选出不在 MANUAL_TRACER 中的函数名列表
        key_fn=lambda fn: fn.root_name,  # 使用函数 fn 的根名称作为键函数
        base_env={
            "generated_comment": "@"
            + f"generated from {fm.template_dir_for_comments()}/TraceType.cpp",  # 设置生成的注释内容，包含文件模板目录的路径信息
        },
        env_callable=gen_trace_type_func,  # 使用 gen_trace_type_func 函数生成环境变量
        num_shards=5,  # 设置使用的分片数目为 5
        sharded_keys={
            "ops_headers",  # 指定分片的关键字为 "ops_headers"
            "trace_method_definitions",  # 指定分片的关键字为 "trace_method_definitions"
            "trace_wrapper_registrations",  # 指定分片的关键字为 "trace_wrapper_registrations"
        },
    )
```