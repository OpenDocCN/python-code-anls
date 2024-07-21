# `.\pytorch\torch\_functorch\aot_autograd.py`

```py
# 忽略 mypy 的类型检查错误
# 导入 itertools 库，用于创建迭代器的工具函数
import itertools
# 从 contextlib 库导入上下文管理器和空上下文管理器
from contextlib import contextmanager, nullcontext
# 从 functools 库导入 partial 和 wraps 装饰器
from functools import partial, wraps
# 导入类型提示模块中的类型定义
from typing import Any, Callable, Dict, List, Optional, Tuple
# 从 unittest.mock 库导入 patch 函数，用于模拟对象
from unittest.mock import patch

# 导入 PyTorch 库
import torch
# 从 torch.nn 模块导入神经网络模块
import torch.nn as nn
# 导入 PyTorch 内部工具模块 _pytree 和 dlpack
import torch.utils._pytree as pytree
import torch.utils.dlpack
# 从 torch 模块导入 Tensor 类型
from torch import Tensor
# 导入 RNG 相关的模块，用于随机数生成和状态追踪
from torch._decomp.decompositions_for_rng import PhiloxStateTracker, rng_decompositions
# 导入 Python 调度相关的模块
from torch._dispatch.python import enable_python_dispatcher
# 导入动态图自动微分相关的模块
from torch._dynamo import compiled_autograd
# 从 dynamo.utils 模块导入动态图相关的工具函数和装饰器
from torch._dynamo.utils import dynamo_timed, preserve_rng_state
# 导入保护模式相关的模块，用于检测假张量模式
from torch._guards import detect_fake_mode
# 导入张量子类相关的模块和类型
from torch._subclasses import FakeTensor, FakeTensorMode
# 导入 FX 实验性代理张量相关的模块
from torch.fx.experimental.proxy_tensor import make_fx
# 导入 FX 实验性符号形状相关的模块
from torch.fx.experimental.symbolic_shapes import ShapeEnv
# 从 torch.utils 模块导入 Python 调度相关的函数
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
# 从当前目录的 config 模块导入配置
from . import config
# 从 _aot_autograd.autograd_cache 模块导入 AOTAutogradCache 类和相关函数（禁止 F401 警告）
from ._aot_autograd.autograd_cache import (
    AOTAutogradCache,
    autograd_cache_key,
)

# 从 _aot_autograd.collect_metadata_analysis 模块导入分析元数据收集的函数（禁止 F401 警告）
from ._aot_autograd.collect_metadata_analysis import (
    run_functionalized_fw_and_collect_metadata,
)
# 从 _aot_autograd.functional_utils 模块导入功能性工具函数（禁止 F401 警告）
from ._aot_autograd.functional_utils import (
    _check_if_mutation_can_be_in_graph,
    are_all_mutations_hidden_from_autograd,
    are_all_mutations_under_no_grad_or_inference_mode,
    assert_functional_graph,
    from_fun,
    gen_alias_from_base,
    has_data_mutation,
    has_metadata_mutation,
    is_fun,
    sync_functional_tensor,
    to_fun,
)
# 从 _aot_autograd.input_output_analysis 模块导入输入输出分析相关的函数（禁止 F401 警告）
from ._aot_autograd.input_output_analysis import (
    _tensors_definitely_do_not_overlap,
    compute_overlapping_inputs,
    create_graph_signature,
    create_synthetic_base_metadata,
    remove_dupe_metadata,
)
# 从 _aot_autograd.jit_compile_runtime_wrappers 模块导入 JIT 编译运行时包装器相关的函数（禁止 F401 警告）
from ._aot_autograd.jit_compile_runtime_wrappers import (
    aot_dispatch_autograd,
    aot_dispatch_base,
    aot_dispatch_export,
)
# 从 _aot_autograd.logging_utils 模块导入日志工具相关的函数（禁止 F401 警告）
from ._aot_autograd.logging_utils import (
    callback_set,
    describe_input,
    format_guard_bug_msg,
    get_aot_compilation_context,
    get_aot_graph_name,
    get_graph_being_compiled,
    graph_being_compiled,
    model_name,
    nth_graph,
    set_model_name,
    setup_stacktrace_preservation_hooks,
    track_graph_compiling,
)
# 从 _aot_autograd.runtime_wrappers 模块导入运行时包装器相关的类（禁止 F401 警告）
from ._aot_autograd.runtime_wrappers import (
    AOTDedupeWrapper,
    AOTSyntheticBaseWrapper,
)
# 从 _aot_autograd.schemas 模块导入相关的类和类型（禁止 F401 警告）
from ._aot_autograd.schemas import (
    AOTConfig,
    BackwardSignature,
    FQN,
    GraphInputName,
    GraphOutputName,
    GraphSignature,
    InputAliasInfo,
    MutationType,
    OutputAliasInfo,
    OutputType,
    SubclassCreationMeta,
    SubclassMeta,
    TensorAlias,
    ViewAndMutationMeta,
)
# 从 _aot_autograd.subclass_utils 模块导入子类相关的工具函数（禁止 F401 警告）
from ._aot_autograd.subclass_utils import (
    create_metadata_for_subclass,
    requires_subclass_dispatch,
    unwrap_tensor_subclasses,
    wrap_tensor_subclasses,
    wrap_tensor_subclasses_maybe_joint,
)
# 从 _aot_autograd.traced_function_transforms 模块导入跟踪函数转换相关的函数（禁止 F401 警告）
from ._aot_autograd.traced_function_transforms import (
    aot_dispatch_subclass,
    # 创建函数式调用
    create_functional_call,
    # 创建函数化的函数
    create_functionalized_fn,
    # 创建函数化的随机数生成操作包装器
    create_functionalized_rng_ops_wrapper,
    # 创建联合体
    create_joint,
    # 函数输入变异到输出的处理
    fn_input_mutations_to_outputs,
    # 为自动微分准备的函数
    fn_prepped_for_autograd,
# 从特定模块导入多个函数和常量，用于优化静态编译过程
from ._aot_autograd.utils import (
    _get_autocast_states,
    _get_symint_hints,
    call_func_at_runtime_with_args,
    create_tree_flattened_fn,
    KNOWN_TYPES,
    make_boxed_compiler,
    make_boxed_func,
    maybe_to_fresh_input,
    normalize_as_list,
    partial_flatten_asdict,
    root_module_when_exporting_non_strict,
    strict_zip,
)

from .partitioners import default_partition

# 将严格的压缩函数引用赋值给变量 zip
zip = strict_zip

# 全局计数器，每次使用 AOTAutograd 编译图形时递增
# 可用于将运行时错误消息与编译时关联起来（例如，如果在运行时收到指示第 3 个编译图形失败的错误，可以在该图形号处设置断点以进一步调查）
#
# 注意：这与 get_aot_compilation_context 不同，后者跟踪每个编译的基础图形。
# AOT_COUNTER 对应于顶级调用 aot_module/aot_function；每个编译块分配一个计数器（但此块可能涉及编译多个子图；例如，前向/后向传递）
AOT_COUNTER = itertools.count()

# 下面是一个非常长的注释，详细说明 AOT Autograd 处理输入数据变化的边缘情况的逻辑。
# 这些边缘情况是与别名和突变相关的，它们在运行图形时以某种方式显示为副作用。
#
# 参见 `test_aotdispatch.py TestAOTAutograd.test_input_mutation*` 测试用例，了解一些示例函数及其编译后的图形。
# 以下是关于输入数据变异的注释。
#
# 注意 [AOT Autograd：输入数据突变]
#
# 如果我们编译一个突变输入的函数，那么这些输入突变将是用户在运行编译图形后预期看到的真实副作用。
# 但是，我们希望发送给后端的图形必须是完全功能的。
# 我们解决这种差异的方式是，从我们编译的图形中完全删除突变，但我们更新图形以返回（更新后的输入，用户输出）。
# 在编译后的图形执行后的结尾部分，我们将更新后的输入复制回原始输入。
#
# 示例：原始用户代码：
# def f(x):
#     x.mul_(2)
#     out = x.mul(3)
#     return out
#
# AOT Autograd 编译后，我们得到：
# （a）编译的图形
# （b）autograd.Function.forward() 方法，执行编译的图形
# （c）包装函数，调用 autograd.Function.forward() 并执行结尾处理
#
# 下面列出了（a, b, c）的输出。
#
# def compiled_forward_graph(x):
#     x_updated = x.mul(2)
#     out = x_updated.mul(3)
#     return x_updated, out
#
# 这些处理确保了输入数据变异不会影响编译后的图形的功能性。
# # x_updated gets a gradient in the compiled backward
# def compiled_backward_graph(grad_x_updated, grad_out):
#     grad_x = ...
#     return grad_x
#
# def autograd.Function.forward(x):
#     x_updated, out = compiled_forward_graph(x)
#     return x_updated, out
#
# def compiled_wrapper(x):
#     x_updated, out = autograd.Function.apply(x)
#     x.copy_(x_updated)
#     return out
#
# Another important thing to note is that updated inputs (due to data mutations) *do* participate
# in the compiled backward graph! Since the compiled forward graph gets N extra outputs
# (due to updated inputs showing up as graph outputs),
# The compiled backward gets an additional N inputs.
# That way, during the x.copy_(x_updated) bit in the epilogue, gradients will flow from the updated input
# back to the original input.


# Note [AOT Autograd: input metadata mutations]
#
# For the same reason as input mutations, we also don't put input metadata mutations in the graph.
# Instead, we return the updated version of the input (a view), and mutate the input's metadata outside of the graph
#
# Example: original user code:
# def f(x):
#     x.t_()
#     out = x.mul(3)
#     return out
#
# AOT Autograd output (compiled graph, autograd.Function.forward(), wrapper function):
# def compiled_forward_graph(x):
#     x_updated = x.t()
#     out = x_updated.mul(3)
#     return x_updated, out
#
# # x_updated does *not* get a gradient in the compiled backward
# def compiled_backward_graph(grad_out):
#     grad_x = ...
#     return grad_x
#
# def autograd.Function.forward(x):
#     x_updated, out = compiled_forward_graph(x)
#     return x_updated, out
#
# def compiled_wrapper(x):
#     x_updated, out = autograd.Function.apply(x)
#     x.as_strided_(x_updated)
#     return out


# Note [AOT Autograd: outputs aliasing inputs or intermediates!]
#
# AOT Autograd needs special handling for outputs that alias graph inputs or intermediates!
# Why?
# (1) autograd.Function.forward() has a limitation, where views that returned in the forward cannot later be mutated.
# (2) views don't need to be compiled in the graph anyway - it's cheap to generate them outside of the compiled graph,
#     in an epilogue.
# For outputs that alias inputs, we do the following:
# (a) *still* return the aliased output as a graph output
# (b) In the AOT Autograd wrapper/epilogue, we don't return that aliased output. Instead, we use it to regenerate the output.
#
# For outputs that alias *intermediates*, we do the following:
# (a) Return the output in the compiled forward, **and** return it's ._base (a graph intermediates) as an output in the forward
# (b) Use (output, graph_intermediate) to regenerate the alias, and return that to the user (instead of the compiled fw output).
# You might wonder why we return the aliased output directly in the graph (and making the graph compute it),
# only to not return it and instead generate a fresh alias off of the intermediate,
# instead of (say) just storing metadata about the size/stride of the output somewhere to generate the alias. There are two reasons:
# (1) Getting the actual alias tensor allows us to use view-replay to generate the alias, instead of an as_strided() call
# (2) Inductor (and other backends) are free to change the memory format of graph outputs, if it results in better performance.
#     This can result in problems if a user later tries to .view() that output expecting it to have one set of strides,
#     when it has a different set of strides.
#     By including the view op directly in the graph, inductor takes that into account when deciding what memory format
#     the graph intermediate should be.
#
# Another important thing to note is how our traced backward() graph handles aliases.
# (this applies to outputs aliasing inputs, outputs aliasing intermediates,
#  *and* updated inputs returned in the compiled forward due to metadata-only mutations).
# Any outputs that alias (either inputs or intermediates) do NOT participate in the compiled backward graph
# It would be wasteful to include them in the compiled backward(), because we regenerate them eagerly
# at the end of the forward.
#
# Example: original user code:
# def f(x):
#     out1 = x.t()
#     intermediate = x.mul(2)
#     out2 = intermediate.view(-1)
#     return out1, out2
#
# AOT Autograd output (compiled graph, autograd.Function.forward(), wrapper function):
# def compiled_forward_graph(x):
#     out1 = x.t()
#     intermediate = x.mul(2)
#     out2 = intermediate.view(-1)
#     # the compiled graph also returns the intermediate
#     return out1, out2, intermediate
#
# # intermediate gets a gradient in the compiled backward.
# # both output aliases (out1 and out2) do not.
# def compiled_backward_graph(grad_intermediate):
#     grad_x = ...
#     return grad_x
#
# def autograd.Function.forward(x):
#     out1, out2, intermediate = compiled_forward_graph(x)
#     return out1, out2, intermediate
#
# def compiled_wrapper(x):
#     out1, out2, intermediate = autograd.Function.apply(x)
#     # regenerate out1 from the input
#     out1_regenerated = out1._view_func(x)
#     # regenerate out1 from the intermediate
#     out2_regenerated = out2._view_func(intermediate)
#     return out1_regenerated, out2_regenerated


# Note [AOT Autograd: mutations to inputs that alias other inputs]
#
# Another edge case that is (only partially) handled today is when an input is mutated, but itself aliases another input.
# AOT Autograd needs to **ensure** that functionalization knows that the two inputs are aliased to each other.
# That way, when the aliased input is accessed later in the graph, functionalization knows to "update" the alias
# given the mutation that occurred.
#
# This is handled by updating the calling convention: we create a "synthetic base" that becomes a new input
# in the compiled function, and we regenerate the original (aliased) inputs directly off of the base
# inside of the compiled function.
#
# This logic is fully encapsulated in aot_wrapper_synthetic_base()
#
# Example: original user code:
# def f(x, x_view):
#     x.mul_(2)
#     out = x * x_view
#     return out
# f(x, x.view(-1))
#
# AOT Autograd output (compiled graph, autograd.Function.forward(), wrapper function):
# def compiled_forward_graph(base)
#     x = generate_x(base)
#     x_view = generate_x_view(base)
#     x_updated = x.mul(2)
#     x_view_updated = x_updated.view(-1)
#     out = x_updated * x_view_updated
#     return x_updated, out
#
# # The calling convention change from (aliases) -> (base) happens
# # *outside* of the autograd.Function.forward().
# # That means the forward() only has 1 input (base),
# # and the backward() only has 1 output (grad_base)
# def compiled_backward_graph(grad_out):
#     grad_base = ...
#     return grad_base
#
# def autograd.Function.forward(base):
#     x_updated, out = compiled_forward_graph(base)
#     return x_updated, out
#
# # The compiled wrapper is where we create synthetic bases.
# # The info on which inputs are mutated is also tracked *before* synthetic base creation.
# def compiled_wrapper(x, x_view):
#     base = merge_view_inputs(x, x_view)
#     x_updated, out = autograd.Function.apply(base)
#     # x and x_view are aliased in eager mode, so this mutation to x will automatically affect x_view.
#     x.copy_(x_updated)
#     return out


# Note [AOT Autograd: Views to avoid tangents aliasing inputs]
#
# We view every forward output when creating out tangent tensors to handle the problematic
# case in which a subclass does extra aliasing between graph outputs/inputs in a way that
# is not visible above the subclass.
#
# Ordinarily, when constructing the joint function that we want to trace in AOTAutograd,
# we're guaranteed that the tangent tensors that we pass
# into the joint are distinct tensors from the primals. This is because when
# decide which forward outputs to create tangents for, we only create tangents
# for forward outputs that are not aliases of inputs (See Note
# [AOT Autograd: outputs aliasing inputs or intermediates!]).
#
# However, when wrapper tensor subclasses enter the picture, it is possible
# to have an output of the forward that is a subclass that is not an
# input / alias of an input, but one of its inner tensors is an alias!
# NestedTensor is an example: Performing an out-of-place pointwise op on a
# NestedTensor constructs a fresh NestedTensor that holds onto the input's
# offsets tensor directly.
#
# Having tangent tensors that are the same as the (primal) forward inputs,
# can cause problems during tracing as make_fx() will specialize on our
# duplicate inputs: If we passed in the same tensor for primals_1 and
# tangents_1 during tracing, make_fx() will happily sub out all usages of
# tangents_1 with primals_1 in the graph, which is not what we want.
#
# To work around this, we view every forward output when creating out tangent
# tensors so that tangents can never be the same as forward inputs even if
# forward inputs alias forward outputs.

# Note [Side-Effectful Tokens in AOTAutograd]
#
# We allow some side-effectful operators in
# the post-AOTAutograd (functional) graph, such as prints and torchbind operations.
# To ensure compatibility with future graph passes assuming a functional graph,
# "effect tokens" (dummy values like torch.tensor([])) show data dependence between
# these side-effectful operators. An example graph:
#
# def gm(self, token0, reader):
#    token1, frame = with_token(ordered_effect_op, (reader,), token0)
#    frame = frame * 2
#    token2, frame2 = with_token(ordered_effect_op, (reader,), token1)
#    frame2 = frame2 * 2
#    return token2, frame, frame2
#
# Tokens are threaded through operators via `with_effects` and returned updated.
# Input signature: (*tokens, *params_buffers, *user_inputs),
# Output signature: (*tokens, *outputs).
#
# Inductor omits tokens in final code, challenging to alter graph signature post-creation.
# Post-forward graph gen, token removal via pass:
#
# def gm(self, reader):
#    token0 = torch.ops.prims._make_token()
#    token1, frame = with_token(ordered_effect_op, (reader,), token0)
#    frame = frame * 2
#    token2, frame2 = with_token(ordered_effect_op, (reader,), token1)
#    frame2 = frame2 * 2
#    sink_token = torch.ops.prims._sink_tokens([token2])
#    return frame, frame2
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

aot_autograd_decompositions = {}

# Dynamically times the following function's execution
@dynamo_timed
def create_aot_dispatcher_function(
    flat_fn, flat_args: List[Any], aot_config: AOTConfig
) -> Tuple[Callable, ViewAndMutationMeta]:
    """
    Traces forward and backward graphs of `flat_fn` to produce a joint Fx graph
    with Aten ops. See tracing mechanism for capture specifics.
    
    Joint graph passes through `partition_fn` isolating forward and backward
    sections. These are compiled via `fw_compiler` and `bw_compiler`.

    Compiled graphs wrapped in a Callable with meta info returned as Tuple.
    """
    pass  # Placeholder for function implementation
    """
    这段代码用于生成和配置一个自动微分函数的运行环境。

    主要入口点是根据传入的参数 `flat_args` 确定是否采用伪张量模式。
    如果 `aot_config.decompositions` 为空，初始化为一个空字典。

    将自动微分的分解策略 `aot_autograd_decompositions` 和 `aot_config.decompositions` 合并。

    如果配置参数 `config.functionalize_rng_ops` 为真，则更新分解策略，加入随机数函数的分解。

    调用 `detect_fake_mode` 函数检查 `flat_args` 是否已经使用了伪张量模式，如果没有，则根据配置创建 `FakeTensorMode`。

    在确保环境下，配置 Python 调度器模式，具体实现依赖于是否启用了动态形状环境。

    在运行期间，根据需求推迟张量打包/解包钩子的应用。如果有保存的张量钩子激活，则不进行追踪，
    而是让它们在 torch.compile 生成的自定义 autograd.Function 的运行时环境中执行。

    参数说明：
    - `flat_args`: 扁平化的输入参数列表
    - `aot_config`: 预先编译的配置对象，包含自动微分和伪张量的相关配置
    - `aot_autograd_decompositions`: 预定义的自动微分分解策略
    - `rng_decompositions`: 随机数函数的分解策略
    - `FakeTensorMode`: 用于处理伪张量模式的类
    - `ShapeEnv`: 动态形状环境的实例
    """
    
    # 如果没有定义自动微分的分解策略，则初始化为空字典
    if aot_config.decompositions is None:
        aot_config.decompositions = {}

    # 将预定义的自动微分分解策略和现有的分解策略合并
    aot_config.decompositions = {
        **aot_autograd_decompositions,
        **aot_config.decompositions,
    }

    # 如果配置要求功能化随机数操作，则更新分解策略
    if config.functionalize_rng_ops:
        aot_config.decompositions = {
            **rng_decompositions,
            **aot_config.decompositions,
        }

    # 检查输入的参数列表 `flat_args` 是否已经是伪张量模式，如果不是则创建新的伪张量模式实例
    fake_mode = detect_fake_mode(flat_args)
    if fake_mode is None:
        shape_env = ShapeEnv() if aot_config.dynamic_shapes else None
        fake_mode = FakeTensorMode(shape_env=shape_env)
    else:
        shape_env = fake_mode.shape_env

    # 根据是否启用了动态形状环境，配置 Python 调度器模式
    python_dispatcher_mode = (
        enable_python_dispatcher() if shape_env is not None else nullcontext()
    )

    # 延迟执行张量打包/解包钩子直到运行时，根据需要设置多线程状态、保存随机状态、伪张量模式和 Python 调度器模式
    with torch.autograd.set_multithreading_enabled(
        False
    ), preserve_rng_state(), (
        fake_mode
    ), (
        python_dispatcher_mode
    ):
# 当前使用 torch.compile 与张量子类输入一起使用：
# {','.join([str(type(x)) for x in fake_flat_args])}。
# 我们正在尝试编译一个图，其中有两个相互别名的图输出，在子类使用情况下目前不支持此功能。
# 如果遇到此问题，请提交一个 GitHub issue。

if aot_config.is_export:
    # aot_export: 目前禁止输入元数据变异，以保持共享代码路径的简单性。
    # 在图中保留 .resize_() 将需要一些工作。
    # 允许它但保持图的功能性将需要一些调用约定的更改。
    if len([x for x in fw_metadata.input_info if x.mutates_metadata]) != 0:
        raise RuntimeError(
            f"""\
发现一个输入接收了元数据变异，例如调用 `.resize_()` 或 `.transpose_()`。
目前在 aot_export 工作流中禁止此操作。如果您需要此功能，请提交一个 GitHub issue。

fw_metadata={str(fw_metadata)}"""
        )
    
    # 在导出过程中，暂时禁止对需要梯度的输入进行数据变异。
    # 这应该很少见，并且很难正确处理。当我们跟踪反向传播时，
    # 我们当前使用 autograd.grad 而不是 .backward()，这使得很难确保我们在输入受到变异之前就完全运行了 autograd。
    if (
        len(
            [
                x
                for x in fw_metadata.input_info
                if x.requires_grad and x.mutates_data
            ]
        )
        != 0
    ):
        raise RuntimeError(
            f"""\
发现一个需要梯度的图输入，并接收了变异。
目前在 aot_export 工作流中禁止此操作。如果您需要此功能，请提交一个 GitHub issue。

fw_metadata={str(fw_metadata)}"""
        )
    
    # 如果需要子类分发，则抛出运行时错误。
    if req_subclass_dispatch:
        raise RuntimeError(
            """\
aot_export 当前不支持可追踪的张量子类。
如果您需要此功能，请在 <CREATE_ISSUE_LINK> 上评论。"""
        )

    # 需要决定功能化 RNG 的策略：通过全局配置切换似乎不好，
    # 并且打开它将需要非常规范的调用约定更改以适应任何导出运行时。
    if config.functionalize_rng_ops:
        raise RuntimeError(
            """\
功能化的 RNG 目前在 aot_export 工作流中不受支持。请提交一个 GitHub issue。"""
        )
    def aot_function(
        fn: Callable,
        fw_compiler: Callable,
        bw_compiler: Optional[Callable] = None,
        partition_fn: Callable = default_partition,
        decompositions: Optional[Dict] = None,
        num_params_buffers: int = 0,
        keep_inference_input_mutations: bool = False,
        inference_compiler: Optional[Callable] = None,
        *,
        # Whether or not to trace with dynamic shapes
        dynamic=False,
        enable_log=True,
    ) -> Callable:
        """
        Traces the forward and backward graph of :attr:`fn` using torch dispatch
        mechanism, and then compiles the generated forward and backward graphs
        through :attr:`fw_compiler` and :attr:`bw_compiler`.

        :func:`aot_function` traces the forward and backward graph ahead of time,
        and generates a joint forward and backward graph.  :attr:`partition_fn` is
        then used to separate out forward and backward graphs. The partitioner
        function can be used to perform optimizations such as recomputation. One can
        set `decompositions` dictionary to decompose the operators into a sequence
        of core or simpler operators supported by the backend compilers.

        .. warning::
            This API is experimental and likely to change.
        """

        # 内部函数，根据配置选择合适的调度器函数
        def choose_dispatcher(needs_autograd, aot_config):
            """
            Pick a dispatcher based on the config rules.
            """
            if aot_config.is_export:
                # 如果是导出模式，则使用导出调度器函数
                # 导出模式只使用“图形位”，而其他两个调度器包括一些额外的工作来处理运行时的收尾
                return partial(aot_dispatch_export, needs_autograd=needs_autograd)
            elif needs_autograd and not aot_config.pre_dispatch:
                # 如果需要自动求导且不是预调度模式，则使用自动求导调度器函数
                return aot_dispatch_autograd
            else:
                # 否则使用基本调度器函数
                return aot_dispatch_base

        # 根据是否需要自动求导和AOT配置选择合适的编译器函数
        compiler_fn = choose_dispatcher(needs_autograd, aot_config)

        # 编译函数和前向元数据
        compiled_fn, fw_metadata = compiler_fn(
            flat_fn,
            _dup_fake_script_obj(fake_flat_args),
            aot_config,
            fw_metadata=fw_metadata,
        )

        # 返回编译后的函数和前向元数据
        return compiled_fn, fw_metadata
    # 如果未提供反向编译器函数，则使用前向编译器函数作为默认值
    if bw_compiler is None:
        bw_compiler = fw_compiler
    # 如果未提供推断编译器函数，则同样使用前向编译器函数作为默认值
    if inference_compiler is None:
        inference_compiler = fw_compiler
    # 创建一个 AOTConfig 对象，用于配置 Ahead-Of-Time 编译器的参数
    aot_config = AOTConfig(
        fw_compiler=fw_compiler,  # 设置前向编译器函数
        bw_compiler=bw_compiler,  # 设置反向编译器函数
        inference_compiler=inference_compiler,  # 设置推断编译器函数
        partition_fn=partition_fn,  # 设置用于将联合前向和反向图分区的函数
        decompositions=decompositions,  # 定义更大的 Aten 操作分解为更简单或核心 Aten 操作的字典
        num_params_buffers=num_params_buffers,  # 设置参数缓冲区的数量
        aot_id=next(AOT_COUNTER),  # 设置当前 Ahead-Of-Time 编译器实例的唯一标识符
        keep_inference_input_mutations=keep_inference_input_mutations,  # 控制是否保留推断时输入的突变
        dynamic_shapes=dynamic,  # 控制是否启用动态形状支持
        aot_autograd_arg_pos_to_source=None,  # 指定 AOT 编译期间的自动微分参数位置映射
        is_export=False,  # 控制是否导出模型
        no_tangents=False,  # 控制是否禁用切线
        enable_log=enable_log,  # 控制是否启用日志记录
    )
    cached_res = None

    @wraps(fn)
    def returned_function(*args, **kwargs):
        nonlocal cached_res
        # 将输入的张量参数展开成一维数组
        flat_args = pytree.arg_tree_leaves(*args, **kwargs)

        # 编译函数并将其保存在缓存中
        if cached_res is None:
            # 创建展平后的函数和输出规范
            flat_fn, out_spec = create_tree_flattened_fn(fn, args, kwargs)

            # 创建预编译分发函数，并根据给定配置参数进行操作
            compiled_fn, _ = create_aot_dispatcher_function(
                flat_fn,
                flat_args,
                aot_config,
            )
            cached_res = (compiled_fn, out_spec)

        # 从缓存中获取已编译的函数和输出规范
        cached_fn, out_spec = cached_res
        # 使用缓存的函数处理展开后的参数
        out = cached_fn(flat_args)
        # 将处理后的结果展开成原始张量形状
        return out_spec.unflatten(out)

    return returned_function
def aot_module(mod: nn.Module, *args, **kwargs) -> nn.Module:
    """
    Traces the forward and backward graph of :attr:`mod` using torch dispatch
    tracing mechanism. It is wrapper function, that underneath uses
    :func:`aot_function` to perform tracing and compilation.

    :func:`aot_module` lifts the parameters and buffers of ``nn.Module`` as inputs
    to a new callable which is then compiled through :func:`aot_function`.

    .. warning::
        This API is experimental and likely to change.

    Args:
        mod (Callable): A ``nn.Module`` module.
        args : args to be passed to :func:`aot_function`
        kwargs : kwargs to be passed to :func:`aot_function`

    Returns:
        Returns a ``nn.Module`` that retains the eager behavior of the original
        :attr:`mod`, but with forward and backward graph compiled.

    """
    # 检查模块中是否有虚拟参数或缓冲区
    torch._dynamo.utils.assert_no_fake_params_or_buffers(mod)

    def functional_call(named_params, named_buffers, *args, **kwargs):
        # 将命名参数和缓冲区合并成一个字典
        params_and_buffers = {**named_params, **named_buffers}
        # 调用 torch.func.functional_call，传递模块、合并后的参数和缓冲区以及其他参数
        return torch.func.functional_call(mod, params_and_buffers, args, kwargs)

    # 获取模块中的命名参数和缓冲区
    named_params = dict(mod.named_parameters(remove_duplicate=False))
    named_buffers = dict(mod.named_buffers(remove_duplicate=False))
    # 计算参数和缓冲区的总数
    num_params_buffers = len(named_params) + len(named_buffers)
    # 调用 aot_function 进行函数追踪和编译，得到编译后的函数对象
    compiled_f = aot_function(
        functional_call, *args, num_params_buffers=num_params_buffers, **kwargs
    )

    # 定义一个新的 nn.Module 类 AOTModule，用于包装编译后的函数对象
    class AOTModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.orig_module = mod

        def forward(self, *args, **kwargs):
            # 调用编译后的函数对象，传递命名参数、缓冲区以及其他参数
            return compiled_f(
                named_params,
                named_buffers,
                *args,
                **kwargs,
            )

    # 返回 AOTModule 类的实例
    return AOTModule()


def aot_module_simplified(
    mod: nn.Module,
    args,
    fw_compiler: Callable,
    bw_compiler: Optional[Callable] = None,
    partition_fn: Callable = default_partition,
    decompositions: Optional[Dict] = None,
    keep_inference_input_mutations=False,
    inference_compiler: Optional[Callable] = None,
) -> nn.Module:
    """
    This is the simplified or low overhead version of aot_module. For frontends
    like TorchDynamo, the input functions/modules to AOT are static and have
    unpacked inputs/outputs. This gives us an opportunity to remove the
        (1) pytree overhead to parse inputs/outputs,
        (2) AOT Autograd cache,
        (3) Reading of params/buffers in every forward call

    :func:`aot_module_simplified` removes these overheads.
    """
    # 获取模块的参数和缓冲区
    params = {
        **dict(mod.named_parameters(remove_duplicate=False)),
        **dict(mod.named_buffers(remove_duplicate=False)),
    }
    # 将参数和缓冲区展平成列表
    params_flat, params_spec = pytree.tree_flatten(params)
    params_flat = list(params_flat)
    # 计算展平后的参数数量
    params_len = len(params_flat)

    # 如果没有指定反向传播编译器，使用前向传播编译器
    if bw_compiler is None:
        bw_compiler = fw_compiler
    if inference_compiler is None:
        inference_compiler = fw_compiler

如果`inference_compiler`为空，则将其赋值为`fw_compiler`。


    seen_sources = set()

创建一个空的集合`seen_sources`，用于存储已经看到的源对象。


    full_args = []

创建一个空列表`full_args`，用于存储所有的参数。


    # First, the params
    full_args.extend(params_flat)

将`params_flat`中的参数添加到`full_args`列表中，这些参数是作为第一批参数处理的。


    if tracing_context := torch._guards.TracingContext.try_get():
        tracing_context.params_flat = params_flat

尝试获取当前的追踪上下文，如果成功，则将`params_flat`设置为追踪上下文的参数。


    aot_autograd_arg_pos_to_source = None

初始化一个变量`aot_autograd_arg_pos_to_source`，用于存储参数位置到源对象的映射。


    # Then, the params 1:1 mapped sources, if relevant.
    if hasattr(mod, "_param_name_to_source"):
        aot_autograd_arg_pos_to_source = []
        # We now know this came from dynamo, and (1) we care about guards,
        # so setting up aot_autograd_arg_pos_to_source for downstream dedup guards
        # can now be done safely. (2) Dynamo logic protects the 1:1 sizing below.
        for name in params.keys():
            assert name in mod._param_name_to_source, f"{name} not found."
            source = mod._param_name_to_source[name]
            assert source not in seen_sources, source
            seen_sources.add(source)
            aot_autograd_arg_pos_to_source.append(source)

如果模块`mod`具有`_param_name_to_source`属性，说明存在参数名称到源对象的映射关系。这段代码会根据参数的键值对，将每个参数对应的源对象添加到`aot_autograd_arg_pos_to_source`列表中，并确保每个源对象只被添加一次。


    # Next, the input args
    full_args.extend(args)

将输入参数`args`添加到`full_args`列表的末尾。


    if hasattr(mod, "graph"):
        # Non dynamo entrypoints can get to here...
        for node in mod.graph.find_nodes(op="placeholder"):
            if hasattr(node, "_dynamo_source"):
                # ... but not here!
                if aot_autograd_arg_pos_to_source is None:
                    aot_autograd_arg_pos_to_source = []
                source = node._dynamo_source
                assert source not in seen_sources, source
                seen_sources.add(source)
                aot_autograd_arg_pos_to_source.append(source)

如果模块`mod`具有`graph`属性，则遍历图中的所有占位符节点。如果节点具有`_dynamo_source`属性，则将其添加到`aot_autograd_arg_pos_to_source`列表中。确保每个源对象只被添加一次。


    if aot_autograd_arg_pos_to_source is not None:
        assert len(full_args) == len(aot_autograd_arg_pos_to_source)

如果`aot_autograd_arg_pos_to_source`不为空，则断言`full_args`和`aot_autograd_arg_pos_to_source`的长度相等。


    dynamic_shapes = False
    for x in full_args:
        if isinstance(x, FakeTensor):
            dynamic_shapes = x.fake_mode.shape_env is not None
            break

初始化`dynamic_shapes`为`False`，然后检查`full_args`中是否有`FakeTensor`类型的对象，如果有则根据其`shape_env`属性确定`dynamic_shapes`是否为`True`。


    aot_config = AOTConfig(
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        inference_compiler=inference_compiler,
        partition_fn=partition_fn,
        decompositions=decompositions,
        num_params_buffers=params_len,
        aot_id=next(AOT_COUNTER),
        keep_inference_input_mutations=keep_inference_input_mutations,
        dynamic_shapes=dynamic_shapes,
        aot_autograd_arg_pos_to_source=aot_autograd_arg_pos_to_source,
        is_export=False,
        no_tangents=False,
        cache_key=None,
    )

创建一个`AOTConfig`对象`aot_config`，用于配置AOT（Ahead-Of-Time）编译的参数和选项。


    def dispatch_and_compile():
        functional_call = create_functional_call(mod, params_spec, params_len)
        with compiled_autograd.disable():
            compiled_fn, _ = create_aot_dispatcher_function(
                functional_call,
                full_args,
                aot_config,
            )
        return compiled_fn

定义一个内部函数`dispatch_and_compile()`，用于调度和编译功能调用，生成AOT编译函数。


    # Autograd cache stuff
    if config.enable_autograd_cache:
        compiled_fn = AOTAutogradCache.load(dispatch_and_compile, mod, args, aot_config)

如果配置中启用了自动求导缓存，则使用`AOTAutogradCache`加载并缓存`dispatch_and_compile`函数生成的编译函数`compiled_fn`。
    else:
        compiled_fn = dispatch_and_compile()
    # 检查 mod 是否为 torch._dynamo.utils.GmWrapper 类型的实例
    if isinstance(mod, torch._dynamo.utils.GmWrapper):
        # 定义一个名为 boxed_forward 的内部函数，用于处理经过包装的输入参数
        # 这个函数被 flatten_graph_inputs 包装调用，使得输入参数可以在本作用域结束前释放
        def boxed_forward(runtime_args: List[Any]):
            # 将 params_flat 和 runtime_args 合并成一个扁平化的参数列表
            flat_args = []
            flat_args.extend(params_flat)
            flat_args.extend(runtime_args)
            runtime_args.clear()
            # 调用 compiled_fn 处理合并后的参数列表，并返回结果
            return compiled_fn(flat_args)

        # 为了方便使用，将 mod 的 zero_grad、named_parameters、named_buffers 属性赋值给 boxed_forward
        boxed_forward.zero_grad = mod.zero_grad
        boxed_forward.named_parameters = mod.named_parameters
        boxed_forward.named_buffers = mod.named_buffers
        # 返回 boxed_forward 函数作为结果
        return boxed_forward

    # TODO: 这里有一些深层次的问题；compiled_fn 使用了经过包装的调用约定，
    # 但是 aot_module_simplified 在历史上返回了一个未经包装的调用约定的函数。
    # 这个问题应该被修复...
    # 注意：GraphModule/nn.Module 在这里依赖于非经过包装的调用约定
    def forward(*runtime_args: Tuple[Any]):
        # 将 params_flat 和 runtime_args 合并成一个完整的参数列表
        full_args = []
        full_args.extend(params_flat)
        full_args.extend(runtime_args)
        # 调用 compiled_fn 处理合并后的参数列表，并返回结果
        return compiled_fn(full_args)

    # 为了方便使用，将 mod 的 zero_grad、named_parameters、named_buffers 属性赋值给 forward
    forward.zero_grad = mod.zero_grad
    forward.named_parameters = mod.named_parameters
    forward.named_buffers = mod.named_buffers

    # 返回 forward 函数作为结果
    return forward
def aot_export_module(
    mod: nn.Module,
    args,
    *,
    decompositions: Optional[Dict] = None,  # 可选参数，用于指定分解方式的字典
    trace_joint: bool,  # 如果为True，将返回联合前向和反向图，以及有关反向损失和梯度的元数据
    output_loss_index: Optional[int] = None,  # 如果trace_joint为True，指定损失函数在输出中的索引
    pre_dispatch: bool = False,  # 是否预调度，默认为False
    kwargs=None,  # 其他关键字参数
) -> Tuple[torch.fx.GraphModule, GraphSignature]:
    """
    This function takes in a module, and returns:
    (1) an FX graph that can be exported
    (2) some metadata about the graph

    If `trace_joint=True` we will return a joint graph of the forward + backward.

    The traced FX graph will have the following properties compared to the original module:
    (1) Inputs and outputs to the module will be pytree-flattened
    (2) Parameters and buffers on the module will be lifted into graph inputs,
        graph_inputs = (*parameters, *buffers, *user_inputs)
    (3) The graph will be fully functionalized
    (4) Any input mutations will be converted into additional outputs in the graph,
        meaning whoever calls this graph is responsible for applying the mutations
        back to the original inputs.
    (5) If is_joint is provided the graph will return parameter gradients in addition to user outputs.
        The graph output will look like:
        graph_outputs = (*updated_inputs, *user_outputs, *param_gradients)

    There are also several restrictions on what modules can use this API. In particular:
    (1) If trace_joint is specified, we expect the loss function to be **fused**
        into the module forward. One of the outputs to the forward must be a scalar loss,
        which is specified with `output_loss_index`.
        All other outputs to the forward are presumed to not require gradients.
    (2) This API cannot capture optimizers (although in theory we could build an API for this).
    (3) Metadata mutations on params/buffers/inputs are banned.
    (4) Data mutations on anything that requires gradients are banned (parameters)
    (5) If an input is mutated, it is not allowed to alias any other inputs.
    (6) Parameters must not be duplicated.
    """

    if pre_dispatch and trace_joint:
        raise RuntimeError("pre_dispatch is not supported when trace_joint is True.")

    named_parameters = dict(mod.named_parameters(remove_duplicate=False))  # 获取模型中的参数列表，保留重复项
    named_buffers = dict(mod.named_buffers(remove_duplicate=False))  # 获取模型中的缓冲区列表，保留重复项

    params_and_buffers = {
        **dict(named_parameters),  # 将参数转换为字典形式
        **dict(named_buffers),  # 将缓冲区转换为字典形式
    }
    params_and_buffers_flat, params_spec = pytree.tree_flatten(params_and_buffers)  # 对参数和缓冲区进行pytree扁平化处理
    params_and_buffers_flat = tuple(params_and_buffers_flat)  # 转换成元组形式
    params_len = len(params_and_buffers_flat)  # 获取参数和缓冲区的总数

    kwargs = kwargs or {}  # 如果kwargs为None，则初始化为空字典
    # 使用给定参数创建一个功能调用对象，通常是一个函数调用或类方法调用
    functional_call = create_functional_call(
        mod, params_spec, params_len, store_orig_mod=True
    )

    # 初始化一个变量，用于存储后续跟踪的前向输出数量
    num_fw_outs = None

    # 如果需要追踪函数调用过程
    if trace_joint:
        # 定义一个用于追踪的函数，其作用是添加关于反向传播的额外断言：
        # 输出必须包含一个标量损失，我们将根据该损失计算梯度。
        # 我们不会对其他任何内容计算梯度，因此需要使用 detach() 方法来分离输出张量。
        def fn_to_trace(*args):
            nonlocal num_fw_outs  # 使用 nonlocal 关键字声明外部变量 num_fw_outs
            # 调用之前定义的 functional_call 函数，执行功能调用
            out = functional_call(*args)
            # 如果未指定输出损失的索引，则抛出运行时错误
            if output_loss_index is None:
                raise RuntimeError(
                    """\
        # 如果 trace_joint=True，则要求前向传播的输出之一必须是标量损失。
        # 必须指定哪个（索引）输出是损失，使用 output_loss_index。
        )
    if isinstance(out, (torch.Tensor)):
        # 如果输出是单个张量，则转换为元组
        out = (out,)
    if not isinstance(out, (tuple, list)):
        # 如果输出既不是张量也不是张量列表/元组，则引发运行时错误
        raise RuntimeError(
            f"Expected forward output to be either a tensor or a list/tuple of tensors. found {type(out)}"
        )

    for i, o in enumerate(out):
        # 我们只想要针对用户传入的损失创建反向图。
        # 这意味着其他每个输出都不应该需要梯度。
        # 为了不让这成为一个错误（并强制用户将其前向传播的所有其他输出分离），
        # 我们将在这里自动将它们分离。
        if o.requires_grad and i != output_loss_index:
            # 如果找到需要梯度的前向传播输出，但不是标量损失，则引发运行时错误
            raise RuntimeError(
                f"""\
Found an output of the forward that requires gradients, that was not the scalar loss.
We require all outputs to the forward that are not the scalar loss to not require gradient,
because we will only compute a backward graph against the scalar loss.
You can fix this by calling .detach() on each of your forward outputs that is not the loss.
You specified that output index {output_loss_index} is the loss, but we found that
the output at index {i} requires gradients."""
            )
    out_loss = out[output_loss_index]
    num_fw_outs = len(out)
    if not out_loss.requires_grad:
        # 如果标量损失不需要梯度，则引发运行时错误
        raise RuntimeError(
            f"""\
The output at index {output_loss_index} was marked as the loss, but it does not require gradients"""
        )
    if out_loss.numel() != 1:
        # 如果标量损失的大小不为1，则引发运行时错误
        raise RuntimeError(
            f"""\
We require the output marked as the loss (at index {output_loss_index}) to be a scalar, but it has shape {out_loss.shape}"""
        )
    return out

ctx = nullcontext
else:
    # 在 no_grad 下运行，这样我们的跟踪机制只会跟踪推断图。
    # 但是如果 pre_dispatch=True，则我们希望为训练正确跟踪 set_grad_enabled 调用。
    ctx = nullcontext if pre_dispatch else torch.no_grad
    # 要跟踪的函数是 functional_call
    fn_to_trace = functional_call

full_args = []
# 首先是参数
# 注意：参数必须首先出现，Inductor 通过比较 AOTAutograd 内外的参数计数差异来推断“固定”参数，
# 并假定参数前缀是固定参数。
full_args.extend(params_and_buffers_flat)
# 然后是输入参数
full_args.extend(args)
    with ctx():
        # 使用上下文管理器ctx()，执行下面的代码块
        fx_g, metadata, in_spec, out_spec = _aot_export_function(
            fn_to_trace,
            full_args,
            decompositions=decompositions,
            num_params_buffers=params_len,
            no_tangents=True,
            pre_dispatch=pre_dispatch,
            kwargs=kwargs,
        )
    if trace_joint:
        # 如果需要追踪联合图

        def flattened_joint(*args):
            # 定义一个函数flattened_joint，用于处理联合图
            # 这里的思路是AOTAutograd创建的联合图具有一些严格的属性：
            # (1) 它接受两个参数（primals、tangents），并且pytree_flatten它们
            # (2) 它返回一个元组（fw_outs, gradients）
            # 对于希望将联合图分成单独的前向和后向图的人来说，这是一个非常有用的约定。
            # 然而，
            # (1) 对于导出单个联合图的人来说，最好不要在图中有任何pytrees。
            # (2) 在aot_export_module情况下，我们保证前向输出一个损失，因此不需要运行联合图的切线。
            # (3) AOTAutograd为前向中的每个输入创建一个grad_input，
            # 包括那些不需要梯度的张量的None，
            # 我们不希望在导出图中看到这些。
            # 该函数通过移除任何切线输入和从原始FX图中移除pytrees来“修复”上述问题。

            # 创建一个假的切线列表，与前向输出数量和变异输入运行时索引数量相同
            fake_tangents = [
                None
                for _ in range(
                    metadata.num_outputs + metadata.num_mutated_inp_runtime_indices
                )
            ]
            # 调用fx_g函数，传入args和fake_tangents作为参数
            fw_outs, gradients = fx_g(args, fake_tangents)
            # 断言梯度的长度等于args的长度
            assert len(gradients) == len(args)
            # 初始化输出梯度列表
            output_gradients = []
            # 遍历args和对应的梯度
            for i, (a, grad) in enumerate(zip(args, gradients)):
                # 如果a是torch.Tensor并且需要梯度
                if isinstance(a, torch.Tensor) and a.requires_grad:
                    # 断言grad不为None
                    assert (
                        grad is not None
                    ), """\
# 如果 trace_joint 为 True，则使用 nullcontext（一个空的上下文管理器），否则使用 torch.no_grad 上下文管理器
if trace_joint:
    ctx = nullcontext
else:
    # 在 torch.no_grad 下运行，以便我们的追踪机制只追踪推理图
    ctx = torch.no_grad

# 在上下文管理器中执行上述选择的上下文（ctx），以便进行图的导出和追踪
with ctx():
    # 调用 _aot_export_function 函数，获取导出的图（fx_g）、元数据（metadata）、输入规范（in_spec）、输出规范（out_spec）
    fx_g, metadata, in_spec, out_spec = _aot_export_function(
        func,
        args,
        decompositions=decompositions,
    )
    # 解构输入规范，获取输入规范（in_spec）和 _kw_in_spec（关键字输入规范）
    in_spec, _kw_in_spec = in_spec.children_specs
# 到此为止，我们可以直接返回我们追踪的（联合或推理图）。
# 不过首先：进行一系列断言，确保我们的图与原始函数相比不需要任何调用约定更改。
    # 这些限制是在导出过程中的一般限制之外的。

    # 检查输入是否有数据突变
    if (
        len([x for x in metadata.input_info if x.mutates_data or x.mutates_metadata])
        != 0
    ):
        raise RuntimeError(
            f"aot_export_joint_simple does not support input mutations. {str(metadata)}"
        )
    
    # 检查输出是否存在别名
    if (
        len([x for x in metadata.output_info if x.output_type != OutputType.non_alias])
        != 0
    ):
        raise RuntimeError(
            f"aot_export_joint_simple does not support outputs that alias inputs. {str(metadata)}"
        )
    
    # 检查输入是否为叶子节点（非复杂结构）
    if in_spec.is_leaf():
        raise RuntimeError(
            f"aot_export_joint_simple requires inputs to be a single list/tuple. in_spec={str(in_spec)}"
        )
    
    # 检查输入的子节点是否都为叶子节点（非复杂结构）
    if not all(child.is_leaf() for child in in_spec.children_specs):
        raise RuntimeError(
            f"aot_export_joint_simple requires individual inputs not to be pytrees. in_spec={str(in_spec)}"
        )
    
    # 检查输出是否为叶子节点（非复杂结构）
    if out_spec.is_leaf():
        raise RuntimeError(
            f"aot_export_joint_simple requires outputs to be a single list/tuple. out_spec={str(out_spec)}"
        )
    
    # 检查输出的子节点是否都为叶子节点（非复杂结构）
    if not all(child.is_leaf() for child in out_spec.children_specs):
        raise RuntimeError(
            f"aot_export_joint_simple requires individual outputs not to be pytrees. out_spec={str(out_spec)}"
        )
    
    # TODO: 可能需要临时修补 config.functionalize_rng
    # 以防止在导出高阶操作时运行它。

    if config.debug_assert:
        # 对分区后的模块进行前向传播的烟雾测试，检查是否无需调用约定更改即可运行前向传播。
        fw_module, bw_module = aot_config.default_partition(  # noqa: F821
            fx_g, args, num_fwd_outputs=len(fw_metadata.output_infos)  # noqa: F821
        )
        # 尝试使用原始用户输入运行 fw_module
        fake_mode = detect_fake_mode(args)
        if fake_mode is None:
            fake_mode = FakeTensorMode()
        with fake_mode:
            fw_module(*args)
    
    # 返回导出的函数或图形对象 fx_g
    return fx_g
# Private for now because we aren't providing a contract on what to return
# for joint graphs (we could when there's a clearer use case)
# In the future, we may need to add more export API's that provide their own strong guarantees.
# This is meant as a general helper function for handling various export-y use cases.
def _aot_export_function(
    func: Callable,
    args,
    *,
    num_params_buffers: int = 0,
    decompositions: Optional[Dict] = None,
    # If we're exporting a joint graph and we don't want any tangent inputs in the graph
    # (because we are backpropping through a scalar 1 loss),
    # we need to explicitly specify not to include tangents in the graph.
    # It's not enough just to check that our tangent is a scalar, since we also
    # need to know if it is a 1 (no need to make it a graph input), or something else
    # (requiring it to be a graph input).
    # We don't know this info at trace time though, so we need to make it an explicit config.
    no_tangents: bool = False,
    pre_dispatch: bool = False,
    kwargs=None,
) -> Tuple[torch.fx.GraphModule, ViewAndMutationMeta, pytree.TreeSpec, pytree.TreeSpec]:
    kwargs = kwargs or {}

    # Flatten the function and its arguments into a tree structure
    flat_fn, out_spec = create_tree_flattened_fn(func, args, kwargs)
    flat_args, in_spec = pytree.tree_flatten((args, kwargs))

    # Check if any argument is a FakeTensor to determine if dynamic shapes are involved
    dynamic_shapes = False
    for x in flat_args:
        if isinstance(x, FakeTensor):
            dynamic_shapes = x.fake_mode.shape_env is not None
            break

    # Configure the AOT export settings
    aot_config = AOTConfig(
        fw_compiler=None,
        bw_compiler=None,
        inference_compiler=None,
        partition_fn=None,
        decompositions=decompositions,
        num_params_buffers=num_params_buffers,
        aot_id=next(AOT_COUNTER),
        # For now there's no use case involving keeping input mutations in the graph
        # (which we can only do in the inference case anyway).
        # We can add this later if we need to.
        keep_inference_input_mutations=False,
        dynamic_shapes=dynamic_shapes,
        aot_autograd_arg_pos_to_source=None,
        is_export=True,
        no_tangents=no_tangents,
        pre_dispatch=pre_dispatch,
    )

    # Create the AOT dispatcher function and obtain the resulting FX graph and meta information
    fx_g, meta = create_aot_dispatcher_function(
        flat_fn,
        flat_args,
        aot_config,
    )
    # Return the FX graph, meta information, input specification, and output specification
    return fx_g, meta, in_spec, out_spec.spec


@contextmanager
def _detect_attribute_assignment(mod: torch.nn.Module):
    # Do not allow assignment of tensor attributes during export unless
    # the attribute is registered as a buffer.
    # This context manager ensures that attribute assignments are controlled
    # and prevents inadvertent changes to non-buffer attributes.
    yield
    # 定义标准属性集合，这些属性通常由 PyTorch 模块自动生成和使用
    STD_ATTRS = {
        "_backward_hooks",
        "_backward_pre_hooks",
        "_buffers",
        "_forward_hooks",
        "_forward_hooks_always_called",
        "_forward_hooks_with_kwargs",
        "_forward_pre_hooks",
        "_forward_pre_hooks_with_kwargs",
        "_is_full_backward_hook",
        "_load_state_dict_post_hooks",
        "_load_state_dict_pre_hooks",
        "_modules",
        "_non_persistent_buffers_set",
        "_parameters",
        "_state_dict_hooks",
        "_state_dict_pre_hooks",
        "training",
    }

    # 获取模块中所有不属于标准属性集合的属性
    def _get_attributes(mod):
        return {k: v for k, v in mod.__dict__.items() if k not in STD_ATTRS}

    # 在进入函数之前保存属性的状态
    snapshot = pytree.tree_map(lambda x: x, _get_attributes(mod))
    try:
        # 执行 yield 语句，此处可能是为了支持上下文管理器
        yield
    finally:
        # 在退出函数时，比较当前属性状态与保存的快照，找出哪些张量属性被赋值
        assigned_tensor_attributes = []

        # 收集被赋值的张量属性
        def _collect_assigned_tensor_attributes(kp, v, _v):
            if _v is not v:
                attr, *rest = kp
                if isinstance(v, torch.Tensor):
                    assigned_tensor_attributes.append(
                        f"self.{attr.key}{pytree.keystr(rest)}"
                    )
                # TODO(avik): 现在允许对所有其他类型进行赋值。
                # 也许将来我们希望将此限制为原始类型？

        # 遍历当前属性状态和快照，收集被赋值的张量属性
        pytree.tree_map_with_path(
            _collect_assigned_tensor_attributes, snapshot, _get_attributes(mod)
        )
        # 恢复所有属性的状态（包括原始类型的属性）
        mod.__dict__.update(snapshot)

        # 如果有张量属性被赋值，则抛出 ValueError 异常
        if assigned_tensor_attributes:
            if len(assigned_tensor_attributes) > 1:
                noun, verb = "attributes", "were"
            else:
                noun, verb = "attribute", "was"
            raise ValueError(
                f"The tensor {noun} {', '.join(assigned_tensor_attributes)} {verb} assigned during export. "
                "Such attributes must be registered as buffers using the `register_buffer` API "
                "(https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer)."
            )
# 将预编译的函数对象赋给变量 compiled_function
compiled_function = aot_function
# 将预编译的模块对象赋给变量 compiled_module
compiled_module = aot_module
```