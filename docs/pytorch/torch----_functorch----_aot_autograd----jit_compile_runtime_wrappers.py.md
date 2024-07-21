# `.\pytorch\torch\_functorch\_aot_autograd\jit_compile_runtime_wrappers.py`

```py
# mypy: allow-untyped-defs
"""
Functions in this module do most of the "work" of AOTAutograd.
An aot_dispatch_* function:
- Takes in the input flat_fn, flat_args, and some metadata
- Runs a set of pre compile wrappers (e.g. argument deduping)
- Runs the actual compiler
- Wraps the returned callable in a set of post compile wrappers
- Returns the wrapped callable and metadata.
"""

import logging  # 导入日志模块
from contextlib import nullcontext  # 导入用于创建空上下文管理器的模块

from typing import Any, Callable, List, Optional, Sequence, Tuple  # 导入类型提示模块

import torch  # 导入PyTorch模块
import torch.utils.dlpack  # 导入PyTorch的dlpack模块
from torch import Tensor  # 导入PyTorch中的Tensor类型
from torch._dynamo.utils import lazy_format_graph_code  # 导入懒加载格式化图形代码的工具函数
from torch._guards import CompileContext, TracingContext  # 导入编译和跟踪上下文保护模块
from torch._logging import getArtifactLogger, trace_structured  # 导入获取艺术品记录器和结构化跟踪模块
from torch.fx.experimental._backward_state import BackwardState  # 导入后向状态模块
from torch.fx.experimental.proxy_tensor import is_sym_node  # 导入代理张量模块
from torch.fx.experimental.symbolic_shapes import fx_placeholder_vals  # 导入符号形状模块
from .. import config  # 导入配置模块
from .autograd_cache import (  # 导入自动求导缓存相关类和方法
    AOTAutogradCache,
    AOTAutogradCacheEntry,
    CompiledBackward,
    CompiledForward,
)
from .dispatch_and_compile_graph import (  # 导入分发和编译图形相关模块
    aot_dispatch_autograd_graph,
    aot_dispatch_base_graph,
)
from .logging_utils import track_graph_compiling  # 导入跟踪图形编译的工具函数

from .runtime_wrappers import (  # 导入运行时包装器模块
    AOTDedupeWrapper,
    AOTDispatchAutograd,
    AOTDispatchSubclassWrapper,
    AOTSyntheticBaseWrapper,
    AutogradLazyBackwardCompileInfo,
    CompilerWrapper,
    DebugAssertWrapper,
    FakifiedOutWrapper,
    FunctionalizedRngRuntimeWrapper,
    make_runtime_safe,
    post_compile,
    pre_compile,
    RuntimeWrapper,
)
from .schemas import AOTConfig, MutationType, ViewAndMutationMeta  # 导入模式相关类

from .subclass_utils import compute_inner_mutated_inp_indices_from_subclass_meta  # 导入子类工具函数

from .utils import _get_symint_hints, make_boxed_func, strict_zip, unlift_tokens  # 导入工具函数

zip = strict_zip  # 别名定义：zip函数为strict_zip函数

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器
aot_joint_log = getArtifactLogger(__name__, "aot_joint_graph")  # 获取联合图形日志记录器
aot_graphs_log = getArtifactLogger(__name__, "aot_graphs")  # 获取图形日志记录器

aten = torch.ops.aten  # 别名定义：torch的操作aten模块

# Returns a Callable and a ViewAndMutationMeta.
# Currently, only export needs the ViewAndMutationMeta after this function.
DispatchReturn = Tuple[Callable, ViewAndMutationMeta]  # 定义DispatchReturn类型为包含可调用对象和ViewAndMutationMeta的元组


def _create_wrappers_for_dispatch(needs_autograd: bool) -> List[CompilerWrapper]:
    """
    Wrappers that run on every dispatch function
    """
    return [AOTDedupeWrapper(), AOTSyntheticBaseWrapper(trace_joint=needs_autograd)]
    # 返回运行在每个分发函数上的包装器列表，包括去重包装器和合成基础包装器


# Export's dispatching logic is unique in a few ways: it only needs the "graph"
# bits of aot_autograd, and doesn't need to do any specific wrapping.
def aot_dispatch_export(
    flat_fn: Callable,
    flat_args: List[Any],
    aot_config: AOTConfig,
    *,
    fw_metadata: ViewAndMutationMeta,
    needs_autograd: bool,
) -> DispatchReturn:
    wrappers = _create_wrappers_for_dispatch(needs_autograd)
    # 使用_create_wrappers_for_dispatch函数创建包装器列表
    # 调用 pre_compile 函数对模型进行预编译，获取预编译后的函数、参数和框架元数据
    flat_fn, flat_args, fw_metadata = pre_compile(
        wrappers,
        flat_fn,
        flat_args,
        aot_config,
        fw_metadata=fw_metadata,
    )

    # 如果需要自动微分并且没有预调度的 AOT 配置，则调用 aot_dispatch_autograd_graph 函数
    # 生成自动微分图，并获取生成的图对象和框架元数据
    if needs_autograd and not aot_config.pre_dispatch:
        graph, _, _ = aot_dispatch_autograd_graph(
            flat_fn, flat_args, aot_config, fw_metadata=fw_metadata
        )
    else:
        # 否则，调用 aot_dispatch_base_graph 函数生成基础图，并获取生成的图对象和框架元数据
        graph, _, _ = aot_dispatch_base_graph(
            flat_fn, flat_args, aot_config, fw_metadata=fw_metadata
        )

    # 执行 post_compile 函数对编译后的图进行后处理，获取编译后的函数和更新后的框架元数据
    compiled_fn, fw_metadata = post_compile(
        wrappers, graph, aot_config, runtime_metadata=fw_metadata
    )

    # 断言编译后的函数类型为 torch.fx.GraphModule
    assert isinstance(compiled_fn, torch.fx.GraphModule)

    # 返回编译后的函数对象和最终的框架元数据
    return compiled_fn, fw_metadata
def aot_dispatch_base(
    flat_fn,
    flat_args: List[Any],
    aot_config: AOTConfig,
    *,
    fw_metadata: ViewAndMutationMeta,
) -> DispatchReturn:
    """
    Handles functions that don't need autograd. Runs wrappers and compiles with fw_compiler.
    """
    # 创建不需要自动求导的函数包装器
    wrappers = _create_wrappers_for_dispatch(needs_autograd=False)
    # 对函数、参数和元数据进行预编译处理
    flat_fn, flat_args, fw_metadata = pre_compile(
        wrappers, flat_fn, flat_args, aot_config, fw_metadata=fw_metadata
    )

    # 进行基于 AOT 的分发基础图编译
    fw_module, updated_flat_args, maybe_subclass_meta = aot_dispatch_base_graph(  # type: ignore[misc]
        flat_fn, flat_args, aot_config, fw_metadata=fw_metadata
    )

    # 创建 FakifiedOutWrapper 实例，并进行预编译处理
    fakified_out_wrapper = FakifiedOutWrapper()
    (
        fw_module,
        updated_flat_args,
        fw_metadata,
    ) = fakified_out_wrapper.pre_compile(
        fw_module, updated_flat_args, aot_config, fw_metadata=fw_metadata
    )

    # 创建 FunctionalizedRngRuntimeWrapper 实例，并进行预编译处理
    functionalized_rng_wrapper = FunctionalizedRngRuntimeWrapper()
    (
        fw_module,
        updated_flat_args,
        fw_metadata,
    ) = functionalized_rng_wrapper.pre_compile(
        fw_module, updated_flat_args, aot_config, fw_metadata=fw_metadata
    )

    # 检查是否启用了自动混合精度（Automatic Mixed Precision，AMP），设置相应的上下文
    disable_amp = torch._C._is_any_autocast_enabled()
    context = torch._C._DisableAutocast if disable_amp else nullcontext

    # 进入上下文并跟踪图的编译过程，用于推断（inference）
    with context(), track_graph_compiling(aot_config, "inference"):
        # 选择使用推断编译器，如果未指定则使用通用编译器
        compiler = (
            aot_config.inference_compiler
            if aot_config.inference_compiler is not None
            else aot_config.fw_compiler
        )

        # 如果存在追踪上下文，则更新其元数据
        if tracing_context := torch._guards.TracingContext.try_get():
            tracing_context.fw_metadata = (
                fw_metadata
                if maybe_subclass_meta is None
                else maybe_subclass_meta.fw_metadata
            )

        # 在跟踪上下文中报告输出步长
        with TracingContext.report_output_strides() as fwd_output_strides:
            # 使用编译器对模块和更新后的参数进行编译
            compiled_fw = compiler(fw_module, updated_flat_args)

        # 如果需要后续编译处理，则设置前向输出步长
        if fakified_out_wrapper.needs_post_compile:
            fakified_out_wrapper.set_fwd_output_strides(fwd_output_strides)

    # 使运行时安全化
    make_runtime_safe(fw_metadata, maybe_subclass_meta)

    # 如果编译后的函数对象没有 "_boxed_call" 属性，则使用 make_boxed_func 进行包装
    if not hasattr(compiled_fw, "_boxed_call"):
        compiled_fw = make_boxed_func(compiled_fw)

    # 使用 functionalized_rng_wrapper 进行后编译处理
    compiled_fw = functionalized_rng_wrapper.post_compile(
        compiled_fw, aot_config, runtime_metadata=fw_metadata
    )
    # 如果启用了自动微分缓存，并且存在AOT配置的缓存键
    if config.enable_autograd_cache and aot_config.cache_key:
        # 尝试从编译后的前向图中获取缓存键
        if fw_key := getattr(compiled_fw, "_fx_graph_cache_key", None):
            # 创建AOT自动微分缓存条目
            entry = AOTAutogradCacheEntry(
                compiled_fw=CompiledForward(fw_key),
                compiled_bw=None,
                runtime_metadata=fw_metadata,
                dispatch_wrappers=wrappers,
                maybe_subclass_meta=maybe_subclass_meta,
                num_fw_outs_saved_for_bw=None,
                indices_of_inps_to_detach=[],
            )
            # 将条目保存到AOT自动微分缓存中
            AOTAutogradCache.save(aot_config.cache_key, entry)

    # 对编译后的前向函数应用fakified_out_wrapper的后编译处理
    compiled_fw = fakified_out_wrapper.post_compile(
        compiled_fw,
        aot_config,
        runtime_metadata=fw_metadata,
    )

    # 为什么需要传递num_fw_outs_saved_for_bw参数？
    # 参见注释: [子类的分区处理，第二部分]
    # 使用AOTDispatchSubclassWrapper对编译后的前向函数进行后编译处理
    compiled_fw_func = AOTDispatchSubclassWrapper(
        trace_joint=False,
        # TODO: 一旦我们使用pre_compile，这将是该函数顶部的flat_fn
        fw_only=None,
        maybe_subclass_meta=maybe_subclass_meta,
        num_fw_outs_saved_for_bw=None,
    ).post_compile(
        compiled_fw,
        aot_config,  # 未使用
        runtime_metadata=fw_metadata,
    )

    # 如果编译后的前向函数没有"_boxed_call"属性，则创建一个包装函数
    if not hasattr(compiled_fw_func, "_boxed_call"):
        compiled_fw_func = make_boxed_func(compiled_fw_func)

    # 使用RuntimeWrapper对编译后的前向函数进行后编译处理
    compiled_fn = RuntimeWrapper(
        indices_of_inps_to_detach=[],
        trace_joint=False,
        disable_amp=disable_amp,
    ).post_compile(
        compiled_fw_func,
        aot_config,
        runtime_metadata=fw_metadata,
    )

    # 对编译后的函数应用post_compile处理
    compiled_fn = post_compile(
        wrappers, compiled_fn, aot_config, runtime_metadata=fw_metadata
    )
    # 返回编译后的函数
    return compiled_fn
# 定义函数aot_dispatch_autograd，接受多个参数和一个命名参数fw_metadata，并返回DispatchReturn对象
def aot_dispatch_autograd(
    flat_fn,
    flat_args: List[Any],
    aot_config: AOTConfig,
    *,
    fw_metadata: ViewAndMutationMeta,
) -> DispatchReturn:
    """
    Autograd logic. Generates a joint graph, partitions it, manipulates the input with various wrappers,
    and returns a wrapped torch.autograd.Function with a forward and backward.
    """
    # 调用_create_wrappers_for_dispatch函数创建用于自动求导的包装器列表
    wrappers = _create_wrappers_for_dispatch(needs_autograd=True)
    # 调用pre_compile函数对输入进行预处理和编译，更新flat_fn, flat_args, fw_metadata
    flat_fn, flat_args, fw_metadata = pre_compile(
        wrappers,
        flat_fn,
        flat_args,
        aot_config,
        fw_metadata=fw_metadata,
    )

    # 设置fw_metadata的deterministic属性，检查是否启用了确定性算法
    fw_metadata.deterministic = torch.are_deterministic_algorithms_enabled()
    # 调用aot_dispatch_autograd_graph生成联合图，返回fx_g（图）、joint_inputs（输入）、maybe_subclass_meta（子类元数据）
    fx_g, joint_inputs, maybe_subclass_meta = aot_dispatch_autograd_graph(
        flat_fn, flat_args, aot_config, fw_metadata=fw_metadata
    )

    # 从aot_dispatch_autograd_graph函数复制而来，检查是否启用了自动混合精度（AMP）
    disable_amp = torch._C._is_any_autocast_enabled()

    # 如果启用了日志记录，使用lazy_format_graph_code格式化并记录联合图信息
    if aot_config.enable_log:
        aot_joint_log.info(
            "%s",
            lazy_format_graph_code(
                "Joint graph",
                fx_g,
                aot_config.aot_id,
                include_stride=True,
                include_device=True,
                colored=True,
            ),
        )
        # 调用fx_g.print_readable输出可读的联合图信息
        trace_structured(
            "aot_joint_graph",
            payload_fn=lambda: fx_g.print_readable(
                print_output=False, include_stride=True, include_device=True
            ),
        )

    # 尝试获取TracingContext和CompileContext的保存上下文
    saved_context = TracingContext.try_get()
    saved_compile_context = CompileContext.try_get()

    # 找到flat_args中是BackwardState类型的参数的索引列表
    backward_state_indices = [
        idx for idx, x in enumerate(flat_args) if isinstance(x, BackwardState)
    ]
    # 断言backward_state_indices的长度不超过1
    assert len(backward_state_indices) <= 1

    # 创建AutogradLazyBackwardCompileInfo对象，用于惰性反向传播编译信息
    lazy_backward_info = AutogradLazyBackwardCompileInfo(
        bw_module,
        placeholder_list,
        saved_context,
        saved_compile_context,
    )

    # 调用make_runtime_safe函数，确保运行时安全性
    make_runtime_safe(fw_metadata, maybe_subclass_meta)

    # 初始化try_save_cache_entry为None，用于尝试保存缓存条目的回调函数
    try_save_cache_entry: Optional[Callable] = None
    # 如果配置启用自动求导缓存
    if config.enable_autograd_cache:
        
        # 定义保存缓存条目的函数，忽略 F811 错误（函数未使用）
        def try_save_cache_entry(compiled_bw_func):
            # 获取前向函数的图缓存键
            fw_key = getattr(compiled_fw_func, "_fx_graph_cache_key", None)
            # 获取后向函数的图缓存键
            bw_key = getattr(compiled_bw_func, "_fx_graph_cache_key", None)
            # 如果配置了 AOT 缓存键，并且存在前向和后向函数的缓存键
            if aot_config.cache_key and fw_key and bw_key:
                # 创建 AOTAutogradCacheEntry 实例
                entry = AOTAutogradCacheEntry(
                    CompiledForward(fw_key),  # 编译后的前向函数
                    CompiledBackward(  # 编译后的后向函数
                        bw_key, backward_state_indices, num_symints_saved_for_bw
                    ),
                    fw_metadata,  # 前向函数的元数据
                    wrappers,  # 包装器
                    maybe_subclass_meta,  # 可能的子类元信息
                    num_fw_outs_saved_for_bw,  # 保存给后向函数的前向输出数量
                    _indices_of_inps_to_detach,  # 要分离的输入索引
                )
                # 将条目保存到 AOTAutogradCache 中
                AOTAutogradCache.save(aot_config.cache_key, entry)

        # 如果已经编译了后向函数
        if compiled_bw_func is not None:
            # 尝试保存缓存条目
            try_save_cache_entry(compiled_bw_func)
            # 置空尝试保存缓存条目的函数引用

    # 使用 AOTDispatchAutograd 进行后处理编译
    compiled_fn = AOTDispatchAutograd.post_compile(
        compiled_fw_func,  # 编译后的前向函数
        compiled_bw_func,  # 编译后的后向函数
        maybe_subclass_meta,  # 可能的子类元信息
        num_symints_saved_for_bw,  # 保存给后向函数的符号整数数量
        backward_state_indices,  # 后向状态索引
        disable_amp,  # 禁用 AMP
        _indices_of_inps_to_detach,  # 要分离的输入索引
        lazy_backward_info,  # 懒惰后向信息
        aot_config,  # AOT 配置
        fw_metadata=fw_metadata,  # 前向函数的元数据
        try_save_cache_entry=try_save_cache_entry,  # 尝试保存缓存条目的函数引用
    )

    # 如果配置启用调试断言
    if config.debug_assert:
        # 为平坦化参数列表中的每个张量创建需要梯度的列表
        flat_requires_grad: List[Optional[bool]] = [
            a.requires_grad if isinstance(a, Tensor) else None for a in flat_args
        ]
        # 使用 DebugAssertWrapper 进行后处理编译
        compiled_fn = DebugAssertWrapper(
            flat_requires_grad=flat_requires_grad  # 平坦化参数的需要梯度列表
        ).post_compile(compiled_fn, aot_config, runtime_metadata=fw_metadata)

    # 使用 post_compile 进行最终后处理编译
    compiled_fn = post_compile(
        wrappers,  # 包装器列表
        compiled_fn,  # 编译后的函数
        aot_config,  # AOT 配置
        runtime_metadata=fw_metadata,  # 运行时元数据
    )
    # 返回最终编译后的函数
    return compiled_fn
```