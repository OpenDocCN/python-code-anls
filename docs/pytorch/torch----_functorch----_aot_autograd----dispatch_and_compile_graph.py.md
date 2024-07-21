# `.\pytorch\torch\_functorch\_aot_autograd\dispatch_and_compile_graph.py`

```py
"""
This module dispatches the graphs to either the forward-only or joint compilation
pathways, taking into account the AOTConfig and the collected ViewAndMutationMetadata.
"""

import dataclasses
from typing import Any, List, Optional, Tuple

import torch
import torch.utils._pytree as pytree
import torch.utils.dlpack
from torch import Tensor
from torch._dispatch.python import enable_python_dispatcher

from torch._dynamo.utils import lazy_format_graph_code
from torch._logging import getArtifactLogger, trace_structured
from torch._subclasses.functional_tensor import FunctionalTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._python_dispatch import _detect_infra_mode

from .. import config
from .functional_utils import (
    assert_functional_graph,
    propagate_input_mutation_stacktraces,
)
from .schemas import AOTConfig, SubclassMeta, ViewAndMutationMeta
from .traced_function_transforms import (
    aot_dispatch_subclass,
    create_functionalized_fn,
    create_joint,
    fn_input_mutations_to_outputs,
    fn_prepped_for_autograd,
)
from .utils import root_module_when_exporting_non_strict, unlift_tokens

# Initialize logger for AOT graphs
aot_graphs_log = getArtifactLogger(__name__, "aot_graphs")


def _create_graph(f, args, *, aot_config: AOTConfig) -> torch.fx.GraphModule:
    """
    Create a torch.fx.GraphModule from a function `f` and its arguments `args`,
    using the provided AOTConfig.

    Args:
        f: The function to trace.
        args: Arguments to `f`.
        aot_config: Configuration object specifying AOT settings.

    Returns:
        torch.fx.GraphModule: The traced graph module.
    """
    # FunctionalTensorMode must be enabled here.
    # See Note [Accessing .grad_fn on FunctionalTensor]
    with enable_python_dispatcher(), FunctionalTensorMode(
        pre_dispatch=aot_config.pre_dispatch, export=aot_config.is_export
    ):
        # Create the FX graph using make_fx utility function
        fx_g = make_fx(
            f,
            decomposition_table=aot_config.decompositions,
            record_module_stack=True,
            pre_dispatch=aot_config.pre_dispatch,
        )(*args)

    return fx_g


def aot_dispatch_base_graph(
    flat_fn,
    flat_args: List[Tensor],
    aot_config: AOTConfig,
    *,
    fw_metadata: ViewAndMutationMeta,
) -> Tuple[torch.fx.GraphModule, List[Any], Optional[SubclassMeta]]:
    """
    Dispatch the base graph for ahead-of-time (AOT) compilation, handling functionalization
    and input mutations based on provided configuration.

    Args:
        flat_fn: The function to dispatch.
        flat_args: Arguments to `flat_fn`.
        aot_config: Configuration object specifying AOT settings.
        fw_metadata: Metadata related to view and mutation for forward pass.

    Returns:
        Tuple[torch.fx.GraphModule, List[Any], Optional[SubclassMeta]]: A tuple containing:
            - The traced graph module.
            - Updated arguments after functionalization.
            - Optional subclass metadata.
    """
    # aot_dispatch_base requires functionalization, but doesn't need to handle as many cases as the autograd case.
    # The cases that aot_dispatch_base doesn't need to handle include:
    # - outputs that are aliases of graph intermediates
    # - outputs that are aliases of graph inputs
    # While cases that it does need to handle include:
    # - input mutations (including when inputs are aliases of each other)
    # - input metadata mutations
    fn_to_trace = fn_input_mutations_to_outputs(
        flat_fn,
        fw_metadata,
        keep_data_input_mutations=aot_config.keep_inference_input_mutations,
    )

    # Create functionalized function and update flat arguments
    fn_to_trace, updated_flat_args = create_functionalized_fn(
        fn_to_trace,
        flat_args,
        meta=fw_metadata,
        aot_config=aot_config,
        trace_joint=False,
    )

    # TODO: replace with AOTDispatchSubclassWrapper once we refactor
    # fn_input_mutations_to_outputs and create_functionalized_fn
    # into CompilerWrappers.
    (
        fn_to_trace,
        updated_flat_args_subclasses_desugared,
        maybe_subclass_meta,
    ) = aot_dispatch_subclass(
        fn_to_trace,
        updated_flat_args,
        is_joint_structure=False,
        meta=fw_metadata,
        fw_only=flat_fn,
    )
    # 调用 aot_dispatch_subclass 函数，获取返回的三个值，并解构赋值给对应的变量
    aot_graphs_log.debug(
        "aot_config id: %s, fw_metadata=%s,subclass_metadata=%s",
        str(aot_config.aot_id),
        str(fw_metadata),
        str(maybe_subclass_meta),
    )
    # 记录调试信息，包括 aot_config 的 id、fw_metadata 和 maybe_subclass_meta 的值

    # We track buffer assignments when exporting in non-strict mode.
    # (In contrast, strict mode errors on any attribute assignment.)
    # 当以非严格模式导出时，我们跟踪缓冲区的赋值情况。
    mod_when_exporting_non_strict = root_module_when_exporting_non_strict(flat_fn)
    if aot_config.is_export and mod_when_exporting_non_strict is not None:
        # For any buffer that is assigned, we want to associate it to the final proxy node
        # that it is assigned to. This node can then be added as a buffer mutation output.
        assigned_buffers = {}

        def _map_assigned_buffer_to_proxy(_mod, name, buffer):
            # We intercept buffer assignments on the root module through this hook.
            # 我们通过这个钩子拦截根模块上的缓冲区赋值。
            if _mod._buffers is mod_when_exporting_non_strict._buffers:
                # The value assigned to a buffer is a functional tensor, which wraps a fake tensor.
                # 分配给缓冲区的值是一个功能张量，它包装了一个虚拟张量。
                assert isinstance(
                    buffer, torch._subclasses.functional_tensor.FunctionalTensor
                )
                fake = buffer.from_functional()
                # The fake tensor in turn is associated with a proxy node.
                # 虚拟张量又与一个代理节点相关联。
                proxy_mode = _detect_infra_mode(torch._C._TorchDispatchModeKey.PROXY)
                assert proxy_mode is not None
                proxy = torch.fx.experimental.proxy_tensor.get_proxy_slot(
                    fake, proxy_mode.tracer
                ).proxy.node
                # We map the assigned buffer to this proxy node.
                # 将分配的缓冲区映射到这个代理节点。
                assigned_buffers[name] = proxy.name
            return buffer

        # Register module buffer registration hook to track buffer assignments
        # 注册模块缓冲区注册钩子，以跟踪缓冲区赋值
        handle = torch.nn.modules.module.register_module_buffer_registration_hook(
            _map_assigned_buffer_to_proxy
        )

    # Apply tensor detachment for updated_flat_args_subclasses_desugared
    # 对 updated_flat_args_subclasses_desugared 应用张量分离
    saved_updated_flat_args_subclasses_desugared = pytree.tree_map_only(
        torch.Tensor, lambda t: t.detach(), updated_flat_args_subclasses_desugared
    )

    # Create a computation graph based on the traced function and arguments
    # 基于跟踪的函数和参数创建计算图
    fw_module = _create_graph(
        fn_to_trace,
        updated_flat_args_subclasses_desugared,
        aot_config=aot_config,
    )
    # 如果配置要求进行导出并且非严格模式下的模块存在
    if aot_config.is_export and mod_when_exporting_non_strict is not None:
        # 计算非严格模式下模块的命名参数数量
        i = len(dict(mod_when_exporting_non_strict.named_parameters()))
        # 遍历非严格模式下模块的命名缓冲区
        for name, _ in mod_when_exporting_non_strict.named_buffers():
            # 如果缓冲区名在已分配的缓冲区中且不会改变数据
            if name in assigned_buffers and not fw_metadata.input_info[i].mutates_data:  # type: ignore[possibly-undefined]
                # 更新元数据，将输入信息中对应位置标记为改变数据
                fw_metadata.input_info[i] = dataclasses.replace(
                    fw_metadata.input_info[i], mutates_data=True
                )
                # 增加运行时被改变的输入数量计数
                fw_metadata.num_mutated_inp_runtime_indices += 1
            i += 1

        # 将缓冲区分配的节点作为输出节点添加到图中
        add_nodes = []
        output_node = list(fw_module.graph.nodes)[-1]
        # 遍历已分配缓冲区的值
        for name in assigned_buffers.values():  # type: ignore[possibly-undefined]
            # 遍历前向模块的图节点
            for node in fw_module.graph.nodes:
                # 如果节点的名称与已分配缓冲区的名称相同
                if node.name == name:
                    # 将节点添加到需要添加的节点列表中
                    add_nodes.append(node)
                    # 将输出节点添加为节点的用户
                    node.users[output_node] = None
        # 更新输出节点的参数
        output_node.args = ((*add_nodes, *output_node.args[0]),)

        # 移除处理
        handle.remove()  # type: ignore[possibly-undefined]

    # 只要选择移除了输入变异，此时图中不应该有变异操作
    copy_count = assert_functional_graph(fw_module.graph)

    # 消除死代码
    fw_module.graph.eliminate_dead_code()
    # 重新编译前向模块
    fw_module.recompile()

    # 再次确认功能图中的复制计数
    copy_count2 = assert_functional_graph(fw_module.graph)
    # 传播输入变异的堆栈跟踪
    propagate_input_mutation_stacktraces(fw_module.graph)

    # 查看 AOTAutograd 中的副作用令牌
    num_tokens = len(fw_metadata.tokens)
    # 如果存在令牌并且配置允许解除效果令牌
    if num_tokens != 0 and config.unlift_effect_tokens:
        # 解除效果令牌
        unlift_tokens(fw_module, fw_metadata)
        # 更新后的扁平化参数子类化解糖
        saved_updated_flat_args_subclasses_desugared = (
            saved_updated_flat_args_subclasses_desugared[num_tokens:]
        )

    # 断言复制计数与复制计数2相等
    assert copy_count == copy_count2

    # 如果配置要求启用日志记录
    if aot_config.enable_log:
        # 记录 AOTAutograd 中前向图的信息
        aot_graphs_log.info(
            "%s",
            lazy_format_graph_code(
                "Forward graph",
                fw_module,
                aot_config.aot_id,
                include_stride=True,
                include_device=True,
                colored=True,
            ),
        )
        # 跟踪结构化输出前向图
        trace_structured(
            "aot_forward_graph",
            payload_fn=lambda: fw_module.print_readable(
                print_output=False, include_stride=True, include_device=True
            ),
        )

    # 如果配置要求导出
    # TODO: 应将此分解为一个仅返回图的导出函数
    if aot_config.is_export:
        # 断言 maybe_subclass_meta 为 None，因为 aot_export_module 目前不支持张量子类输入
        assert (
            maybe_subclass_meta is None
        ), "aot_export_module does not support tensor subclass inputs for now."
    # 返回前向模块、更新后的扁平化参数子类化解糖、maybe_subclass_meta
    return fw_module, saved_updated_flat_args_subclasses_desugared, maybe_subclass_meta
# 先决条件是 flat_args 中没有重复的参数（例如，相同的 Tensor 对象不会出现两次。
# 但是，两个张量输入可能引用相同的存储，只要它们有单独的 TensorImpls。）
def aot_dispatch_autograd_graph(
    flat_fn,
    flat_args: List[Any],
    aot_config: AOTConfig,
    *,
    fw_metadata: ViewAndMutationMeta,
) -> Tuple[torch.fx.GraphModule, Tuple[List[Any], List[Any]], Optional[SubclassMeta]]:
    # traced_tangents 包含了在追踪的前向传播中应该获得梯度输出的输出集合。
    # 它包括原始前向传播的输出以及由于输入突变而更新的任何输入。
    # 但是，它不包括任何是输入或中间结果的别名，或者任何仅元数据的输入突变。
    joint_inputs = (flat_args, fw_metadata.traced_tangents)

    # 准备用于自动求导的函数 fn_prepared_for_autograd
    fn_prepared_for_autograd = fn_prepped_for_autograd(
        flat_fn,
        fw_metadata,
    )
    
    # 创建联合函数以进行追踪
    joint_fn_to_trace = create_joint(fn_prepared_for_autograd, aot_config=aot_config)

    # 创建功能化的联合函数和更新后的联合输入
    joint_fn_to_trace, updated_joint_inputs = create_functionalized_fn(
        joint_fn_to_trace,
        joint_inputs,
        meta=fw_metadata,
        aot_config=aot_config,
        trace_joint=True,
    )

    # TODO: 替换为 AOTDispatchSubclassWrapper 一旦我们重构 fn_input_mutations_to_outputs 和 create_functionalized_fn
    # 到 CompilerWrappers 中。
    # 使用 AOTDispatchSubclass 执行联合结构的追踪
    subclass_tracing_info = aot_dispatch_subclass(
        joint_fn_to_trace,
        updated_joint_inputs,
        is_joint_structure=True,
        meta=fw_metadata,
        fw_only=flat_fn,
    )

    # 获取普通张量追踪函数和普通张量参数
    joint_fn_to_trace = subclass_tracing_info.plain_tensor_trace_fn
    updated_joint_inputs = subclass_tracing_info.plain_tensor_args

    # 当调用 _create_graph 时，可能会改变联合输入的元数据。
    # 但是调用者期望得到原始的联合输入。因此，我们创建所有输入的别名，确保有一个不被修改的副本。
    #
    # 这会破坏 requires_grad/grad_fn 信息。然而，在 AOTAutograd 下面的后端对此信息不关心，所以这没关系。
    saved_updated_joint_inputs = pytree.tree_map_only(
        torch.Tensor, lambda t: t.detach(), updated_joint_inputs
    )
    
    # 可能的子类元数据
    maybe_subclass_meta = subclass_tracing_info.maybe_subclass_meta
    
    # 记录 AOT 图的日志信息
    aot_graphs_log.info(
        "aot_config id: %s, fw_metadata=%s,subclass_metadata=%s",
        str(aot_config.aot_id),
        str(fw_metadata),
        str(maybe_subclass_meta),
    )

    # 创建图形对象 fx_g
    fx_g = _create_graph(joint_fn_to_trace, updated_joint_inputs, aot_config=aot_config)

    # 在此时图中不应该有任何变异操作。
    assert_functional_graph(fx_g.graph)

    # 与上面的检查重复，但在追踪引入虚假张量的情况下值得保留。
    # 参见注释: [Fake Modules and AOTAutograd]
    # 调用 assert_no_fake_params_or_buffers 函数，确保 fx_g 中没有虚假参数或缓冲区
    torch._dynamo.utils.assert_no_fake_params_or_buffers(fx_g)
    # 对 fx_g 的计算图进行死代码消除优化
    fx_g.graph.eliminate_dead_code()
    # 重新编译 fx_g，可能会修改其计算图
    fx_g.recompile()
    # TODO: 在 AOTAutograd 中，我们创建像 _indices_of_inps_to_detach 这样的元数据，
    # 用于检测前向传播中需要手动 detach() 的输入。
    # 高阶操作可能最终也需要做同样的处理。
    # 如果配置要求导出，则检查是否有子类化的元数据输入，目前不支持在 AOT 导出模块中使用子类化的张量输入。
    if aot_config.is_export:
        assert (
            maybe_subclass_meta is None
        ), "aot_export_module 目前不支持张量子类化输入。"
    # 返回更新后的 fx_g 计算图，保存的更新联合输入，以及可能的子类化元数据
    return fx_g, saved_updated_joint_inputs, maybe_subclass_meta
```