# `.\pytorch\torch\_inductor\freezing.py`

```
# mypy: allow-untyped-defs
# Import future annotations to allow forward references in type annotations
from __future__ import annotations

# Import necessary modules
import itertools  # Module for efficient looping
import logging  # Module for logging messages

import weakref  # Support for weak references
from typing import Any, List, Optional, Tuple  # Type hints for variables

import torch  # PyTorch module for tensor computations
import torch.utils._pytree as pytree  # Utility module for PyTorch trees
from torch._dynamo.utils import dynamo_timed, lazy_format_graph_code  # Utilities for dynamo tracing
from torch._functorch.aot_autograd import MutationType  # Type of mutation for autograd
from torch._functorch.compile_utils import fx_graph_cse  # Utility for FX graph common subexpression elimination
from torch._inductor.constant_folding import constant_fold, replace_node_with_constant  # Constants handling utilities

from torch._inductor.fx_passes.freezing_patterns import freezing_passes  # FX passes for freezing patterns
from torch._inductor.fx_passes.post_grad import view_to_reshape  # Post gradient passes

from . import config  # Import local configuration module

aten = torch.ops.aten  # Torch operator namespace
prims = torch.ops.prims  # Torch primitive operator namespace

log = logging.getLogger(__name__)  # Logger object for current module


def replace_params_with_constants(
    gm: torch.fx.GraphModule,
    flat_params: list[Any],
    fw_metadata: torch._functorch.aot_autograd.ViewAndMutationMeta,
) -> List[int]:
    """
    Replaces the parameters of a PyTorch GraphModule with constants wherever possible.
    Returns a list of indices representing the input parameters that were not converted to constants.
    """
    # Find all placeholder nodes representing parameters in the graph
    params = gm.graph.find_nodes(op="placeholder")
    fake_inp_nodes = params[: len(params)]  # Fake input nodes representing parameters
    preserved_arg_indices = []  # List to store indices of preserved arguments

    # Extract indices of aliased input arguments
    aliased_input_args = [
        out_info.base_idx
        for out_info in fw_metadata.output_info
        if out_info.base_idx is not None
    ]

    # Identify indices of mutated inputs
    mutated_inps = [
        i
        for i, m in enumerate(fw_metadata.input_info)
        if m.mutation_type
        in (MutationType.MUTATED_IN_GRAPH, MutationType.MUTATED_OUT_GRAPH)
    ]

    # Replace parameter nodes with constants or preserve them based on mutation status
    for i, (real_input, node) in enumerate(zip(flat_params, fake_inp_nodes)):
        if i in mutated_inps or i in aliased_input_args:
            preserved_arg_indices.append(i)
            continue
        replace_node_with_constant(gm, node, real_input)

    # Extend preserved argument indices with non-parameter inputs
    preserved_arg_indices.extend(range(len(flat_params), len(params)))

    # Recompile the graph module after modifications
    gm.recompile()

    # Return indices of preserved arguments
    return preserved_arg_indices


def freeze(
    dynamo_gm: torch.fx.GraphModule,
    aot_autograd_gm: torch.fx.GraphModule,
    example_inputs: List[torch._subclasses.FakeTensor],
) -> Tuple[torch.fx.GraphModule, List[int]]:
    """
    Inlines parameters that are not mutated into constants and optimizes the graph through constant propagation
    and other techniques. If enabled, the function also discards the original parameters of the module for memory efficiency.

    Assumes that this function is run in dynamo tracing post aot_autograd.

    Args:
        dynamo_gm (torch.fx.GraphModule): The Dynamo constructed GraphModule.
        aot_autograd_gm (torch.fx.GraphModule): The aot_autograd constructed GraphModule to be frozen.
        example_inputs (List[torch.Tensor]): A list of example input tensors to be used in the freezing process.
    """
    # Function for freezing the graph module:
    # Inline non-mutated parameters into constants and optimize through various techniques
    # Discard original parameters if memory efficiency is enabled

    # To be implemented further based on detailed requirements
    pass
    # 返回一个包含冻结的 GraphModule 和被保留输入索引列表的元组。
    """
    # 将 conv 的权重转换为通道最后时可能会遇到 .view 在 fake_tensor_prop 过程中的错误。
    # 因此我们需要先将 view 转换为 reshape。详细信息请参阅 compile_fx.py 中的 fx_codegen_and_compile。

    # 将视图操作 view 转换为 reshape 操作。
    view_to_reshape(aot_autograd_gm)

    # 尝试获取追踪上下文，如果存在，则获取相关元数据和参数。
    if tracing_context := torch._guards.TracingContext.try_get():
        fw_metadata = tracing_context.fw_metadata
        params_flat = tracing_context.params_flat
        assert fw_metadata is not None and params_flat is not None

        # 将参数替换为常量，并返回被保留参数的索引列表。
        preserved_arg_indices = replace_params_with_constants(
            aot_autograd_gm, params_flat, fw_metadata
        )
    else:
        # 如果不存在追踪上下文，则找到图中所有的占位符节点，并返回它们的索引列表。
        inputs = aot_autograd_gm.graph.find_nodes(op="placeholder")
        preserved_arg_indices = list(range(len(inputs)))

    # 对 FX 图进行公共子表达式消除优化。
    cse_graph = fx_graph_cse(aot_autograd_gm.graph)
    aot_autograd_gm.graph = cse_graph
    # 重新编译 GraphModule。
    aot_autograd_gm.recompile()

    # 根据保留的输入索引从示例输入中获取相应的输入。
    aot_example_inputs = [example_inputs[ind] for ind in preserved_arg_indices]

    # 执行冻结 passes，优化模型。
    freezing_passes(aot_autograd_gm, aot_example_inputs)

    # 对 GraphModule 进行常量折叠优化。
    constant_fold(aot_autograd_gm)

    # 如果配置中设置了 freezing_discard_parameters，则失效 nn 模块和丢弃追踪的 GraphModule 参数。
    if config.freezing_discard_parameters:
        invalidate_eager_modules()
        discard_traced_gm_params(dynamo_gm)

    # 记录冻结后的图的调试信息。
    log.debug(
        "%s", lazy_format_graph_code("FROZEN GRAPH", aot_autograd_gm, colored=True)
    )

    # 返回冻结后的 GraphModule 和保留的输入索引列表。
    return aot_autograd_gm, preserved_arg_indices
# 定义了一个继承自 torch.Tensor 的特殊类 ErasedTensor
class ErasedTensor(torch.Tensor):
    
    # 静态方法，用于创建一个新的 ErasedTensor 实例
    @staticmethod
    def __new__(cls, elem, name, owning_mod):
        # 调用父类的构造方法创建新的 Tensor 对象，并将 elem 转移到 meta 设备上
        return super().__new__(cls, elem.to(device="meta"))
    
    # 初始化方法，接收元素 elem、名称 name（可选）和 owning_mod（模块引用）
    def __init__(self, elem, name: Optional[str], mod):
        # 设置擦除后的名称
        self.erased_name = name
        # 使用 weakref 创建 owning_mod 的弱引用
        self.owning_mod_ref = weakref.ref(mod)
    
    # 类方法，用于处理 Torch 分发
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # 从参数 args 和 kwargs 中获取所有 ErasedTensor 实例
        erased_tensors = [
            e
            for e in pytree.arg_tree_leaves(*args, **kwargs)
            if isinstance(e, ErasedTensor)
        ]
        # 断言至少存在一个 ErasedTensor 实例
        assert len(erased_tensors) > 0
        # 获取第一个 ErasedTensor 实例
        e = erased_tensors[0]
        
        # 抛出运行时错误，指示在 Dynamo 冻结后尝试运行 Pytorch Eager 模块
        raise RuntimeError(
            f"Trying to run Pytorch Eager Module after Dynamo Freezing. "
            "The original parameters have been discarded for memory efficiency. "
            f"Found in op {func} for erased parameter {e.erased_name} of {e.owning_mod_ref()}"
        )

# 禁用当前模式下的 Torch Python 分发器
@torch.utils._python_dispatch._disable_current_modes()
def invalidate_eager_modules():
    # 获取当前的 TracingContext，并遍历所有 nn_modules
    for mod in torch._guards.TracingContext.get().module_context.nn_modules.values():
        # 如果 mod 不是 torch.nn.Module 实例，则跳过
        if not isinstance(mod, torch.nn.Module):
            continue
        
        # 遍历模块中的所有命名参数和缓冲区
        for attr_name, tensor in list(
            itertools.chain(
                mod.named_parameters(recurse=False), mod.named_buffers(recurse=False)
            )
        ):
            # 创建一个 ErasedTensor 对象 e_t
            with torch._dispatch.python.no_python_dispatcher():
                e_t = ErasedTensor(tensor, attr_name, mod)
            
            # 如果 tensor 是 torch.nn.Parameter，则设置 e_t 的 requires_grad 属性为 True
            if isinstance(tensor, torch.nn.Parameter):
                e_t.requires_grad_(True)
                e_t._is_param = True  # type: ignore[attr-defined]
            
            # 将 e_t 设置为模块的属性 attr_name
            setattr(mod, attr_name, e_t)

# 禁用当前模式下的 Torch Python 分发器
@torch.utils._python_dispatch._disable_current_modes()
def discard_traced_gm_params(mod: torch.fx.GraphModule):
    # 遍历模块中的所有命名参数和缓冲区
    for attr_name, tensor in list(
        itertools.chain(
            mod.named_parameters(recurse=False), mod.named_buffers(recurse=False)
        )
    ):
        # 创建一个 ErasedTensor 对象 e_t
        with torch._dispatch.python.no_python_dispatcher():
            e_t = ErasedTensor(tensor, attr_name, mod)
        
        # 如果 tensor 是 torch.nn.Parameter，则设置 e_t 的 requires_grad 属性为 True
        if isinstance(tensor, torch.nn.Parameter):
            e_t.requires_grad_(True)
            e_t._is_param = True  # type: ignore[attr-defined]
        
        # 将 e_t 设置为模块的属性 attr_name
        setattr(mod, attr_name, e_t)

# 强制输出布局函数，接收一个 torch.fx.GraphModule 对象 gm
def enforce_output_layout(gm: torch.fx.GraphModule):
    """
    Make sure the output node's layout does not change due to compiler optimizations
    by adding aten.as_strided nodes with the expected strides.
    
    Only used for inference so we can assume all graph outputs are model outputs.
    """
    # 获取图中的所有节点，将最后一个节点视为输出节点
    *_, output_node = gm.graph.nodes
    # 获取输出节点的参数列表
    out_list = output_node.args[0]
    # 在输出节点之前插入操作节点
    with gm.graph.inserting_before(output_node):
        # 遍历输出节点列表
        for n in out_list:
            # 检查节点的值是否为 torch.Tensor 类型，并且值是非重叠且稠密的
            if not isinstance(
                n.meta["val"], torch.Tensor
            ) or not torch._prims_common.is_non_overlapping_and_dense(n.meta["val"]):
                # 如果不满足条件则跳过当前节点
                continue

            # 为了强制使用急切布局，添加一个新节点
            ft = n.meta["val"]
            new_node = gm.graph.call_function(
                prims.inductor_force_stride_order.default, (n, ft.stride())
            )

            # 不能调用 n.replace_all_uses_with(new_node)
            # 因为这会替换 new_node 中对 n 的使用
            # 替换输出节点中的输入节点 n 为新节点 new_node
            output_node.replace_input_with(n, new_node)

    # 对图进行静态分析
    gm.graph.lint()
    # 重新编译计算图
    gm.recompile()
`
# 确保输入数据的布局符合 as_strided 节点的要求，避免编译器优化导致布局改变
def enforce_as_strided_input_layout(gm: torch.fx.GraphModule):
    """
    Make sure the as_strided node's input's layout does not change due to compiler
    optimizations, because the as_strided strides info depends on input tensor stride info.
    """

    # 定义所有可能的 as_strided 操作
    as_strided_ops = [
        torch.ops.aten.as_strided.default,
        torch.ops.aten.as_strided_.default,
        torch.ops.aten.as_strided_scatter.default,
    ]
    # 查找图中所有包含 as_strided 操作的节点
    strided_nodes = [n for n in gm.graph.nodes if n.target in as_strided_ops]
    for n in strided_nodes:
        with gm.graph.inserting_before(n):
            # 添加一个节点来强制使用急切布局
            # 获取节点的第一个参数，并从其元数据中获取值
            ft = n.args[0].meta["val"]
            # 创建一个新节点，使用指定的函数来强制布局
            new_node = gm.graph.call_function(
                prims.inductor_force_stride_order.default, (n.args[0], ft.stride())
            )
            # 替换原始节点的输入参数为新节点
            n.replace_input_with(n.args[0], new_node)

    # 对图进行静态分析检查
    gm.graph.lint()
    # 重新编译图模块
    gm.recompile()


# 装饰器，用于计时函数执行时间
@dynamo_timed
# 将卷积权重张量转换为通道为最后格式
def convert_conv_weights_to_channels_last(gm: torch.fx.GraphModule):
    """
    Convert 4d convolution weight tensor to channels last format.

    This pass is performed before freezing so the added nodes can be constant
    folded by freezing.
    """
    # 查找图中所有包含卷积操作的节点
    convs = [n for n in gm.graph.nodes if n.target == aten.convolution.default]
    for conv in convs:
        weight_node = conv.args[1]
        # 检查权重节点是否为4维张量且不是通道为最后格式
        if len(weight_node.meta["val"].size()) != 4 or weight_node.meta[
            "val"
        ].is_contiguous(memory_format=torch.channels_last):
            # 如果不符合条件，则跳过
            continue

        with gm.graph.inserting_before(conv):
            # 创建一个新节点，使用 clone 函数将权重节点转换为通道为最后格式
            new_node = gm.graph.call_function(
                aten.clone.default,
                (weight_node,),
                {"memory_format": torch.channels_last},
            )
            # 替换原始节点的输入参数为新节点
            conv.replace_input_with(weight_node, new_node)

    # 强制确保输入布局符合 as_strided 节点的要求
    enforce_as_strided_input_layout(gm)
    # 强制确保输出布局符合指定要求
    enforce_output_layout(gm)
```