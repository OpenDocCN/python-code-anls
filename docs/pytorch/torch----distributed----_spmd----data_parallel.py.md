# `.\pytorch\torch\distributed\_spmd\data_parallel.py`

```py
# 设置 mypy 来允许未类型化的函数定义
mypy: allow-untyped-defs
# 导入操作符模块，用于运算符相关操作
import operator
# 导入上下文管理器模块，用于管理上下文
from contextlib import contextmanager
# 导入枚举模块，用于定义枚举类型
from enum import Enum
# 导入类型提示模块，用于类型提示
from typing import Any, cast, Dict, List, Optional, Tuple

# 导入 PyTorch 核心库
import torch
# 导入 PyTorch FX 模块
import torch.fx as fx
# 导入 PyTorch 库
import torch.library
# 导入 PyTorch 神经网络模块
import torch.nn as nn
# 导入 PyTorch 的 _pytree 模块
import torch.utils._pytree as pytree
# 导入 PyTorch 分布式相关模块
from torch.distributed._spmd.batch_dim_utils import BatchDimAnalyzer
# 导入 PyTorch 分布式张量相关模块
from torch.distributed._tensor import DeviceMesh, distribute_tensor, Replicate, Shard
# 导入 PyTorch 分布式张量操作模块
from torch.distributed._tensor._op_schema import (
    OpStrategy,
    PlacementStrategy,
    StrategyType,
    TupleStrategy,
)
# 导入 PyTorch 分布式张量重分布模块
from torch.distributed._tensor._redistribute import redistribute_local_tensor
# 导入 PyTorch 分布式张量实用工具模块
from torch.distributed._tensor._utils import compute_local_shape
# 导入 PyTorch 分布式张量放置类型模块
from torch.distributed._tensor.placement_types import _Partial, DTensorSpec, Placement
# 导入 PyTorch FX 图模块
from torch.fx import GraphModule
# 导入 PyTorch FX 实验性代理张量模块
from torch.fx.experimental.proxy_tensor import make_fx
# 导入 PyTorch FX 传递形状属性模块
from torch.fx.passes.shape_prop import _extract_tensor_metadata
# 导入 PyTorch 神经网络实用工具中的成员访问器模块
from torch.nn.utils._named_member_accessor import NamedMemberAccessor

# 使用 torch.ops.aten 来引用 ATen 运算符
aten = torch.ops.aten

# 定义一个虚拟操作，用于数据并行标记梯度
_spmd_lib_def = torch.library.Library("_spmd", "DEF")
_spmd_lib_def.define("tag_grad(Tensor self) -> Tensor")

# 定义一个虚拟实现，用于数据并行标记梯度
_spmd_lib_impl = torch.library.Library("_spmd", "IMPL")
_spmd_lib_impl.impl("tag_grad", lambda x: x, "CompositeExplicitAutograd")

# 定义数据并行操作的风格枚举类型
class DataParallelStyle(Enum):
    """This enum represents the style of the data-parallel operation.

    We have three types of Data Parallel style:
    1. DEFAULT: the default data parallel style, which is to represent a mixed
                replicate and fully shard behavior. For each parameter that is able
                to be sharded evenly, we shard it, otherwise we would replicate the
                parameter. This style avoids potential padding if the parameters
                cannot be sharded evenly, but it would generate a mixed of all_reduce
                and reduce_scatter.
    2. REPLICATE: the data parallel style that replicates all model parameters.
                  This is similar to the behavior of DistributedDataParallel.
    3. FULLY_SHARD: the data parallel style that shards all model parameters. This
                    is similar to the behavior of FullyShardedDataParallel, the
                    difference is that FullyShardedDataParallel (ZERO-3), which
                    shards the model using FlatParameter based sharding,
                    while this style shards each parameter into DTensor.
    """

    DEFAULT = 0
    REPLICATE = 1
    FULLY_SHARD = 2


# 定义节点类型枚举，用于记录图中张量的类型
class NodeType(Enum):
    """NodeType is an enum that records the type of the tensors in the graph.

    This is used to determine the data parallel strategy.
    """

    PARAM = 0        # 模型参数节点
    ACT = 1          # 激活函数节点
    GRAD = 2         # 梯度节点
    STATE = 3        # 状态节点
    NON_TENSOR = 4   # 非张量节点（例如图的输出）

# 定义数据并行策略类，是 OpStrategy 的一种特殊情况，仅记录“数据并行风格”放置策略
class DataParallelStrategy(OpStrategy):
    """DataParallelStrategy is a special case of OpStrategy that only records the "data parallel style" placement
    # 定义一个名为 `DistributedDataParallel` 的类，用于管理在分布式数据并行计算中每个节点的策略。
    
    class DistributedDataParallel:
        """
        定义一个名为 `DistributedDataParallel` 的类，用于管理在分布式数据并行计算中每个节点的策略。
    
        It takes a list of PlacementStrategy, where each PlacementStrategy describes
        one way to distribute the tensor and computation. In the DataParallel case,
        there're two possible ways to distribute the parameters:
            1. replicate the parameter over a set of devices (DDP like behavior)
            2. shard the parameter on its tensor dimension 0 over a set of devices
               (FSDP like behavior).
    
        它接受一个 PlacementStrategy 列表，其中每个 PlacementStrategy 描述了分发张量和计算的一种方式。
        在 DataParallel 案例中，有两种可能的参数分发方式：
            1. 在一组设备上复制参数（类似于 DDP 的行为）
            2. 在张量的维度 0 上对参数进行分片，分布到一组设备上（类似于 FSDP 的行为）。
    
        In addition to the strategy list, we also need to:
        1. `node_type`: record the type of each node in the graph, so that we can
            determine how to propagate in a data parallel fashion.
        2. `reduce_over_batch` is specifically tied to data parallel as the loss
            calculation usually results in scalar tensor where it comes from a
            reduction over the batch dimension. We need to know this information
            so that we could keep the output as sharded.
    
        除了策略列表，我们还需要：
        1. `node_type`: 记录图中每个节点的类型，以便我们可以确定如何以数据并行的方式传播。
        2. `reduce_over_batch`: 特别与数据并行相关，因为损失计算通常会导致标量张量，
            其中来自对批处理维度的缩减。我们需要知道这些信息，以便保持输出为分片的形式。
        """
    
        def __init__(
            self,
            node_type: NodeType,
            strategy_list: List[PlacementStrategy],
            reduction_over_batch: bool = False,
        ):
            """
            构造函数，初始化 DistributedDataParallel 对象。
    
            Args:
            - node_type: 节点类型，用于记录图中每个节点的类型。
            - strategy_list: PlacementStrategy 对象的列表，描述了分布和计算的方式。
            - reduction_over_batch: 是否在批处理维度上进行减少，通常与数据并行相关。
    
            """
            super().__init__(strategy_list)
            self.node_type = node_type
            self.reduction_over_batch = reduction_over_batch
    
        def __str__(self) -> str:
            """
            返回对象的字符串表示形式，包括节点类型和其父类的字符串表示形式。
    
            Returns:
            - str: 对象的字符串表示形式，格式为 "type: {self.node_type}, {super().__str__()}"
            """
            return f"type: {self.node_type}, {super().__str__()}"
@contextmanager
def gradients_tagging(params: Dict[str, torch.Tensor]):
    """Tag the gradient of the parameters with a special tag, so that we can identify them during SPMD expansion.

    It's safe to trace those hooks and we would remove those nodes later.
    """
    tagging_hooks = []
    try:
        # Iterate over each parameter tensor to register hooks for gradient tagging
        for p in params.values():
            # Register a hook that tags gradients using a specific function
            h = p.register_hook(torch.ops._spmd.tag_grad)
            tagging_hooks.append(h)
        # Yield control back to the caller
        yield
    finally:
        # Remove all registered hooks after the context manager exits
        for h in tagging_hooks:
            h.remove()


def _gen_shard_strategy(
    mesh: DeviceMesh, shard_dim: int, input_specs: Optional[List[DTensorSpec]] = None
) -> PlacementStrategy:
    """Util function to generate a shard strategy on shard_dim."""
    return PlacementStrategy(
        # Generate a placement strategy specifying sharding based on shard_dim
        output_specs=DTensorSpec(mesh=mesh, placements=(Shard(shard_dim),)),
        input_specs=input_specs,
    )


def _gen_replicate_strategy(
    mesh: DeviceMesh, input_specs: Optional[List[DTensorSpec]] = None
) -> PlacementStrategy:
    """Util function to generate a replicate strategy."""
    return PlacementStrategy(
        # Generate a placement strategy specifying replication across devices
        output_specs=DTensorSpec(mesh=mesh, placements=(Replicate(),)),
        input_specs=input_specs,
    )


def _gen_partial_strategy(mesh: DeviceMesh) -> PlacementStrategy:
    """Util function to generate a partial strategy."""
    # NOTE: we use AVG by default, avg reduction is needed depending on
    # the loss function, for most loss function it should do
    # gradient averaging. There might be certain cases it should
    # not do gradient averaging (i.e. sum) but it's pretty rare.
    # TODO: Only NCCL supports AVG so using backend like Gloo would
    # crash, we should figure out a way to support avg reduction
    # for non-NCCL backend
    return PlacementStrategy(
        # Generate a placement strategy specifying partial reduction with averaging
        output_specs=DTensorSpec(mesh=mesh, placements=(_Partial("avg"),)),
    )


def build_data_parallel_strategies(
    train_step_graph: GraphModule,
    num_params: int,
    num_states: int,
    mesh: DeviceMesh,
    batch_dim: int = 0,
) -> Dict[fx.Node, StrategyType]:
    """Loop through the train step graph and build the data parallel strategy for each fx Node."""
    activation_idx = num_params + num_states
    non_compute_ops = [
        aten.clone.default,
        aten.detach.default,
        aten.ones_like.default,
        aten.reshape.default,
        aten.t.default,
        aten.view.default,
        torch.ops._spmd.tag_grad.default,
        operator.getitem,
    ]

    tuple_strategy_ops = [aten._fused_adam.default]

    dp_strategy_map: Dict[fx.Node, StrategyType] = {}
    batch_dim_analyzer = BatchDimAnalyzer(batch_dim)
    placeholder_idx = 0
    num_param_grad = 0

    # first we backward propagate to mark the param gradients sharding
    # with tag_grad node helps and then delete the tag_grad nodes
    # 遍历训练步骤图中的节点列表，以逆序方式进行遍历
    for node in reversed(list(train_step_graph.graph.nodes)):
        # 通过标记找到一个 param_grad 节点
        if node.target == torch.ops._spmd.tag_grad.default:
            cur_node = node
            # 当当前节点的目标在非计算操作中时，继续向上遍历直到找到最顶层的节点
            while cur_node.target in non_compute_ops:
                cur_node = cur_node.args[0]
                # 根据当前节点生成部分策略
                partial_strategy = _gen_partial_strategy(mesh)
                # 将当前节点与部分策略映射到数据并行策略对象中
                dp_strategy_map[cur_node] = DataParallelStrategy(
                    NodeType.GRAD, [partial_strategy]
                )
            # 统计 param_grad 节点的数量
            num_param_grad += 1
            # 将 tag_grad 节点替换为其第一个参数节点，从图中移除 tag_grad 节点
            node.replace_all_uses_with(node.args[0])
            train_step_graph.graph.erase_node(node)

            # 如果已处理的 param_grad 数量等于总参数数量，提前结束循环
            if num_param_grad == num_params:
                break

    # 接下来进行前向传播，标记所有的分片操作
    return dp_strategy_map  # type: ignore[return-value]
# 标记数据并行分片
def mark_data_parallel_shardings(
    train_step_graph: GraphModule,
    num_parameters: int,
    num_states: int,
    dp_strategy_map: Dict[fx.Node, StrategyType],
    parallel_mode: DataParallelStyle = DataParallelStyle.FULLY_SHARD,
) -> None:
    """Mark the sharding for the nodes in the train_step_graph."""
    # 计算激活索引位置
    activation_idx = num_parameters + num_states
    # 占位符索引初始为0
    placeholder_idx = 0

# 将值分区为局部组件的实用函数
def _partition_val(val: Any, spec: DTensorSpec) -> Any:
    """Util function to convert a full tensor val to its local component."""
    # 如果值是 torch.Tensor 类型
    if isinstance(val, torch.Tensor):
        local_shard = val
        # 如果是标量张量，则已经是本地的，不需要做任何处理
        if val.ndim == 0:
            return local_shard

        # 遍历规范中的放置策略
        for idx, placement in enumerate(spec.placements):
            if placement.is_shard():
                placement = cast(Shard, placement)
                # 获取网格的尺寸
                num_chunks = spec.mesh.size(mesh_dim=idx)
                # 获取当前坐标
                my_coord = spec.mesh.get_coordinate()
                assert my_coord is not None, "current rank not in mesh!"
                my_coord_on_mesh_dim = my_coord[idx]
                # 分割张量并获取本地分片
                local_shard = placement._split_tensor(
                    local_shard, num_chunks, with_padding=False, contiguous=False
                )[0][my_coord_on_mesh_dim]
        return local_shard
    # 如果值是元组或列表类型
    elif isinstance(val, (tuple, list)):
        # 递归地对每个元素进行分区处理
        return val.__class__(_partition_val(v, spec) for v in val)
    else:
        # 如果值的类型不支持，则抛出运行时错误
        raise RuntimeError(f"val type {type(val)} not supported")

# 对图进行分区，将单设备图分区为分布式图
def partitioner(graph: GraphModule) -> GraphModule:
    """Graph partitioner that partitions the single device graph to distributed graph."""
    # 定义需要调整形状的操作集合
    shape_adjustment_ops = {
        aten._unsafe_view.default: 1,
        aten.expand.default: 1,
        aten.new_zeros.default: 1,
        aten.ones.default: 0,
        aten.reshape.default: 1,
        aten.view.default: 1,
        aten.zeros.default: 0,
    }
    # 对图中的每个节点进行遍历
    for node in graph.graph.nodes:
        # 如果节点的元数据中包含 "sharding"，则删除该信息
        if "sharding" in node.meta:
            del node.meta["sharding"]
        # 如果节点的元数据中包含 "val" 并且其类型为 torch.Tensor
        if "val" in node.meta and isinstance(node.meta["val"], torch.Tensor):
            # 提取局部张量的元数据并添加到 "tensor_meta" 中
            local_tensor_meta = _extract_tensor_metadata(node.meta["val"])
            node.meta["tensor_meta"] = local_tensor_meta

    # 对图进行检查和修正
    graph.graph.lint()
    # 重新编译图
    graph.recompile()
    return graph

# 将图分区为数据并行图
def partition_data_parallel(
    graph: GraphModule,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    params_buffers: Dict[str, torch.Tensor],
    named_states: Dict[str, Any],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    mesh: DeviceMesh,
    parallel_style: DataParallelStyle,
    input_batch_dim: int,
) -> GraphModule:
    """Partition the graph to into a data parallel graph.

    This function also shards/replicates the model parameters and optimizer states to DTensors.
    """
    # 获取 params_buffers 列表的长度，即参数缓冲区的数量
    num_params_buffers = len(params_buffers)
    # 将 named_states 中的所有叶子节点展开成列表
    flattened_states = pytree.tree_leaves(named_states)
    # 统计展开后状态列表的长度
    num_states = len(flattened_states)

    # 通过调用图的方法消除死代码，并检查是否有变化
    changed = graph.graph.eliminate_dead_code()
    # 如果消除了死代码，则重新编译图
    if changed:
        graph.recompile()

    # 1. 首先为整个图构建数据并行策略
    strategy_map = build_data_parallel_strategies(
        graph, num_params_buffers, num_states, mesh=mesh, batch_dim=input_batch_dim
    )

    # 2. 接下来根据并行样式标记每个节点的数据并行策略
    mark_data_parallel_shardings(
        graph,
        num_parameters=num_params_buffers,
        num_states=num_states,
        dp_strategy_map=strategy_map,
        parallel_mode=parallel_style,
    )

    # 3. 将单机图分区为分布式图
    partitioned_graph = partitioner(graph)

    # 为扩展图保留节点类型信息
    for node in partitioned_graph.graph.nodes:
        if node in strategy_map:
            node_strategy = strategy_map[node]
            if isinstance(node_strategy, DataParallelStrategy):
                node.meta["node_type"] = node_strategy.node_type
            elif isinstance(node_strategy, TupleStrategy):
                node.meta["node_type"] = NodeType.NON_TENSOR
            else:
                raise RuntimeError(f"Unknown node strategy {node_strategy}")
        else:
            # 如果节点是扩展节点（集合操作），则标记为与输入节点相同的类型
            input_node = node.all_input_nodes[0]
            node.meta["node_type"] = input_node.meta["node_type"]

    # 4. 最后，根据并行样式将权重和优化状态就地分区为 DTensors
    accessor = NamedMemberAccessor(model)
    # 遍历参数缓冲区中的每个参数和其对应的键
    for param_key, param in params_buffers.items():
        # 设置默认的放置策略为复制
        placement: Placement = Replicate()
        # 如果并行风格是完全分片，则将放置策略设置为分片在第0号处理器上
        if parallel_style == DataParallelStyle.FULLY_SHARD:
            placement = Shard(0)
        # 如果并行风格不是复制，抛出运行时错误，指出不支持的并行风格
        elif parallel_style != DataParallelStyle.REPLICATE:
            raise RuntimeError(f"parallel style {parallel_style} not supported yet")

        # 将参数分发到指定的网格和放置策略上，获取分布式张量
        dtensor_param = distribute_tensor(param, mesh, [placement])
        
        # 更新重新参数化后的模块参数字典和优化器状态字典为分布式张量
        params_buffers[param_key] = dtensor_param.to_local()
        
        # 将模块参数更新为分布式张量
        accessor.set_tensor(param_key, dtensor_param)

        # 如果存在优化器并且参数在优化器状态中
        if optimizer is not None and param in optimizer.state:
            # 获取参数对应的命名状态字典
            param_states = named_states[param_key]
            # 初始化参数的分布式张量状态字典
            param_dtensor_states = {}
            # 遍历参数状态字典中的每个键值对
            for state_key, state_val in param_states.items():
                # 如果状态值是 torch.Tensor 且维度大于0，则进行分片/复制非标量张量
                if isinstance(state_val, torch.Tensor) and state_val.ndim > 0:
                    # 将状态张量分发到指定的网格和放置策略上，获取分布式张量状态
                    dtensor_state = distribute_tensor(state_val, mesh, [placement])
                    param_dtensor_states[state_key] = dtensor_state
                    # 更新参数状态为本地张量
                    param_states[state_key] = dtensor_state.to_local()
                else:
                    # 对于标量张量，保持原状态值不变
                    param_dtensor_states[state_key] = state_val

            # 从优化器状态中移除原始参数状态
            optimizer.state.pop(param)  # type: ignore[call-overload]
            # 将分布式参数和其状态字典添加到优化器状态中
            optimizer.state[dtensor_param] = param_dtensor_states  # type: ignore[index]

    # 返回分区图
    return partitioned_graph
```