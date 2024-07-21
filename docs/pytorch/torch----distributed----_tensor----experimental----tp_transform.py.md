# `.\pytorch\torch\distributed\_tensor\experimental\tp_transform.py`

```py
# 设置 mypy 选项，允许未标记类型的函数定义
mypy: allow-untyped-defs

# 导入必要的模块
import copy  # 导入深拷贝函数
import operator  # 导入运算符模块
from typing import Any, cast, Dict, List, Optional, Sequence, Tuple  # 导入类型提示

import torch  # 导入 PyTorch 库
from torch._subclasses.fake_tensor import FakeTensor  # 导入虚拟张量类
from torch.distributed._tensor import DeviceMesh, distribute_tensor, DTensor  # 导入分布式张量相关模块
from torch.distributed._tensor._op_schema import (
    DTensorSpec,  # 导入分布式张量规范
    OpSchema,  # 导入操作模式模块
    OutputSharding,  # 导入输出分片模块
    OutputSpecType,  # 导入输出规范类型
    PlacementStrategy,  # 导入放置策略
)
from torch.distributed._tensor._redistribute import redistribute_local_tensor  # 导入本地张量重分布函数
from torch.distributed._tensor.placement_types import (
    Placement,  # 导入放置类型
    Replicate,  # 导入复制类型
    Shard,  # 导入分片类型
    TensorMeta,  # 导入张量元信息类型
)
from torch.distributed.tensor.parallel.style import ColwiseParallel, ParallelStyle  # 导入并行风格类
from torch.export import ExportedProgram  # 导入导出的程序类
from torch.export.exported_program import ExportGraphSignature  # 导入导出图签名类
from torch.fx import GraphModule  # 导入图模块类
from torch.fx.experimental.proxy_tensor import make_fx  # 导入创建代理张量函数
from torch.fx.node import Node  # 导入节点类
from torch.fx.passes.infra.pass_base import PassBase, PassResult  # 导入基础通行证类和通行证结果类
from torch.fx.passes.shape_prop import _extract_tensor_metadata  # 导入提取张量元数据函数
from torch.utils import _pytree as pytree  # 导入私有树模块

aten = torch.ops.aten  # 设置 ATen 操作命名空间别名

# 定义函数，将单设备图转换为张量并行图
def tensor_parallel_transformation(
    exported_program: ExportedProgram,  # 导出的程序对象
    rank: int,  # 当前进程的排名
    world_size: int,  # 全局进程数
    device_type: str,  # 设备类型字符串
    parallel_strategies: Dict[str, ParallelStyle],  # 并行策略字典
) -> ExportedProgram:  # 返回类型为导出的程序对象
    """
    The entry point function to perform graph transformations on an exported program
    to transform a single-device graph into a tensor parallel graph.

    .. warning::
        This API is experimental and subject to change.
    """

    gm = exported_program.graph_module  # 获取导出程序对象的图模块
    sig = copy.deepcopy(exported_program.graph_signature)  # 深度复制图签名
    state_dict = copy.copy(exported_program.state_dict)  # 浅复制状态字典

    with gm._set_replace_hook(sig.get_replace_hook()):  # 设置替换钩子
        res = TensorParallelTransformPass(
            rank,
            world_size,
            device_type,
            state_dict,
            exported_program.graph_signature,
            parallel_strategies,
        )(gm)  # 使用自定义的张量并行转换通行证处理图模块
        assert res is not None  # 断言处理结果非空
        gm = res.graph_module  # 更新图模块

    return exported_program._update(gm, sig, state_dict)  # 更新并返回导出的程序对象

# 定义张量并行转换通行证类
class TensorParallelTransformPass(PassBase):
    """
    This pass is responsible for transforming a single-device graph into a tensor parallel
    graph. It will mark the placement strategy of each node in the graph,
    partition the graph into distributed graph, then shard the parameters/buffers accordingly.
    """

    def __init__(
        self,
        rank: int,  # 当前进程的排名
        world_size: int,  # 全局进程数
        device_type: str,  # 设备类型字符串
        state_dict: Dict[str, torch.Tensor],  # 状态字典，键为字符串，值为张量
        graph_signature: ExportGraphSignature,  # 导出图签名对象
        parallel_strategies: Dict[str, ParallelStyle],  # 并行策略字典
    ) -> None:
        super().__init__()  # 调用父类构造函数
        self.rank = rank  # 设置排名属性
        self.mesh = DeviceMesh(device_type, torch.arange(world_size))  # 创建设备网格对象
        self.state_dict: Dict[str, torch.Tensor] = state_dict  # 设置状态字典属性
        self.graph_signature = graph_signature  # 设置图签名属性
        self.parallel_strategies = parallel_strategies  # 设置并行策略属性
    # 定义一个方法 `call`，接受一个 `graph_module` 参数，并返回 `PassResult` 对象
    def call(self, graph_module) -> PassResult:
        # 深拷贝传入的 `graph_module` 对象，以确保不影响原始对象
        gm = copy.deepcopy(graph_module)

        # 根据当前对象的 `state_dict` 的键列表和并行策略生成参数和缓冲区的放置信息
        parameter_placements = _generate_parameter_and_buffer_placements(
            list(self.state_dict.keys()), self.parallel_strategies
        )

        # 标记图结构的分片策略，使用 `graph_signature` 和 `mesh` 对象
        placement_strategies = _mark_sharding(
            gm, self.graph_signature, self.mesh, parameter_placements
        )

        # 对 `gm` 进行分区处理，可能会修改图结构
        _partitioner(gm)

        # 将当前对象的 `state_dict` 按照分片策略进行分片，使用 `graph_signature` 和 `mesh`
        _shard_state_dict(
            self.state_dict, placement_strategies, self.graph_signature, self.mesh
        )

        # 返回处理后的 `gm` 对象和 `True` 表示成功的 `PassResult` 对象
        return PassResult(gm, True)
# 构建参数和缓冲区放置策略，基于线性层的并行风格
def _generate_parameter_and_buffer_placements(
    params_and_buffers: List[str],
    parallel_strategies: Dict[str, ParallelStyle],
) -> Dict[str, Placement]:
    # 初始化空的参数放置字典
    parameter_placements: Dict[str, Placement] = {}
    # 遍历并行策略字典中的每个线性层全限定名和对应的并行风格
    for linear_fqn, parallel_style in parallel_strategies.items():
        # 构造权重和偏置的全限定名
        weight_fqn = f"{linear_fqn}.weight"
        bias_fqn = f"{linear_fqn}.bias"
        # 断言权重全限定名在参数和缓冲区列表中
        assert weight_fqn in params_and_buffers
        # 根据并行风格设置权重的放置策略
        parameter_placements[weight_fqn] = (
            Shard(0) if parallel_style == ColwiseParallel else Shard(1)
        )
        # 如果偏置全限定名也在参数和缓冲区列表中
        if bias_fqn in params_and_buffers:
            # 根据并行风格设置偏置的放置策略
            parameter_placements[bias_fqn] = (
                Shard(0) if parallel_style == ColwiseParallel else Replicate()
            )
    # 返回参数放置字典
    return parameter_placements


# 标记参数和缓冲区占位符节点的放置策略
def _mark_tensor_parallel_shardings(
    gm: GraphModule,
    graph_signature: ExportGraphSignature,
    mesh: DeviceMesh,
    parameter_placements: Dict[str, Placement],
) -> Dict[Node, PlacementStrategy]:
    # 初始化空的放置策略字典
    placement_strategies: Dict[Node, PlacementStrategy] = {}
    # 计算参数和缓冲区的总数
    num_params_and_buffers = len(graph_signature.inputs_to_parameters) + len(
        graph_signature.inputs_to_buffers
    )
    # 初始化占位符索引
    placeholder_idx: int = 0
    # 遍历计算图中的每个节点
    for node in gm.graph.nodes:
        # 如果节点操作为占位符
        if node.op == "placeholder":
            # 如果占位符索引小于参数和缓冲区总数
            if placeholder_idx < num_params_and_buffers:
                # 获取输入节点的全限定名
                fqn: str = _get_input_node_fqn(node.name, graph_signature)
                # 获取输入节点的放置策略，如果没有则使用复制策略
                placement: Placement = (
                    parameter_placements[fqn]
                    if fqn in parameter_placements
                    else Replicate()
                )
                # 创建节点的放置策略并添加到放置策略字典中
                placement_strategies[node] = _create_placement_strategy(
                    node,
                    mesh,
                    placements=(placement,),
                )
                # 增加占位符索引
                placeholder_idx += 1
            else:
                # 对于超出参数和缓冲区总数的占位符节点，使用复制策略
                placement_strategies[node] = _create_placement_strategy(
                    node,
                    mesh,
                    placements=(Replicate(),),
                )
    # 返回节点的放置策略字典
    return placement_strategies


# 返回输入节点的全限定名
def _get_input_node_fqn(input_name: str, graph_signature: ExportGraphSignature) -> str:
    # 如果输入节点在参数输入字典中，则返回对应的全限定名
    if input_name in graph_signature.inputs_to_parameters:
        return graph_signature.inputs_to_parameters[input_name]
    # 如果输入节点在缓冲区输入字典中，则返回对应的全限定名
    elif input_name in graph_signature.inputs_to_buffers:
        return graph_signature.inputs_to_buffers[input_name]
    # 否则抛出数值错误异常
    else:
        raise ValueError(
            f"{input_name} not found in inputs_to_parameters or inputs_to_buffers"
        )


# 标记参数放置的策略
def _mark_sharding(
    gm: GraphModule,
    graph_signature: ExportGraphSignature,
    mesh: DeviceMesh,
    parameter_placements: Dict[str, Placement],
) -> Dict[Node, PlacementStrategy]:
    """
    Mark the placement strategies of the parameter and buffer placeholder nodes.
    """
    # 略
    Mark the sharding strategy for each node in the graph module.
    """
    # 标记图中每个节点的分片策略
    placement_strategies: Dict[
        Node, PlacementStrategy
    ] = _mark_tensor_parallel_shardings(gm, graph_signature, mesh, parameter_placements)

    # 遍历图中的每个节点
    for node in gm.graph.nodes:
        # 处理占位符节点
        if node.op == "placeholder":
            # 如果节点不在分片策略字典中，则创建一个默认的分片策略并添加到字典中
            if node not in placement_strategies:
                placement_strategies[node] = _create_placement_strategy(
                    node, mesh, placements=(Replicate(),)
                )
            # 将节点的 meta 字段中的 sharding 属性设置为对应的分片策略
            node.meta["sharding"] = placement_strategies[node]
        
        # 处理函数调用节点
        elif node.op == "call_function":
            # 如果节点的目标函数是 operator.getitem
            if node.target == operator.getitem:
                # 获取节点的所有输入节点
                input_nodes = node.all_input_nodes
                # 确保该操作只支持一个输入节点
                assert (
                    len(input_nodes) == 1
                ), f"non-compute op only support one input now, found node: {node} with length of inputs: {len(node.args)}"
                # 获取输入节点的分片策略
                arg_strategy = placement_strategies[input_nodes[0]]
                # 根据输入节点的策略创建当前节点的分片策略
                placement_strategies[node] = _create_placement_strategy(
                    node,
                    mesh,
                    placements=arg_strategy.output_spec.placements,
                    input_specs=_get_input_node_specs(node, placement_strategies),
                )
                # 将节点的 meta 字段中的 sharding 属性设置为对应的分片策略
                node.meta["sharding"] = placement_strategies[node]
            
            # 如果节点的操作是其他函数调用
            else:
                # 获取节点的操作模式信息
                op_schema = _get_op_schema(node, placement_strategies)

                # 获取输入和输出的 DTensor 规范
                if (
                    op_schema.op
                    not in DTensor._op_dispatcher.sharding_propagator.op_strategy_funcs
                    and op_schema.op
                    not in DTensor._op_dispatcher.sharding_propagator.op_to_rules
                ):
                    # 如果操作不在默认的分片传播函数中，则标记所有输出为复制
                    output_sharding = _generate_default_output_sharding(
                        node,
                        mesh,
                        op_schema,
                    )
                else:
                    # 否则，使用分片传播器计算输出的分片策略
                    output_sharding = DTensor._op_dispatcher.sharding_propagator.propagate_op_sharding(
                        op_schema,
                    )
                
                # 根据输出的分片策略创建当前节点的分片策略
                placement_strategies[node] = PlacementStrategy(
                    output_specs=_get_output_spec_from_output_sharding(output_sharding),
                    input_specs=output_sharding.redistribute_schema.args_spec
                    if output_sharding.redistribute_schema is not None
                    else _get_input_node_specs(node, placement_strategies),
                )
                # 将节点的 meta 字段中的 sharding 属性设置为对应的分片策略
                node.meta["sharding"] = placement_strategies[node]
        
        # 处理输出节点
        elif node.op == "output":
            # 将输出节点的分片策略设置为 None
            node.meta["sharding"] = None
        
        # 如果节点操作不被支持，则引发运行时错误
        else:
            raise RuntimeError(f"op code {node.op} not supported")
    
    # 返回所有节点的分片策略字典
    return placement_strategies
def _get_output_spec_from_output_sharding(
    output_sharding: OutputSharding,
) -> DTensorSpec:
    """
    Util function to extract output spec from output sharding.
    """
    # 如果输出分片的输出规范是 DTensorSpec 类型，则直接返回
    if isinstance(output_sharding.output_spec, DTensorSpec):
        return output_sharding.output_spec
    else:
        # 对于返回多个输出的操作，这些输出应具有相同的输出规范
        assert isinstance(output_sharding.output_spec, Sequence)
        assert output_sharding.output_spec[0] is not None
        # 将第一个输出的 tensor_meta 设置为 None
        output_sharding.output_spec[0].tensor_meta = None
        return output_sharding.output_spec[0]


def _create_placement_strategy(
    node: Node,
    mesh: DeviceMesh,
    placements: Tuple[Placement, ...],
    input_specs: Optional[Sequence[DTensorSpec]] = None,
) -> PlacementStrategy:
    """
    Util function to construct a placement strategy for a given node.
    """
    # 创建一个放置策略对象，包括输入和输出规范
    placement = PlacementStrategy(
        input_specs=input_specs,
        output_specs=DTensorSpec(
            mesh=mesh,
            placements=placements,
        ),
    )
    # 填充输出规范的张量元数据
    _populate_tensor_meta(node, placement.output_specs)
    return placement


def _populate_tensor_meta(node: Node, output_spec: OutputSpecType) -> None:
    """
    Util function to populate tensor meta of output_spec based on node metadata.
    """
    # 如果节点的值是一个序列，则对每个输出规范进行处理
    if isinstance(node.meta["val"], Sequence):
        assert isinstance(output_spec, Sequence)
        # 逐一为每个规范分配张量元数据
        for spec, fake_tensor in zip(output_spec, node.meta["val"]):
            assert spec is not None
            # 设置规范的张量元数据
            spec.tensor_meta = TensorMeta(
                shape=fake_tensor.shape,
                stride=fake_tensor.stride(),
                dtype=fake_tensor.dtype,
            )
    else:
        # 如果节点的值不是序列，则假定输出规范是 DTensorSpec 类型
        assert isinstance(output_spec, DTensorSpec)
        # 设置输出规范的张量元数据
        output_spec.tensor_meta = TensorMeta(
            shape=node.meta["val"].shape,
            stride=node.meta["val"].stride(),
            dtype=node.meta["val"].dtype,
        )


def _generate_default_output_sharding(
    node: Node,
    mesh: DeviceMesh,
    op_schema: OpSchema,
) -> OutputSharding:
    """
    Util function to create a default output sharding that suggests Replicate placement for both args and outputs.
    """

    def update_arg_spec(arg_spec: DTensorSpec) -> DTensorSpec:
        return DTensorSpec(
            mesh=arg_spec.mesh,
            placements=(Replicate(),),
            tensor_meta=arg_spec.tensor_meta,
        )

    # 更新操作模式的参数规范
    new_op_schema = OpSchema(
        op=op_schema.op,
        args_schema=pytree.tree_map_only(
            DTensorSpec, update_arg_spec, op_schema.args_schema
        ),
        kwargs_schema=op_schema.kwargs_schema,
    )

    def create_output_spec(tensor: FakeTensor) -> DTensorSpec:
        return DTensorSpec(
            mesh=mesh,
            placements=(Replicate(),),
            tensor_meta=TensorMeta(
                shape=tensor.shape,
                stride=tensor.stride(),
                dtype=tensor.dtype,
            ),
        )
    # 返回一个 OutputSharding 对象，用于输出分片
    return OutputSharding(
        # 设置输出规范，通过 pytree.tree_map_only 函数应用 FakeTensor 到 create_output_spec 的结果中
        output_spec = pytree.tree_map_only(
            FakeTensor, create_output_spec, node.meta["val"]
        ),
        # 重新分发的模式和模式的新操作模式
        redistribute_schema = new_op_schema,
        # 设置需要重新分发
        needs_redistribute = True,
    )
# 分区器函数，将单一设备图分区为分布式图
def _partitioner(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    # 遍历图中的每个节点
    for node in gm.graph.nodes:
        # 获取节点的分片信息
        node_sharding = node.meta["sharding"]
        
        # 如果节点操作为 "placeholder"
        if node.op == "placeholder":
            # 获取输出规格
            out_spec = node_sharding.output_spec
            # 对节点值进行分区，更新节点值为本地分区后的值
            local_val = _partition_val(node.meta["val"], out_spec)
            node.meta["val"] = local_val  # 更新节点值
        
        # 如果节点操作为 "call_function"
        elif node.op == "call_function":
            # 获取输出规格
            out_spec = node_sharding.output_spec
            
            # 检查是否存在不对齐的分片，如有则插入重新分片操作
            expected_input_specs = node_sharding.input_specs
            for idx, input_arg in enumerate(node.all_input_nodes):
                input_arg_sharding = input_arg.meta["sharding"]
                input_arg_spec = input_arg_sharding.output_spec
                desired_spec = (
                    out_spec
                    if expected_input_specs is None
                    else expected_input_specs[idx]
                )
                # 如果输入分片规格与期望不符，则插入重新分片操作
                if input_arg_spec != desired_spec:
                    _insert_reshard_gm(
                        gm, node, input_arg, input_arg_spec, desired_spec
                    )
            
            # 将输出值转换为其本地组件
            output_val = node.meta["val"]
            node.meta["val"] = _partition_val(output_val, out_spec)  # 更新输出值
        
        # 如果节点操作为 "output"
        elif node.op == "output":
            for input_arg in node.all_input_nodes:
                # 输出节点的输入参数应为复制类型，否则需要进行重新分发
                input_args_to_check: Sequence[Node] = (
                    input_arg if isinstance(input_arg, Sequence) else [input_arg]
                )
                for arg in input_args_to_check:
                    arg_sharding = arg.meta["sharding"]
                    arg_spec = arg_sharding.output_spec
                    desired_spec = copy.copy(arg_spec)
                    desired_spec.placements = (Replicate(),)
                    # 如果参数规格与期望不符，则插入重新分片操作
                    if arg_spec != desired_spec:
                        _insert_reshard_gm(gm, node, arg, arg_spec, desired_spec)
        
        # 如果节点操作未知，抛出运行时错误
        else:
            raise RuntimeError(f"op code {node} not supported")
    
    # 清理图的元数据
    _clean_up_graph_metadata(gm)
    # 对图进行Lint检查
    gm.graph.lint()
    # 重新编译图
    gm.recompile()
    # 返回分区后的图模块
    return gm


# 辅助函数，将完整张量值转换为其本地组件
def _partition_val(val: Any, spec: DTensorSpec) -> Any:
    """
    util function to convert a full tensor val to its local component
    """
    # 检查值是否为 torch.Tensor 类型
    if isinstance(val, torch.Tensor):
        # 将本地 shard 设置为值
        local_shard = val
        # 如果值的维度为 0，表示已经是标量张量，无需处理，直接返回
        if val.ndim == 0:
            return local_shard

        # 遍历规范中的放置信息
        for idx, placement in enumerate(spec.placements):
            # 如果放置信息是 shard 类型
            if placement.is_shard():
                # 将放置信息转换为 Shard 类型
                placement = cast(Shard, placement)
                # 获取在指定维度上的网格大小
                num_chunks = spec.mesh.size(mesh_dim=idx)
                # 获取当前进程在网格上的坐标
                my_coord = spec.mesh.get_coordinate()
                # 断言当前坐标不为空
                assert my_coord is not None, "current rank not in mesh!"
                # 获取当前坐标在网格维度上的值
                my_coord_on_mesh_dim = my_coord[idx]
                # 将本地 shard 拆分为多个部分，并选择当前进程对应的部分
                local_shard = placement._split_tensor(
                    local_shard, num_chunks, with_padding=False, contiguous=True
                )[0][my_coord_on_mesh_dim]
        # 返回本地 shard
        return local_shard
    # 如果值是列表或元组类型
    elif isinstance(val, (list, tuple)):
        # 递归处理列表或元组中的每个元素
        return val.__class__(_partition_val(v, spec) for v in val)
    # 如果值不是支持的类型
    else:
        # 抛出运行时错误，指示值的类型不受支持
        raise RuntimeError(f"val type {type(val)} not supported")
def _insert_reshard_gm(
    gm: torch.fx.GraphModule,
    node: Node,
    input_arg: Node,
    input_arg_spec: DTensorSpec,
    desired_spec: DTensorSpec,
) -> None:
    """
    Transform the graph for tensor redistribution.
    """
    # 将输入参数的 tensor_meta 属性设置为 input_arg 的元数据中的 tensor_meta
    input_arg_spec.tensor_meta = input_arg.meta["tensor_meta"]
    # 将 desired_spec 的 tensor_meta 属性设置为 input_arg 的元数据中的 tensor_meta
    desired_spec.tensor_meta = input_arg.meta["tensor_meta"]
    # 获取 input_arg 的值数据
    input_arg_tensor = input_arg.meta["val"]

    # 插入 reshard 操作
    def reshard_fn(local_tensor: torch.Tensor) -> torch.Tensor:
        return redistribute_local_tensor(
            local_tensor,
            input_arg_spec,
            desired_spec,
        )

    # 使用 reshard_fn 创建一个新的 GraphModule
    reshard_gm = make_fx(reshard_fn)(input_arg_tensor)
    # 获取 reshard_gm 的所有节点
    reshard_gm_nodes = list(reshard_gm.graph.nodes)
    # 获取 reshard_gm 的第一个节点作为输入节点
    input_node = reshard_gm_nodes[0]
    # 在 gm 的图中 node 节点之前插入新的节点
    with gm.graph.inserting_before(node):
        # 复制 nn_module_stack 的元数据给输出和 all-reduce 节点
        for reshard_node in reshard_gm.graph.nodes:
            if reshard_node.op not in ["placeholder", "output"]:
                # 如果 input_arg 不是占位符，则复制 input_arg 的 nn_module_stack
                # 否则，复制 node 的 nn_module_stack
                reshard_node.meta["nn_module_stack"] = (
                    copy.copy(input_arg.meta["nn_module_stack"])
                    if not input_arg.op == "placeholder"
                    else copy.copy(node.meta["nn_module_stack"])
                )
        # 将 reshard_gm 的图复制到 gm 的图中，将 input_node 映射到 input_arg
        output_node = gm.graph.graph_copy(
            reshard_gm.graph,
            val_map={
                input_node: input_arg,
            },
        )
    # 将 node 的输入参数替换为 output_node
    node.replace_input_with(input_arg, output_node)


def _clean_up_graph_metadata(gm: torch.fx.GraphModule) -> None:
    """
    Clean up the graph by removing sharding and partitioning related metadata
    """
    # 遍历 gm 的所有节点
    for node in gm.graph.nodes:
        # 如果节点的元数据中包含 "sharding"，则删除它
        if "sharding" in node.meta:
            del node.meta["sharding"]
        # 如果节点的元数据中包含 "val" 并且其值是 torch.Tensor 类型
        if "val" in node.meta and isinstance(node.meta["val"], torch.Tensor):
            # 提取 tensor 的元数据并将其保存到 tensor_meta 中
            local_tensor_meta = _extract_tensor_metadata(node.meta["val"])
            node.meta["tensor_meta"] = local_tensor_meta


def _get_input_node_specs(
    node: Node, placement_strategies: Dict[Node, PlacementStrategy]
) -> Tuple[DTensorSpec, ...]:
    """
    Get the input specs of a node.
    """
    # 创建一个空的输入规格列表
    input_specs_list: List[DTensorSpec] = []
    # 遍历节点的所有输入节点
    for input_arg in node.all_input_nodes:
        # 如果输入节点在 placement_strategies 中
        if input_arg in placement_strategies:
            # 获取输入节点的输出规格
            output_spec = placement_strategies[input_arg].output_specs
            assert isinstance(output_spec, DTensorSpec)
            # 将输出规格添加到输入规格列表中
            input_specs_list.append(output_spec)
        else:
            # 如果输入节点不在 placement_strategies 中，则引发 ValueError
            raise ValueError(f"{input_arg} does not have output_spec populated.")
    # 返回输入规格列表的元组形式
    return tuple(input_specs_list)


def _get_op_schema(
    node: Node, placement_strategies: Dict[Node, PlacementStrategy]
) -> OpSchema:
    """
    Util function to construct the operator schema of a node.
    """
    # 使用 pytree.tree_map_only 函数，根据节点的参数构造运算符模式
    args_schema_list = pytree.tree_map_only(
        Node, lambda arg: placement_strategies[arg].output_specs, node.args
    )
    # 创建一个操作模式的模式对象OpSchema，用于表示操作和其参数模式的结构
    op_schema = OpSchema(
        # 将node.target强制转换为torch._ops.OpOverload类型，并作为操作的标识符op
        op=cast(torch._ops.OpOverload, node.target),
        # 将args_schema_list转换为元组，并作为操作的参数模式args_schema
        args_schema=tuple(args_schema_list),
        # 将node.kwargs强制转换为Dict[str, object]类型，并作为操作的关键字参数模式kwargs_schema
        kwargs_schema=cast(Dict[str, object], node.kwargs),
    )
    # 返回构建好的操作模式对象op_schema
    return op_schema
def _shard_state_dict(
    state_dict: Dict[str, torch.Tensor],
    placement_strategies: Dict[Node, PlacementStrategy],
    graph_signature: ExportGraphSignature,
    mesh: DeviceMesh,
) -> None:
    """
    Inplace partition the weights based on the placement strategy
    """
    # 遍历所有节点及其对应的放置策略
    for node, placement_strategy in placement_strategies.items():
        # 如果节点操作不是占位符，跳过
        if node.op != "placeholder":
            continue
        # 根据节点名称查找完全限定名称（fqn）
        if node.name in graph_signature.inputs_to_parameters:
            fqn = graph_signature.inputs_to_parameters[node.name]
        elif node.name in graph_signature.inputs_to_buffers:
            fqn = graph_signature.inputs_to_buffers[node.name]
        else:
            continue
        # 断言完全限定名称在状态字典中存在
        assert fqn in state_dict, f"{fqn} not found in state dict: {state_dict.keys()}"

        # 获取原始参数张量
        original_param = state_dict[fqn]
        # 根据放置策略对参数进行分布式分配
        dtensor_param = distribute_tensor(
            original_param,
            mesh,
            placement_strategy.output_spec.placements,
        )
        # 将分布式张量转换为本地张量
        local_param = dtensor_param.to_local()
        # 更新状态字典中的参数
        state_dict[fqn] = (
            torch.nn.Parameter(local_param)
            if isinstance(original_param, torch.nn.Parameter)
            else local_param
        )
```