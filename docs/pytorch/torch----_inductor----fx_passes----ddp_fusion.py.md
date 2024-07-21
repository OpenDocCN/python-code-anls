# `.\pytorch\torch\_inductor\fx_passes\ddp_fusion.py`

```
# 导入所需的库和模块
# Owner(s): ["oncall: distributed"]
import collections
import inspect
import logging
import math
import operator
from dataclasses import dataclass
from functools import partial
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import torch
import torch.fx as fx  # 导入 PyTorch FX 模块
from torch._dynamo.utils import counters
from torch.fx.passes.graph_transform_observer import GraphTransformObserver
from torch.fx.passes.shape_prop import _extract_tensor_metadata, TensorMetadata
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from .. import config  # 导入相对路径下的 config 模块
from ..fx_utils import get_fake_args_kwargs  # 导入相对路径下的 get_fake_args_kwargs 函数
from ..virtualized import V  # 导入相对路径下的 V 对象

aten = torch.ops.aten  # 导入 PyTorch 的 ATen 操作符
logger: logging.Logger = logging.getLogger("comm_fusion")  # 创建名为 "comm_fusion" 的日志记录器


def move_block_after(block: List[fx.Node], target_node: fx.Node) -> None:
    # 将 block 中的每个节点移动到 target_node 之后
    for node in block:
        target_node.append(node)
        target_node = node


def move_block_before(block: List[fx.Node], target_node: fx.Node) -> None:
    # 将 block 中的每个节点移动到 target_node 之前
    for node in block:
        target_node.prepend(node)
        target_node = node


def call_function(
    graph: fx.Graph,
    target: Union[str, Callable[..., Any]],
    args: Optional[Tuple[fx.node.Argument, ...]] = None,
    kwargs: Optional[Dict[str, fx.node.Argument]] = None,
) -> fx.Node:
    # 调用指定的函数，并在图中创建一个新节点来表示此调用
    # 接受 target 参数为 str 类型，以避免由于节点目标的类型是 Union[str, Callable[..., Any]] 而导致的类型错误。
    # 这也允许我们避免在每次调用时进行检查。
    if isinstance(target, str):
        raise RuntimeError(f"Call function should not get a str target {target=}")
    node = graph.call_function(target, args, kwargs)
    _, args, kwargs = get_fake_args_kwargs(node)
    with V.fake_mode:
        # 使用 V.fake_mode 上下文管理器，模拟调用函数并将结果存储在节点的 meta 字典中
        node.meta["val"] = target(*args, **kwargs)
        # node.meta["val"] 可能是一个容器，因此我们在这里使用 tree_map 递归提取张量的元数据。
        node.meta["tensor_meta"] = tree_map(
            _extract_tensor_metadata, (node.meta["val"],)
        )[0]
    return node


@dataclass(unsafe_hash=True)
class CommBlock:
    # 通信块的数据结构，用于存储通信相关的节点信息
    shape: Union[torch.Size, List[torch.Size]]
    node_list: List[fx.Node]
    inputs: List[fx.Node]
    wait_nodes: List[fx.Node]
    comm_node: fx.Node
    outputs: Set[fx.Node]


def get_comm_block(comm_node: fx.Node) -> Optional[CommBlock]:
    """
    给定一个集合节点（例如 allreduce），找出所有属于此通信的节点。

    Args:
        comm_node(fx.Node): 目标通信/集合节点。
    Returns:
        返回包含给定 comm_node 相关节点（例如 wait_node）的 CommBlock。
    """
    node_list = []  # 存储相关节点的列表
    wait_nodes = []  # 存储等待节点的列表
    inputs, _ = tree_flatten((comm_node.args, comm_node.kwargs))  # 将输入的参数展开成列表
    input_nodes = [inp for inp in inputs if isinstance(inp, fx.Node)]  # 从输入中筛选出节点类型的参数
    wait_prefixes = "wait_tensor"  # 等待节点的前缀
    # 如果等待节点的使用者符合以下条件，我们认为它们是输出的一部分。
    # 此处待补充，需要根据具体的代码逻辑进一步解释。
    # 定义中间输出的名称，用于标识此部分中的节点
    intermediate_outputs = ("split", "reshape", "getitem", "detach", "alias")

    # 获取与通信节点相关联的第一个用户节点
    first_user = next(iter(comm_node.users))
    # 检查是否只有一个用户并且该用户的目标是等待张量默认函数
    if (
        len(comm_node.users) == 1
        and first_user.target == torch.ops._c10d_functional.wait_tensor.default
    ):
        # 如果是一个输出的集合操作
        node_list = [comm_node, first_user]
        wait_nodes.append(first_user)
    # 检查是否有多个用户并且第一个用户的目标是获取操作符
    elif len(comm_node.users) > 1 and first_user.target == operator.getitem:
        # 如果是多个输出的集合操作
        node_list.append(comm_node)
        for user in comm_node.users:
            if user.target != operator.getitem:
                return None
            if len(user.users) != 1:
                return None
            wait_node = next(iter(user.users))
            if wait_node.target != torch.ops._c10d_functional.wait_tensor.default:
                return None
            wait_nodes.append(wait_node)
            node_list.append(user)
        node_list.extend(wait_nodes)
    else:
        # 如果不符合任何集合操作的条件则返回空
        return None

    # 标识此集合块中所有的输出节点
    outputs: Set[fx.Node] = set()
    nodes = collections.deque(wait_nodes)
    while nodes:
        node = nodes.popleft()
        for user in node.users:
            # 如果用户节点是一个有效的FX节点并且其名称以中间输出名称开头
            if isinstance(user, fx.Node) and user.name.startswith(intermediate_outputs):
                nodes.append(user)
                node_list.append(user)
            else:
                outputs.add(node)
                break

    # 获取输入节点的张量元数据
    tensor_meta = input_nodes[0].meta["tensor_meta"]
    shape: Union[torch.Size, List[torch.Size]]
    # 根据张量元数据的类型确定其形状
    if isinstance(tensor_meta, TensorMetadata):
        shape = tensor_meta.shape
    elif isinstance(tensor_meta, (list, tuple)):
        shape = [tm.shape for tm in tensor_meta]
    else:
        # 记录警告，张量元数据的类型不符合预期
        logger.warning("Unexpected type of tensor_meta %s", type(tensor_meta))
        return None

    # 返回通信块对象，包括形状、节点列表、等待节点、通信节点、输入节点和输出节点
    return CommBlock(
        shape=shape,
        node_list=node_list,
        wait_nodes=wait_nodes,
        comm_node=comm_node,
        inputs=input_nodes,
        outputs=outputs,
    )
# 定义一个函数，用于获取所有通信块（CommBlock）的列表
def get_all_comm_blocks(
    graph: fx.Graph,  # 图形对象，表示计算图
    comm_ops: Tuple[torch._ops.OpOverload, ...],  # 通信操作的元组
    comm_filter: Optional[Callable[..., bool]] = None,  # 可选的过滤器函数，默认为 None
) -> List[CommBlock]:  # 返回一个 CommBlock 对象的列表

    # 如果未提供通信过滤器函数，则定义一个总是返回 True 的函数
    if comm_filter is None:
        def always_true(comm_block: CommBlock) -> bool:
            return True
        comm_filter = always_true  # 将默认过滤器函数设置为 always_true 函数

    blocks = []  # 初始化一个空列表用于存储符合条件的 CommBlock 对象
    for node in graph.nodes:  # 遍历图中的每个节点
        if node.target not in comm_ops:  # 如果节点的目标不在通信操作列表中，则跳过
            continue
        comm_block = get_comm_block(node)  # 获取当前节点对应的通信块
        if comm_block is not None and comm_filter(comm_block):  # 如果通信块不为空且通过过滤器函数
            blocks.append(comm_block)  # 将通信块添加到列表中
    return blocks  # 返回符合条件的通信块列表


def _fuse_allreduce_by_concat(
    graph: fx.Graph,  # 图形对象，表示计算图
    last_input_node: fx.Node,  # 最后一个输入节点
    all_input_nodes: List[fx.Node],  # 所有输入节点的列表
    last_comm_block: CommBlock,  # 最后一个通信块对象
) -> CommBlock:  # 返回一个被融合的通信块对象

    """给定按顺序的输入列表，使用 concat 创建一个融合的 allreduce。"""

    # 将所有输入节点展平到 all_reduce 节点。
    with graph.inserting_after(last_input_node):  # 在最后一个输入节点之后插入新节点
        cat_inputs = []
        for input_node in all_input_nodes:
            assert isinstance(input_node.args[0], fx.Node)
            input_node = input_node.args[0]
            cat_inputs.append(
                call_function(graph, aten.flatten.using_ints, (input_node,))
            )

    # 将所有展平的节点进行 concat 操作。
    with graph.inserting_after(cat_inputs[0]):  # 在第一个展平节点之后插入新节点
        cat_node = call_function(graph, aten.cat, (cat_inputs,))

    # 插入融合的除法节点并移除输入的除法节点。
    # 这是一种优化，对于融合并非强制性的。
    divisors = [div.args[1] for div in all_input_nodes]  # 获取所有除法节点的除数参数
    assert all(divisor == divisors[0] for divisor in divisors)  # 断言所有除数参数相同
    with graph.inserting_after(cat_node):  # 在 concat 节点之后插入新节点
        div_node = call_function(graph, last_input_node.target, (cat_node, divisors[0]))

    # 创建一个新的通信/全局归约节点。
    last_comm_node = last_comm_block.comm_node  # 获取最后一个通信节点
    last_wait_node = last_comm_block.wait_nodes[0]  # 获取最后一个等待节点
    with graph.inserting_after(div_node):  # 在除法节点之后插入新节点
        flatten_args, spec = tree_flatten((last_comm_node.args, last_comm_node.kwargs))
        flatten_args[0] = div_node
        args, kwargs = tree_unflatten(flatten_args, spec)
        fused_comm_node = call_function(graph, last_comm_node.target, args, kwargs)

    # 创建一个新的等待节点。
    with graph.inserting_after(fused_comm_node):  # 在新的通信节点之后插入新节点
        flatten_args, spec = tree_flatten((last_wait_node.args, last_wait_node.kwargs))
        flatten_args[0] = fused_comm_node
        args, kwargs = tree_unflatten(flatten_args, spec)
        fused_wait_node = call_function(graph, last_wait_node.target, args, kwargs)

    # 将融合的全局归约及其参数移动到输入节点之后。
    nodes_to_move = cat_inputs + [cat_node, div_node, fused_comm_node, fused_wait_node]
    move_block_after(nodes_to_move, last_input_node)
    # 返回一个 CommBlock 对象，配置如下参数：
    return CommBlock(
        # 设置 CommBlock 对象的 shape 属性，使用 cat_node 的 tensor_meta 的 shape 属性进行赋值
        shape=cast(TensorMetadata, cat_node.meta.get("tensor_meta")).shape,
        # 设置 CommBlock 对象的 node_list 属性，包含 fused_comm_node 和 fused_wait_node
        node_list=[fused_comm_node, fused_wait_node],
        # 设置 CommBlock 对象的 wait_nodes 属性，仅包含 fused_wait_node
        wait_nodes=[fused_wait_node],
        # 设置 CommBlock 对象的 comm_node 属性，使用 fused_comm_node 进行赋值
        comm_node=fused_comm_node,
        # 设置 CommBlock 对象的 inputs 属性，包含 div_node
        inputs=[div_node],
        # 设置 CommBlock 对象的 outputs 属性，包含 fused_wait_node（作为集合的一个元素）
        outputs={fused_wait_node},
    )
def _scatter_fused_allreduce_waits(
    graph: fx.Graph,
    fused_comm_block: CommBlock,
    orig_comm_blocks: List[CommBlock],
    node_indices: Dict[fx.Node, int],
    split_and_reshape: bool = True,
) -> None:
    """
    Scatters the result of the fused communication node to the original users.
    If the fused method is concat splitting the output and reshape will be inserted,
    before inserting getitem. Otherwise getitem will be used as the users of the
    wait node.
    """
    # 在改变顺序之前，我们需要获取 orig_comm_blocks 中最后一个等待节点的索引。
    # 此索引将稍后用于确定需要移动哪些用户节点以保持正确的拓扑排序顺序。
    last_wait_node_idx = 0
    for node in graph.nodes:
        # 更新最后一个等待节点的索引，确保在字典 node_indices 中找到对应节点的索引值
        last_wait_node_idx = max(
            node_indices.get(node, last_wait_node_idx), last_wait_node_idx
        )
        # 如果当前节点等于 orig_comm_blocks 中最后一个通信块的等待节点，则停止循环
        if node == orig_comm_blocks[-1].wait_nodes[0]:
            break

    if split_and_reshape:
        # 获取融合通信块的等待节点
        fused_wait_node = fused_comm_block.wait_nodes[0]
        # 在 fused_wait_node 后插入新的计算图节点
        with graph.inserting_after(fused_wait_node):
            # 调用函数进行节点分割操作，生成 split_node
            split_node = call_function(
                graph,
                aten.split,
                (
                    fused_wait_node,
                    [math.prod(cast(List[int], cb.shape)) for cb in orig_comm_blocks],
                ),
            )
        # 在 split_node 后插入新的计算图节点
        with graph.inserting_after(split_node):
            fused_outputs = []
            for idx, comm_block in enumerate(orig_comm_blocks):
                # 调用函数获取分割后的索引节点 split_idx_node
                split_idx_node = call_function(
                    graph, operator.getitem, (split_node, idx)
                )
                # 在 split_idx_node 后插入新的计算图节点
                with graph.inserting_after(split_idx_node):
                    # 调用函数进行形状重塑操作，生成 fused_outputs 列表
                    fused_outputs.append(
                        call_function(
                            graph, aten.reshape, (split_idx_node, comm_block.shape)
                        )
                    )
    else:
        # 如果不进行分割和重塑操作，则直接使用融合通信块的等待节点作为输出
        fused_outputs = fused_comm_block.wait_nodes

    # 记录顺序错误的节点
    incorrect_order_nodes = []
    for comm_block, fused_output in zip(orig_comm_blocks, fused_outputs):
        # orig_comm_blocks 的某些后代用户节点可能在融合 all_reduce 之前被调度。
        # 例如，第一个 all_reduce 的用户节点可能在第二个 all_reduce 之前被调度。
        # 由于融合 all_reduce 插入在最后一个 all_reduce 之后，顺序可能是错误的。
        # `incorrect_order_nodes` 记录这些节点。

        # 获取通信块的等待节点 orig_wait
        orig_wait = comm_block.wait_nodes[0]
        # 使用 deque 构建节点队列，包含 orig_wait 的所有用户节点
        nodes = collections.deque(list(orig_wait.users))
        while nodes:
            user_node = nodes.popleft()
            if not isinstance(user_node, fx.Node):
                continue
            # 如果用户节点的索引小于 last_wait_node_idx，则记录为顺序错误的节点
            if node_indices[user_node] < last_wait_node_idx:
                incorrect_order_nodes.append(user_node)
                nodes.extend(list(user_node.users))

        # 将 orig_wait 的所有用途替换为 fused_output
        orig_wait.replace_all_uses_with(fused_output)

    # 获取最后一个融合输出结果
    last_fused_result = fused_outputs[0]
    fused_outputs_set = set(fused_outputs)
    for node in graph.nodes:
        # 如果节点在 fused_outputs_set 中，则更新 last_fused_result
        if node in fused_outputs_set:
            last_fused_result = node

    # 将顺序错误的节点移动到最后一个融合结果节点之后
    incorrect_order_nodes = sorted(
        incorrect_order_nodes, key=lambda node: node_indices[node]
    )
    move_block_after(incorrect_order_nodes, last_fused_result)
# 对一组 allreduce CommBlock 进行融合，生成一个整体的 CommBlock
def _fuse_allreduce(
    graph: fx.Graph,
    comm_blocks: List[CommBlock],
    node_indices: Dict[fx.Node, int],
    use_concat: bool,
) -> CommBlock:
    """Given a list of allreduce CommBlock, fuse the CommBlocks into one CommBlock."""

    # 如果只有一个 CommBlock，则直接返回该 CommBlock
    if len(comm_blocks) == 1:
        return comm_blocks[0]

    # 找到所有 CommBlocks 的最后一个输入节点作为新集合操作的插入点
    last_input_node = comm_blocks[0].inputs[0]
    last_input_index = -1
    all_input_nodes = []
    for comm_block in comm_blocks:
        input_node = comm_block.inputs[0]
        all_input_nodes.append(input_node)
        index = node_indices[input_node]
        if index >= last_input_index:
            assert index != last_input_index
            last_input_node = input_node
            last_input_index = index

    # 根据 use_concat 决定使用哪种方式进行融合操作
    if use_concat:
        fused_comm_block = _fuse_allreduce_by_concat(
            graph, last_input_node, all_input_nodes, comm_blocks[-1]
        )
    else:
        fused_comm_block = _fuse_with_coalesced_op(
            graph, last_input_node, all_input_nodes, comm_blocks[-1]
        )

    # 散列融合后的 allreduce 操作，等待完成
    _scatter_fused_allreduce_waits(
        graph, fused_comm_block, comm_blocks, node_indices, split_and_reshape=use_concat
    )

    # 删除原始 CommBlocks 中的等待节点和通信节点
    for comm_block in comm_blocks:
        for wait in comm_block.wait_nodes:
            graph.erase_node(wait)
        graph.erase_node(comm_block.comm_node)
    # 消除死代码
    graph.eliminate_dead_code()

    # 返回融合后的 CommBlock
    return fused_comm_block


# 将一组 CommBlock 按照内存桶大小进行融合生成器
def _bucket_size_fusion(
    graph: fx.Graph, comm_blocks: List[CommBlock], bucket_size_mb: int
) -> Generator[List[CommBlock], None, None]:
    MB = 1024**2
    bucket_size = 1 * MB
    bucket_cap_size = bucket_size_mb * MB
    curr_size = 0
    curr_blocks = []

    count = 0
    fuse_count = 0
    for i, block in enumerate(comm_blocks):
        curr_blocks.append(block)
        itemsize = block.comm_node.meta["tensor_meta"].dtype.itemsize
        curr_size += cast(torch.Size, block.shape).numel() * itemsize
        count += 1
        # 如果当前大小小于桶大小并且不是最后一个块，则继续添加块
        if curr_size < bucket_size and i != len(comm_blocks) - 1:
            continue

        # 增加融合计数器并打印调试信息（仅在分布式环境下的主节点）
        fuse_count += 1
        if torch.distributed.get_rank() == 0:
            logger.info(
                "DDP bucketing: block%d, count=%d, curr_size=%d, bucket_size=%d",
                fuse_count,
                count,
                curr_size,
                bucket_size,
            )

        # 设置调试计数器
        counters["inductor"]["ddp_buckets"] = fuse_count
        # 生成当前桶的 CommBlocks
        yield curr_blocks

        # 更新桶大小为桶容量大小，并重置当前块列表和当前大小
        bucket_size = bucket_cap_size
        curr_blocks = []
        curr_size = 0
        count = 0


# 对 DDP 通信进行融合处理
def _fuse_ddp_communication(
    graph: fx.Graph, algorithm_fn: Callable[..., Any], fusion_fn: Callable[..., Any]
) -> None:
    # 从图的末尾开始，向前查找直到找到 "output" 操作
    for output in reversed(graph.nodes):
        if output.op == "output":
            break
    # 定义一个函数 ddp_reducer_filter，接受一个 CommBlock 参数并返回布尔值
    def ddp_reducer_filter(block: CommBlock) -> bool:
        # 检查 block.comm_node.args[0] 是否为 fx.Node 类型，并且其 target 是否为 aten.div.Tensor
        if (
            not isinstance(block.comm_node.args[0], fx.Node)
            or block.comm_node.args[0].target != aten.div.Tensor
        ):
            return False

        # 检查 block.wait_nodes[0].users 的长度是否不为 1，如果不是则返回 False
        if len(block.wait_nodes[0].users) != 1:
            # gradient/wait 节点应只被一个用户使用
            return False

        # 判断两种情况：
        # 1. 如果 output 不在 block.wait_nodes[0].users 中
        # 2. 如果 block.wait_nodes[0].users 的第一个用户的 target 不是 aten.copy_.default
        if (
            output not in block.wait_nodes[0].users
            and next(iter(block.wait_nodes[0].users)).target != aten.copy_.default
        ):
            return False

        # 如果以上条件都满足，则返回 True
        return True

    # 定义一个包含两个元素的元组 ops，包含 torch.ops._c10d_functional.all_reduce_.default 和 torch.ops._c10d_functional.all_reduce.default
    ops = (
        torch.ops._c10d_functional.all_reduce_.default,
        torch.ops._c10d_functional.all_reduce.default,
    )
    
    # 调用 get_all_comm_blocks 函数，传入 graph、ops 和 comm_filter=ddp_reducer_filter 作为参数，返回 comm_blocks
    comm_blocks = get_all_comm_blocks(graph, ops, comm_filter=ddp_reducer_filter)
    
    # 创建一个字典 node_indices，将 graph 中的每个节点与其索引对应起来
    node_indices = {node: i for i, node in enumerate(graph.nodes)}

    # 使用 algorithm_fn 函数处理 graph 和 comm_blocks 中的每个 block
    for block in algorithm_fn(graph, comm_blocks):
        # 对每个 block 调用 fusion_fn 函数，并传入 graph、block 和 node_indices 作为参数
        fusion_fn(graph, block, node_indices)
# 将 DDP 通信与合并操作融合
def fuse_ddp_with_coalesced_op(graph: fx.Graph, bucket_size_mb: int) -> None:
    # 调用 _fuse_ddp_communication 函数，传入图形对象 graph，以及使用 _bucket_size_fusion 和 _fuse_allreduce 的部分函数
    _fuse_ddp_communication(
        graph,
        partial(_bucket_size_fusion, bucket_size_mb=bucket_size_mb),  # 部分函数 _bucket_size_fusion，使用指定的 bucket_size_mb
        partial(_fuse_allreduce, use_concat=False),  # 部分函数 _fuse_allreduce，使用参数 use_concat=False
    )


# 将 DDP 通信与连接操作融合
def fuse_ddp_with_concat_op(graph: fx.Graph, bucket_size_mb: int) -> None:
    # 调用 _fuse_ddp_communication 函数，传入图形对象 graph，以及使用 _bucket_size_fusion 和 _fuse_allreduce 的部分函数
    _fuse_ddp_communication(
        graph,
        partial(_bucket_size_fusion, bucket_size_mb=bucket_size_mb),  # 部分函数 _bucket_size_fusion，使用指定的 bucket_size_mb
        partial(_fuse_allreduce, use_concat=True),  # 部分函数 _fuse_allreduce，使用参数 use_concat=True
    )


# 调度通信等待节点
def schedule_comm_wait(graph: fx.Graph) -> None:
    """
    延迟执行 allreduce 的等待张量，直到其首个用户。

    此算法考虑等待节点的中间用户，如 split、getitem，同时也调度这些中间用户。
    这将产生更好的重叠结果。
    """
    # 定义需要处理的操作列表
    ops = (
        torch.ops._c10d_functional.all_reduce_.default,
        torch.ops._c10d_functional.all_reduce.default,
        torch.ops._c10d_functional.all_reduce_coalesced.default,
        torch.ops._c10d_functional.all_reduce_coalesced_.default,
    )
    # 获取所有通信块
    comm_blocks = get_all_comm_blocks(graph, ops)
    if not comm_blocks:
        return

    # 查找所有最终用户
    allreduce_users: Set[fx.Node] = set()
    for allreduce in comm_blocks:
        for output in allreduce.outputs:
            allreduce_users.update(output.users)

    # 为每个节点建立索引映射
    node_indices = {node: i for i, node in enumerate(graph.nodes)}
    for allreduce in comm_blocks:
        # 找到最早的/第一个用户节点 -- target_node
        assert (
            len(allreduce.outputs) >= 1
        ), f"Found a allreduce that has zero outputs/users -- {allreduce}."
        # 初始化目标节点以避免类型问题
        target_node = next(iter(next(iter(allreduce.outputs)).users))
        target_node_index = 2**31
        for user in (user for output in allreduce.outputs for user in output.users):
            index = node_indices[user]
            if index < target_node_index:
                target_node = user
                target_node_index = index

        # 将等待节点和通信块中后续的所有节点移动到第一个用户节点 target_node 之前
        wait_idx = -1
        for wait_idx, node in enumerate(allreduce.node_list):
            if node == allreduce.wait_nodes[0]:
                break
        assert wait_idx >= 0
        move_block_before(allreduce.node_list[wait_idx:], target_node)


# 融合 DDP 通信
def fuse_ddp_communication(
    graph: fx.Graph, passes: List[Union[Callable[..., None], str]], bucket_size_mb: int
) -> None:
    # 使用 enumerate() 函数遍历 passes 列表，获取每个元素和对应的索引 i
    for i, pa in enumerate(passes):
        # 使用 GraphTransformObserver 类进行图形变换观察，设置观察器名称为 fuse_ddp_communication_pass_{i}
        # 参数包括 owning_module，观察器名称，以及日志 URL
        with GraphTransformObserver(
            graph.owning_module,
            f"fuse_ddp_communication_pass_{i}",
            config.trace.log_url_for_graph_xform,
        ):
            # 检查 pa 是否为字符串类型
            if isinstance(pa, str):
                # 如果 pa 是字符串，则从全局变量中获取对应的函数对象
                func = globals()[pa]
            else:
                # 如果 pa 不是字符串，则直接将 pa 赋值给 func
                func = pa
            # 检查 func 函数的参数列表中是否包含 "bucket_size_mb" 参数
            if "bucket_size_mb" in {
                v.name for v in inspect.signature(func).parameters.values()
            }:
                # 如果包含 "bucket_size_mb" 参数，则调用 func 函数，并传入 graph 和 bucket_size_mb 参数
                func(graph, bucket_size_mb=bucket_size_mb)
            else:
                # 如果不包含 "bucket_size_mb" 参数，则只传入 graph 参数调用 func 函数
                func(graph)
```