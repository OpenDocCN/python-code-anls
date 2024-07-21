# `.\pytorch\torch\distributed\_spmd\graph_optimization.py`

```
# 设置类型提示允许未声明的函数
# 所有权：["oncall: distributed"]
import collections  # 导入collections模块，用于创建默认字典等高级数据结构
import itertools  # 导入itertools模块，提供了创建和操作迭代器的函数
import logging  # 导入logging模块，用于记录日志信息
import operator  # 导入operator模块，提供了一些常见的运算符函数
import tempfile  # 导入tempfile模块，用于创建临时文件和目录
import time  # 导入time模块，提供时间相关的函数

from dataclasses import dataclass, field  # 导入dataclass和field，用于定义数据类和字段
from functools import wraps  # 导入wraps，用于装饰函数时保留原函数的元信息
from typing import (  # 导入类型提示相关的各种类型
    Any,
    Callable,
    cast,
    DefaultDict,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import torch  # 导入PyTorch库
import torch.fx as fx  # 导入torch.fx模块，用于处理PyTorch的功能模块
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode  # 导入FakeTensor相关类
from torch.distributed._spmd.graph_utils import (  # 导入图操作相关的函数和类
    CommType,
    dump_graphs_to_files,
    find_node,
    get_output,
    OP,
)
from torch.distributed._spmd.iter_graph_module import IterGraphModule  # 导入迭代图模块
from torch.fx.passes.shape_prop import TensorMetadata  # 导入Tensor元数据相关模块
from torch.utils import _pytree as pytree  # 导入_pytree模块，支持树形结构的操作
from torch.utils._pytree import tree_flatten, tree_unflatten  # 导入树形结构的扁平化和还原函数

logger: logging.Logger = logging.getLogger("graph_optimization")  # 创建名为'graph_optimization'的日志记录器对象
aten = torch.ops.aten  # 获取PyTorch操作aten的命名空间
fake_tensor_mode = FakeTensorMode()  # 创建FakeTensorMode对象用于处理FakeTensor

_optimized_func: Set[str] = set()  # 创建空的字符串集合，用于存储优化过的函数名
# 使用默认字典创建字典，键为目标优化操作，值为该操作的先决条件集合
_prerequisite_sets: DefaultDict[str, Set[str]] = collections.defaultdict(set)
# 使用默认字典创建字典，键为目标优化操作，值为该操作必须在之前应用的操作集合
_apply_before_sets: DefaultDict[str, Set[str]] = collections.defaultdict(set)
_dump_graph_folder: str = ""  # 创建空字符串，用于存储图形优化结果的文件夹路径


def enable_graph_optimization_dump(folder: str = ""):
    global _dump_graph_folder
    # 如果未提供文件夹路径，则创建临时文件夹
    if not folder:
        folder = tempfile.mkdtemp()
    # 将文件夹路径存储在全局变量中
    _dump_graph_folder = folder


# TODO(@fegin): 支持图优化的多次运行
# TODO(@fegin): 这种设计会导致循环导入，当一个优化操作无意中创建一个依赖关系循环时。因此，我们需要将此文件细分为更小的部分，以避免不正确的循环导入。
def graph_optimization_pass(
    prerequisites: Iterable[Callable],
    apply_after: Iterable[Callable],
) -> Callable:
    """定义图优化操作的约定。

    所有优化操作应该使用此装饰器进行封装。
    `prerequisites`用于注明此优化操作的先决优化操作。
    `apply_after`表示此装饰操作必须在`apply_after`中的优化操作之后应用。
    `prerequisites`和`apply_after`的区别在于，所有`prerequisites`中的操作必须在封装操作之前应用，而`apply_after`中的操作是可选的，
    但是如果`apply_after`中的某个操作已应用于图形，则必须在封装操作之前应用。
    优化器操作开发人员需要相应地添加这些字段，用户需要遵循这些限制以避免断言错误。

    当前设计有一个限制：用户只能应用优化一次。在某些情况下，我们可能需要多次运行相同的优化，例如，优化操作 -> 分析结果 -> 应用
    """
    # 定义一个装饰器函数，用于优化图模块的处理过程
    def inner(func: Callable) -> Callable:
        # 定义生成函数的唯一标识符的内部函数
        def make_key(func: Callable) -> str:
            return f"{func.__module__}.{func.__name__}"

        # 获取被装饰函数的标识符
        func_key = make_key(func)
        
        # 将传入的前提条件函数集合存入_prerequisite_sets字典中
        _prerequisite_sets[func_key] = {make_key(f) for f in prerequisites}
        
        # 对于每个在apply_after参数中的函数，将该函数不能在装饰函数之后应用的信息存入_apply_before_sets字典中
        for apply_after_pass in apply_after:
            _apply_before_sets[make_key(apply_after_pass)].add(func_key)

        # 定义装饰后的函数，用于应用优化处理到给定的图模块
        @wraps(func)
        def pass_wrapper(
            gm: Union[fx.GraphModule, IterGraphModule], *args: Any, **kwargs: Any
        ) -> None:
            begin = time.time()
            
            # 断言第一个参数gm必须是fx.GraphModule或IterGraphModule类型的实例
            assert isinstance(gm, (fx.GraphModule, IterGraphModule)), (
                "The first argument of the pass must be either "
                "fx.GraphModule or IterGraphModule."
            )
            
            # 断言当前函数未被优化过，即不能重复应用同一优化函数
            assert func_key not in _optimized_func, f"Cannot apply {func_key} twice."
            
            # 检查是否存在无效的先决条件函数
            invalid_passes = _apply_before_sets[func_key].intersection(_optimized_func)
            assert (
                not invalid_passes
            ), f"{invalid_passes} must be applied after {func_key}."
            
            # 断言当前函数的所有先决条件函数都已被优化应用
            assert _prerequisite_sets[func_key].issubset(_optimized_func), (
                f"{_prerequisite_sets[func_key] - _optimized_func} are the "
                f"prerequisites of {func_key} but are not applied. "
                f"Applied passes are {_optimized_func}."
            )

            # 执行被装饰的函数，对图进行必要的处理
            func(gm, *args, **kwargs)
            gm.graph.lint()
            gm.graph.eliminate_dead_code()
            gm.recompile()
            
            # 将当前函数标识符添加到已优化函数集合中
            _optimized_func.add(func_key)

            # 如果设置了_dump_graph_folder，则将处理后的图保存到指定文件夹中
            prefix = f"after_{func.__name__}"
            if _dump_graph_folder:
                if isinstance(gm, IterGraphModule):
                    dump_graphs_to_files(
                        {
                            f"{prefix}_setup_gm": gm.setup_gm,
                            f"{prefix}_main_gm": gm.main_gm,
                            f"{prefix}_cleanup_gm": gm.cleanup_gm,
                        },
                        _dump_graph_folder,
                    )
                else:
                    dump_graphs_to_files({prefix: gm}, _dump_graph_folder)

            # 记录优化过程所花费的时间
            logger.info("Spent %f seconds applying %s", time.time() - begin, func_key)

        # 返回装饰后的函数
        return pass_wrapper

    # 返回装饰器函数inner
    return inner
# 使用 dataclass 装饰器创建 CommBlock 类，允许对象进行不安全的哈希操作
@dataclass(unsafe_hash=True)
class CommBlock:
    # 类的属性定义：表示形状的可选的 torch.Size 对象
    shape: Optional[torch.Size]
    # 包含节点列表的列表
    node_list: List[fx.Node]
    # 输入节点列表
    inputs: List[fx.Node]
    # 等待节点列表
    wait_nodes: List[fx.Node]
    # 通信节点
    comm_node: fx.Node
    # 输出节点的集合
    outputs: Set[fx.Node]


def get_comm_block(comm_node: fx.Node) -> CommBlock:
    """Find out all the nodes belong to this communcation given a collective node (e.g., allreduce).

    Args:
        comm_node(fx.Node): The target communication/collective node.

    Returns:
        The CommBlock that encapsulates the related nodes (e.g., wait_node) of
        the given comm_node.
    """
    # 设置最大等待距离，防止意外导致的无限循环
    MAX_WAIT_DISTANCE = 5
    # 初始化节点列表和等待节点列表为空列表
    node_list = []
    wait_nodes = []
    # 使用 pytree.arg_tree_leaves 获取通信节点的输入参数列表
    inputs = pytree.arg_tree_leaves(*comm_node.args, **comm_node.kwargs)
    # 过滤输入参数中的节点对象，构成输入节点列表
    input_nodes = [inp for inp in inputs if isinstance(inp, fx.Node)]
    # 初始化距离为 0
    distance = 0
    # 定义等待节点名称前缀和非终端用户节点名称
    wait_prefixes = ("wait_comm", "wait_tensor")
    non_end_users_nodes = ("split", "reshape", "getitem", "detach", "alias")

    # 使用双端队列存储节点和距离
    nodes = collections.deque([comm_node, None])
    while nodes and distance < MAX_WAIT_DISTANCE:
        node = nodes.popleft()
        if node is None:
            distance += 1
            if nodes:
                nodes.append(None)
            continue
        # 将节点添加到节点列表中
        node_list.append(node)
        # 如果节点名称以等待节点前缀开头，将其添加到等待节点列表中
        if node.name.startswith(wait_prefixes):
            wait_nodes.append(node)
        else:
            # 遍历节点的用户，将其添加到节点队列中
            for child in node.users:
                if isinstance(child, fx.Node):
                    nodes.append(child)

    # 如果未找到等待节点，则抛出运行时错误
    if not wait_nodes:
        raise RuntimeError(
            f"The wait nodes are too far away from the comm node {comm_node}."
        )

    # 标识该通信块的所有输出节点
    outputs: Set[fx.Node] = set()
    nodes = collections.deque(wait_nodes)
    while nodes:
        node = nodes.popleft()
        assert node is not None
        for user in node.users:
            # 如果用户是节点并且其名称以非终端用户节点名称开头，则将其添加到节点队列和节点列表中
            if isinstance(user, fx.Node) and user.name.startswith(non_end_users_nodes):
                nodes.append(user)
                node_list.append(user)
            else:
                # 否则将该节点添加到输出节点集合中
                outputs.add(node)
                break

    # TODO: 填充所有张量元数据并移除默认值
    tensor_meta = input_nodes[0].meta.get("tensor_meta", None)
    # 返回 CommBlock 对象，包含形状、节点列表、等待节点列表、通信节点、输入节点列表和输出节点集合
    return CommBlock(
        shape=torch.Size(int(s) for s in tensor_meta.shape) if tensor_meta else None,
        node_list=node_list,
        wait_nodes=wait_nodes,
        comm_node=comm_node,
        inputs=input_nodes,
        outputs=outputs,
    )


def get_all_comm_blocks(
    gm: IterGraphModule, comm_ops: Union[Tuple[str, ...], str]
) -> List[CommBlock]:
    # 返回包含所有与给定通信操作前缀匹配的节点的 CommBlock 对象列表
    return [
        get_comm_block(node)
        for node in gm.graph.nodes
        if node.name.startswith(comm_ops)
    ]


def _create_meta_val(
    fake_tensor_mode: FakeTensorMode,
    val: FakeTensor,
) -> FakeTensor:
    # TODO: 修复内存格式
    pass
    # 返回一个模拟张量对象
    return FakeTensor(
        # 使用给定的模式创建一个假张量
        fake_tensor_mode,
        # 使用空张量作为基础，与给定的张量形状和数据类型相匹配
        torch.empty(
            val.shape,                 # 使用给定张量的形状
            dtype=val.dtype,           # 使用给定张量的数据类型
            device="meta",             # 将张量放置在特定的设备上（这里是 meta 设备）
            requires_grad=val.requires_grad,  # 继承给定张量的梯度需求属性
        ),
        val.device,                    # 使用给定张量的设备信息
    )
def _create_meta_tensor_meta(
    fake_tensor_mode: FakeTensorMode,
    val: FakeTensor,
) -> TensorMetadata:
    # 创建并返回包含张量元数据的对象，包括形状、数据类型、是否需要梯度、步长
    return TensorMetadata(
        shape=val.shape,
        dtype=val.dtype,
        requires_grad=val.requires_grad,
        stride=val.stride,  # type: ignore[arg-type]
        # TODO: fix these value
        memory_format=None,
        is_quantized=False,
        qparams={},
    )


def _call_function(
    gm: IterGraphModule,
    fake_tensor_mode: FakeTensorMode,
    meta_val: Optional[FakeTensor],
    function: Any,
    *args: Any,
    **kwargs: Any,
) -> fx.Node:
    # 在计算图模块中调用指定的函数，并生成一个计算图节点
    node = gm.graph.call_function(function, args, kwargs)

    if meta_val is None:
        flat_args, spec = tree_flatten((args, kwargs))
        new_flat_args = []
        memory_format = None
        # 扁平化参数列表，并为非节点参数创建新的伪值元数据
        for arg in flat_args:
            if not isinstance(arg, fx.Node):
                new_flat_args.append(arg)
                continue
            val = arg.meta["val"]
            new_flat_args.append(_create_meta_val(fake_tensor_mode, val))

        fake_args, fake_kwargs = tree_unflatten(new_flat_args, spec)
        # 使用伪参数调用函数以获取新的伪值
        new_meta_val = function(*fake_args, **fake_kwargs)
    else:
        new_meta_val = meta_val
    # 将新的伪值设置为节点的元数据
    node.meta["val"] = new_meta_val
    # 创建并设置张量元数据并将其添加到节点的元数据中
    node.meta["tensor_meta"] = _create_meta_tensor_meta(fake_tensor_mode, new_meta_val)
    return node


def _scatter_wait_result(
    gm: IterGraphModule,
    fused_comm_block: CommBlock,
    comm_blocks: List[CommBlock],
    node_indices: Dict[fx.Node, int],
) -> None:
    """Scatter the result of the fused communication node to the original users -- splitting the output and reshape each subitem."""
    # 查找与融合通信节点相关的最后一个等待节点的索引
    last_wait_node_idx = 0
    for node in gm.graph.nodes:
        if node == fused_comm_block.comm_node:
            break
        last_wait_node_idx = max(
            node_indices.get(node, last_wait_node_idx), last_wait_node_idx
        )

    fused_comm_node = fused_comm_block.comm_node
    fused_wait_node = fused_comm_block.wait_nodes[0]

    with gm.graph.inserting_after(fused_wait_node):
        # 在等待节点之后插入调用函数节点，以分割等待节点的输出并重塑每个子项
        split_node = gm.graph.call_function(
            aten.split,
            (
                fused_wait_node,
                # TODO(@fegin): support symbolic shapes
                [int(cast(torch.Size, cb.shape).numel()) for cb in comm_blocks],
            ),
        )

    # 分散分割结果
    need_sort_nodes = []
    last_split_reshape_node = split_node
    # 在图中的 split_node 之后插入新的节点
    with gm.graph.inserting_after(split_node):
        # 遍历通信块列表中的每个通信块
        for idx, comm_block in enumerate(comm_blocks):
            # 对于原始的等待节点，找到第一个等待节点
            orig_wait = comm_block.wait_nodes[0]
            # 使用一个队列来处理用户节点
            nodes = collections.deque(list(orig_wait.users))
            # 将用户节点移动到正确的拓扑排序位置，即在最后一个融合allreduce结果之后
            while nodes:
                user_node = nodes.popleft()
                # 如果用户节点不是 fx.Node 类型则跳过
                if not isinstance(user_node, fx.Node):
                    continue
                # 如果用户节点的索引小于 last_wait_node_idx，则需要排序
                if node_indices[user_node] < last_wait_node_idx:
                    need_sort_nodes.append(user_node)
                    nodes.extend(list(user_node.users))

            # 创建调用 operator.getitem 的节点，获取 split_node 的索引处的节点
            split_idx_node = gm.graph.call_function(operator.getitem, (split_node, idx))
            # 在 split_idx_node 之后插入节点，调用 aten.reshape，形状为 comm_block.shape
            with gm.graph.inserting_after(split_idx_node):
                wait_output_node = gm.graph.call_function(
                    aten.reshape, (split_idx_node, comm_block.shape)
                )
            # 替换图中所有使用 orig_wait 的节点为 wait_output_node
            gm.graph.node_replace_all_uses_with(orig_wait, wait_output_node)

        # 如果 last_split_reshape_node 等于 split_node，则更新为 wait_output_node
        if last_split_reshape_node == split_node:
            last_split_reshape_node = wait_output_node  # type: ignore[possibly-undefined]

    # 对需要排序的节点列表进行排序，按照 node_indices[node] 的顺序
    need_sort_nodes = sorted(need_sort_nodes, key=lambda node: node_indices[node])
    # 将 need_sort_nodes 移动到 last_split_reshape_node 之后
    gm.graph.move_after(need_sort_nodes, last_split_reshape_node)

    # 消除图中的死代码
    gm.graph.eliminate_dead_code()
# 将 CommBlocks 使用 concat 进行融合，给定 CommBlock 的列表（仅包含 allreduce 类型）。
def _fuse_with_cat(
    gm: IterGraphModule,
    comm_blocks: List[CommBlock],
    node_indices: Dict[fx.Node, int],
) -> CommBlock:
    """Fuse the CommBlocks using concat given a list of CommBlock (only allreduce)."""
    # 找到最后一个输入节点。
    last_input_node = comm_blocks[0].inputs[0]
    last_input_index = -1
    all_input_nodes = []
    for comm_block in comm_blocks:
        input_node = comm_block.inputs[0]
        # 如果输入节点是克隆节点，则这是基于 CommTensor 的实现。
        if input_node.name.startswith("clone"):
            input_node = cast(fx.Node, input_node.args[0])
        all_input_nodes.append(input_node)
        index = node_indices[input_node]
        if index >= last_input_index:
            assert index != last_input_index
            last_input_node = input_node
            last_input_index = index

    # 在最后一个输入准备好之后，展平所有输入。
    with gm.graph.inserting_after(last_input_node):
        cat_inputs = []
        for input_node in all_input_nodes:
            cat_inputs.append(
                _call_function(
                    gm, fake_tensor_mode, None, aten.flatten.using_ints, input_node
                )
            )

    # 在 cat_inputs[0] 之后插入节点。
    with gm.graph.inserting_after(cat_inputs[0]):
        cat_node = _call_function(gm, fake_tensor_mode, None, aten.cat, cat_inputs)

    # 创建一个新的 Comm 节点。
    last_comm = comm_blocks[-1]
    last_comm_node = last_comm.comm_node
    last_wait_node = last_comm.wait_nodes[0]
    with gm.graph.inserting_after(cat_node):
        # 展平参数并调用函数创建融合的 Comm 节点。
        flatten_args, spec = tree_flatten((last_comm_node.args, last_comm_node.kwargs))
        flatten_args[0] = cat_node
        args, kwargs = tree_unflatten(flatten_args, spec)
        fused_comm_node = _call_function(
            gm,
            fake_tensor_mode,
            cat_node.meta["val"],
            last_comm_node.target,
            *args,
            **kwargs,
        )

    # 创建一个新的 Wait 节点。
    with gm.graph.inserting_after(fused_comm_node):
        # 展平参数并调用函数创建融合的 Wait 节点。
        flatten_args, spec = tree_flatten((last_wait_node.args, last_wait_node.kwargs))
        flatten_args[0] = fused_comm_node
        args, kwargs = tree_unflatten(flatten_args, spec)
        fused_wait_node = _call_function(
            gm,
            fake_tensor_mode,
            cat_node.meta["val"],
            last_wait_node.target,
            *args,
            **kwargs,
        )

    # 将融合的 Comm 节点及其参数移动到源节点之后。
    nodes_to_move = cat_inputs + [cat_node, fused_comm_node, fused_wait_node]
    gm.graph.move_after(nodes_to_move, last_input_node)

    # 获取 cat_node 的 tensor_meta，创建融合后的 CommBlock。
    tensor_meta = cat_node.meta.get("tensor_meta")
    fused_comm_block = CommBlock(
        shape=tensor_meta.shape,  # type: ignore[union-attr]
        node_list=[fused_comm_node, fused_wait_node],
        wait_nodes=[fused_wait_node],
        comm_node=fused_comm_node,
        inputs=[cat_node],
        outputs={fused_wait_node},
    )
    # 调用名为 `_scatter_wait_result` 的函数，并传入参数 `gm`, `fused_comm_block`, `comm_blocks`, `node_indices`
    _scatter_wait_result(gm, fused_comm_block, comm_blocks, node_indices)
    # 返回变量 `fused_comm_block` 的值作为函数的结果
    return fused_comm_block
# 对于给定的图模块 `gm` 和通信块列表 `comm_blocks`，加速通信操作的执行顺序
def _expedite_comm_ops(gm: IterGraphModule, comm_blocks: List[CommBlock]) -> None:
    # 创建一个字典，将图节点映射为它们的索引位置
    node_indices = {node: i for i, node in enumerate(gm.graph.nodes)}
    # 遍历每个通信块
    for comm_block in comm_blocks:
        # 初始化最后一个输入节点为通信块的通信节点
        last_input = comm_block.comm_node
        last_input_idx = -1
        # 遍历通信块的输入节点列表
        for input in comm_block.inputs:
            # 获取输入节点的索引
            input_idx = node_indices[input]
            # 如果当前输入节点的索引大于上一个输入节点的索引，更新最后一个输入节点和索引
            if input_idx > last_input_idx:
                last_input = input
                last_input_idx = input_idx
        # 将最后一个输入节点添加到图中作为通信节点的后继节点
        gm.graph.node_append(last_input, comm_block.comm_node)


@graph_optimization_pass(
    prerequisites=[],  # 优化前提为空列表
    apply_after=[],   # 在其它优化之后应用为空列表
)
def comm_fusion_with_concat(
    gm: IterGraphModule,
    bucket_size_mb: int,
) -> None:
    """Run fuse communication with concat.

    This implementation uses concat to concat the bucketed gradients.
    """
    # 获取所有类型为ALLREDUCE的通信块列表
    comm_blocks = get_all_comm_blocks(gm, (CommType.ALLREDUCE, "all_reduce"))
    # 首先确保所有的allreduce操作立即在梯度之后执行
    _expedite_comm_ops(gm, comm_blocks)
    # 根据新的顺序重新获取通信块列表
    comm_blocks = get_all_comm_blocks(gm, (CommType.ALLREDUCE, "all_reduce"))
    # 创建一个字典，将图节点映射为它们的索引位置
    node_indices = {node: i for i, node in enumerate(gm.graph.nodes)}

    # 设置默认的单个桶大小为1MB
    bucket_size = 1 * 1024**2
    # 将桶的容量大小设为指定的MB大小乘以1024的平方
    bucket_cap_size = bucket_size_mb * 1024**2
    # 初始化起始索引、结束索引和当前桶大小为0
    begin = end = curr_size = 0
    # 遍历所有通信块
    while end < len(comm_blocks):
        # TODO: determine the dtype  # 确定数据类型的具体实现
        # 计算当前通信块的大小并更新当前桶的大小
        curr_size += cast(torch.Size, comm_blocks[end].shape).numel() * 4
        end += 1
        # 如果当前桶大小小于指定的桶大小，继续添加通信块到当前桶中
        if curr_size < bucket_size:
            continue
        # 对当前桶中的通信块执行融合操作
        _fuse_with_cat(gm, comm_blocks[begin:end], node_indices)
        # 将下一个桶的大小设置为指定的桶容量大小
        bucket_size = bucket_cap_size
        # 更新起始索引和重置当前桶大小
        begin = end
        curr_size = 0
    else:
        # 如果起始索引小于通信块列表的长度，对剩余的通信块执行融合操作
        if begin < len(comm_blocks):
            _fuse_with_cat(gm, comm_blocks[begin:end], node_indices)


@graph_optimization_pass(
    prerequisites=[comm_fusion_with_concat],  # 优化前提为融合通信和concat
    apply_after=[],  # 在其它优化之后应用为空列表
)
def schedule_comm_wait(gm: IterGraphModule) -> None:
    """Delay the execution of wait tensors of allreduce until its first user."""
    # 获取所有类型为ALLREDUCE的通信块列表
    comm_blocks = get_all_comm_blocks(gm, (CommType.ALLREDUCE, "all_reduce"))

    # 查找所有allreduce操作的最终用户
    allreduce_users: Set[fx.Node] = set()
    # 遍历所有通信块
    for allreduce in comm_blocks:
        # 遍历每个allreduce操作的输出节点
        for output in allreduce.outputs:
            # 将输出节点的用户添加到最终用户集合中
            allreduce_users.update(output.users)

    # 创建一个字典，将图节点映射为它们的索引位置
    node_indices = {node: i for i, node in enumerate(gm.graph.nodes)}
    # 遍历通信块中的所有 allreduce 操作
    for allreduce in comm_blocks:
        # 断言确保每个 allreduce 操作至少有一个输出
        assert (
            len(allreduce.outputs) >= 1
        ), f"Found a allreduce that has zero outputs/users -- {allreduce}."
        
        # 初始化目标节点为第一个输出的第一个用户
        target_node = next(iter(next(iter(allreduce.outputs)).users))
        
        # 初始化目标节点索引为一个非常大的值，2 的 31 次方
        target_node_index = 2**31
        
        # 遍历所有输出的所有用户，找到索引最小的用户作为目标节点
        for user in (user for output in allreduce.outputs for user in output.users):
            index = node_indices[user]
            if index < target_node_index:
                target_node = user
                target_node_index = index

        # 找到等待节点并将其及之后的所有输出节点移动到最早的用户节点之前
        wait_idx = -1
        for wait_idx, node in enumerate(allreduce.node_list):
            if node == allreduce.wait_nodes[0]:
                break
        assert wait_idx >= 0
        
        # 确保找到等待节点，然后通过图操作将其之后的节点移动到目标节点之前
        gm.graph.move_before(allreduce.node_list[wait_idx:], target_node)
# 定义一个图优化的处理函数，用于从图模块中移除不必要的 `copy_` 节点
@graph_optimization_pass(
    prerequisites=[],
    apply_after=[],
)
def remove_copy_from_optimizer(gm: IterGraphModule) -> None:
    """Erase the orphant copy_ that generated when tracing optimizer.

    Two reasons why we could not simply use the DCE of fx.Graph.
    1. fx.Graph treats copy_ as a side-effect node and does not erase it.
    2. Users may want to preserve some orphan `copy_` that is not from the
       optimizer.
    If the second reason does not hold, this pass can be rewritten as using
    DCE from fx.Graph (with the overwrite to the side-effect node list).
    """
    # 最大的 `copy_` 节点追溯深度
    MAX_COPY_DISTANCE = 5
    # 存储待移除的 `copy_` 节点集合
    remove_candidates: Set[fx.Node] = set()
    
    # 从图模块的节点列表中逆序遍历
    for node in reversed(gm.graph.nodes):
        # 如果节点有用户使用，跳过
        if node.users:
            continue
        # 如果节点操作不是函数调用或者目标不是 `aten.copy_.default`，跳过
        if node.op != OP.CALL_FUNCTION or node.target != aten.copy_.default:
            continue
        
        # 存储 `copy_` 节点的祖先节点
        copy_ancestors: Set[fx.Node] = set()
        # 使用双端队列存储节点和距离
        nodes = collections.deque([node, None])
        distance = 0
        should_remove = False
        
        # 进行节点及其祖先节点的追溯，直到达到最大深度或者满足移除条件
        while nodes and distance < MAX_COPY_DISTANCE:
            visiting = nodes.popleft()
            if visiting is None:
                distance += 1
                if nodes:
                    nodes.append(None)
                continue
            copy_ancestors.add(visiting)
            # 如果节点操作是函数调用且目标以指定字符串开头，则设置应移除标志为真
            if visiting.op == OP.CALL_FUNCTION and str(visiting.target).startswith(
                ("aten._foreach_", "aten._fused_")
            ):
                should_remove = True
            # 获取节点的所有父节点并加入队列
            parents = pytree.arg_tree_leaves(*visiting.args, **visiting.kwargs)
            for parent in parents:
                if isinstance(parent, fx.Node):
                    nodes.append(parent)
        
        # 如果应移除标志为真，则将所有祖先节点添加到待移除集合中
        if should_remove:
            remove_candidates.update(copy_ancestors)

    # 再次逆序遍历图模块的节点列表，移除待移除集合中的 `copy_` 节点
    for node in reversed(gm.graph.nodes):
        if node.users:
            continue
        if node not in remove_candidates:
            continue
        gm.graph.erase_node(node)


# 定义一个名为 `AdamArgs` 的命名元组，用于存储 `fused_adam` 函数的参数列表
AdamArgs = collections.namedtuple(
    "AdamArgs",
    ["params", "grads", "exp_avgs", "exp_avg_sqs", "max_exp_avg_sqs", "state_steps"],
)


# TODO(fegin): Have a template class for all Block class.
# 定义一个名为 `FusedAdamBlock` 的数据类，用于表示融合 Adam 算法的块
@dataclass(unsafe_hash=True)
class FusedAdamBlock:
    optim_node: fx.Node
    generate_output: bool
    # 存储参数节点的输出列表，顺序遵循参数顺序
    param_outputs: List[fx.Node] = field(default_factory=list)
    # 存储梯度节点的输出列表
    grad_outputs: List[fx.Node] = field(default_factory=list)
    # 存储 exp_avgs 节点的输出列表
    exp_avgs_outputs: List[fx.Node] = field(default_factory=list)
    # 存储 exp_avg_sqs 节点的输出列表
    exp_avg_sqs_outputs: List[fx.Node] = field(default_factory=list)
    # TODO(fegin): populate/generate the max_exp_avg_sqs if exists
    # 存储 max_exp_avg_sqs 节点的输出列表
    max_exp_avg_sqs: List[fx.Node] = field(default_factory=list)
    def generate_outputs(self):
        # 遍历所有参数并生成相应的输出列表。
        # 假设相应的输出节点尚未创建。
        def _generate_outputs(arg_idx, output_list):
            # 获取优化节点所属的图
            graph = self.optim_node.graph
            # 在优化节点后插入新节点
            with graph.inserting_after(self.optim_node):
                # 创建一个调用 operator.getitem 函数的节点，获取参数列表中的指定元素
                optim_getitem = graph.call_function(
                    operator.getitem, (self.optim_node, arg_idx)
                )
            # 遍历优化节点指定参数索引处的参数
            for i, arg in enumerate(self.optim_node.args[arg_idx]):
                # 在 optim_getitem 节点后插入新节点
                with graph.inserting_after(optim_getitem):
                    # 创建一个调用 operator.getitem 函数的节点，获取更新后的参数
                    updated_arg = graph.call_function(
                        operator.getitem, (optim_getitem, i)
                    )
                # 在 updated_arg 节点后插入新节点
                with graph.inserting_after(updated_arg):
                    # 创建一个调用 aten.copy_ 函数的节点，复制参数
                    output_copy = graph.call_function(aten.copy_, (arg, updated_arg))
                # 将复制后的输出节点添加到输出列表中
                output_list.append(output_copy)

        # 生成参数输出列表
        _generate_outputs(0, self.param_outputs)
        # 不生成梯度输出列表，因为它没有被使用
        _generate_outputs(2, self.exp_avgs_outputs)
        # 生成 exp_avg_sqs_outputs 输出列表
        _generate_outputs(3, self.exp_avg_sqs_outputs)

    def populate_outputs(self):
        # 从图中填充现有的输出列表。
        def _populate_outputs(args_idx, output_list):
            # 初始时，optim_getitem 节点为 self.optim_node
            optim_getitem = self.optim_node
            # 遍历使用 self.optim_node 的所有用户节点
            for user in self.optim_node.users:
                # 断言用户节点的目标为 operator.getitem
                assert (
                    user.target == operator.getitem
                ), f"The user of {self.optim_node} is not getitem."
                # 如果用户节点的第二个参数与 args_idx 相等，则更新 optim_getitem 节点
                if user.args[1] == args_idx:
                    optim_getitem = user
                    break
            # 断言 optim_getitem 节点不等于 self.optim_node，即找到了对应的 getitem 节点
            assert (
                optim_getitem != self.optim_node
            ), f"Cannot find the getitem node for {self.optim_node}"
            # 将 self.optim_node 添加到 output_list 中相应次数，作为输出列表的初始值
            output_list.extend(
                [self.optim_node] * len(cast(List[fx.Node], self.optim_node.args[0]))
            )
            # 遍历 optim_getitem 的用户节点
            for updated_arg in optim_getitem.users:
                # 断言用户节点的目标为 operator.getitem
                assert (
                    updated_arg.target == operator.getitem
                ), f"Unexpected node target {updated_arg.target}."
                # 获取更新参数的索引
                idx = updated_arg.args[1]
                # 获取 updated_arg 的第一个用户节点作为输出副本
                output_copy = next(iter(updated_arg.users))
                # 断言 output_copy 的目标以 "aten.copy_" 开头，即确保是复制节点
                assert str(output_copy.target).startswith(
                    "aten.copy_"
                ), f"Unexpected node target {output_copy.target}."
                # 将 output_copy 赋给输出列表中相应索引的位置
                output_list[idx] = output_copy
            # 遍历输出列表，确保没有任何元素等于 self.optim_node，即所有输出均已替换
            for i, output in enumerate(output_list):
                assert output != self.optim_node, f"{i}th output is not replaced."

            # 断言输出列表不为空
            assert output_list, f"The output for {self.optim_node} is empty."

        # 填充 param_outputs 输出列表
        _populate_outputs(0, self.param_outputs)
        # 填充 exp_avgs_outputs 输出列表
        _populate_outputs(2, self.exp_avgs_outputs)
        # 填充 exp_avg_sqs_outputs 输出列表
        _populate_outputs(3, self.exp_avg_sqs_outputs)

    def __post_init__(self):
        # 如果 param_outputs 不为空，则直接返回
        if self.param_outputs:
            return
        # 如果需要生成输出，则调用 generate_outputs 方法
        if self.generate_output:
            self.generate_outputs()
        # 否则，调用 populate_outputs 方法
        else:
            self.populate_outputs()
@dataclass(unsafe_hash=True)
class ForeachAddBlock:
    add_node: fx.Node
    generate_output: bool
    # The output list of the copy nodes. The order follows the argument order.
    outputs: List[fx.Node] = field(default_factory=list)

    def generate_outputs(self):
        # Iterate all the args and generate the corresponding output lists
        # Assuming the corresponding output nodes are not created yet.
        graph = self.add_node.graph
        for i, arg in enumerate(cast(Tuple[Any, ...], self.add_node.args[0])):
            with graph.inserting_after(self.add_node):
                updated_arg = graph.call_function(operator.getitem, (self.add_node, i))
            with graph.inserting_after(updated_arg):
                output_copy = graph.call_function(aten.copy_, (arg, updated_arg))
            self.outputs.append(output_copy)
        assert self.outputs, f"The output for {self.add_node} is empty."

    def populate_outputs(self):
        # Populate the existing output lists from the graph.
        self.outputs = [
            self.add_node for _ in cast(Tuple[Any, ...], self.add_node.args[0])
        ]
        for updated_arg in self.add_node.users:
            assert (
                updated_arg.target == operator.getitem
            ), f"Unexpected node target {updated_arg.target}"
            idx = cast(int, updated_arg.args[1])
            output_copy = next(iter(updated_arg.users))
            assert str(output_copy.target).startswith(
                "aten.copy_"
            ), f"The expected output node is different, {str(output_copy.target)}"
            self.outputs[idx] = output_copy
        for i, output in enumerate(self.outputs):
            assert output != self.add_node, f"{i}th output is not replaced."

    def __post_init__(self):
        if self.outputs:
            return

        if self.generate_output:
            self.generate_outputs()
        else:
            self.populate_outputs()


@dataclass(unsafe_hash=True)
class FusedOptimizerBlock:
    step: ForeachAddBlock
    optim: FusedAdamBlock


def get_fused_optimizer_block(optim_node: fx.Node) -> FusedOptimizerBlock:
    """Given a fused optimizer node and return the FusedOptimizerBlock."""
    MAX_STEP_DISTANCE = 5
    # Find the step (foreach_add)
    nodes = collections.deque([optim_node, None])
    step_node = optim_node
    distance = 0
    while nodes and distance < MAX_STEP_DISTANCE:
        node = nodes.popleft()
        if node is None:
            distance += 1
            if nodes:
                nodes.append(None)
            continue
        elif node.op == OP.CALL_FUNCTION and str(node.target).startswith(
            "aten._foreach_add"
        ):
            step_node = node
            break
        else:
            nodes.extend(
                a
                for a in pytree.arg_tree_leaves(*node.args, **node.kwargs)
                if isinstance(a, fx.Node)
            )

    # The function does not have a final return statement
    # 如果步骤节点和优化器节点相同，抛出运行时错误
    if step_node == optim_node:
        raise RuntimeError(
            "Cannot find step node (foreach_add) for the optimizer node "
            f"{optim_node} with {MAX_STEP_DISTANCE} BFS distance. "
            "The API design does not match the tracing graph."
        )

    # 创建一个 ForeachAddBlock 对象，使用步骤节点，生成输出设置为 False
    step = ForeachAddBlock(step_node, generate_output=False)
    # 创建一个 FusedAdamBlock 对象，使用优化器节点，生成输出设置为 False
    optim = FusedAdamBlock(optim_node, generate_output=False)
    # 返回一个 FusedOptimizerBlock 对象，该对象包含上述创建的 step 和 optim 对象
    return FusedOptimizerBlock(step, optim)
def get_all_fused_optimizer_blocks(
    gm: IterGraphModule, optim_ops: Union[Tuple[str, ...], str]
) -> List[FusedOptimizerBlock]:
    """Find all the FusedOptimizerBlock that the optimizer operators are in `optim_ops`."""
    # 返回所有包含在`optim_ops`中的优化器块列表
    return [
        get_fused_optimizer_block(node)
        for node in gm.graph.nodes
        if node.name.startswith(optim_ops)
    ]


def _split_fused_adam(
    gm: IterGraphModule,
    orig_optim_block: FusedOptimizerBlock,
    split_gradients: Set[fx.Node],
) -> Tuple[FusedOptimizerBlock, FusedOptimizerBlock]:
    """Split the `orig_optim_block` into two FusedOptimizerBlock.

    The first one will be the optimizer that optimize `split_gradients`. The second one is
    used to optimize the remaining gradients.
    An assert will be raised if one of the optimizer optimize zero gradients.
    """
    # 解析`orig_optim_block`，将其拆分为两个`FusedOptimizerBlock`
    orig_optim_args = AdamArgs(*orig_optim_block.optim.optim_node.args)
    optim_args = (AdamArgs([], [], [], [], [], []), AdamArgs([], [], [], [], [], []))
    # 以下唯一的提示用于拆分优化器是顺序/索引。
    orig_optim_indices: Tuple[List[int], List[int]] = ([], [])
    orig_step_indices: Tuple[List[int], List[int]] = ([], [])

    for idx, gradient in enumerate(orig_optim_args.grads):
        group_idx = 0 if gradient in split_gradients else 1
        orig_optim_indices[group_idx].append(idx)
        # 从`orig_optim_args`中获取第idx个梯度的参数
        for orig_arg, optim_arg in zip(orig_optim_args, optim_args[group_idx]):
            # 只有在原始参数列表不为空时才将参数添加到列表中。
            # 如果原始参数列表为空，则新列表也必须是一个空列表。
            if orig_arg:
                optim_arg.append(orig_arg[idx])

        # 如果步骤的参数顺序与优化器的参数顺序相同，则无需执行任何操作。
        # 但是，依赖这一假设是有风险的，因此我们填充`orig_step_indices`。
        orig_step_output = optim_args[group_idx].state_steps[-1]
        assert str(orig_step_output.target).startswith(
            "aten.copy_"
        ), f"The copy output is {orig_step_output.target}, expect aten.copy_"
        orig_step_getitem = orig_step_output.args[1]
        assert "getitem" in str(
            orig_step_getitem.target
        ), f"The copy getitem is {orig_step_getitem.target}, expect operator.getitem"
        orig_step_idx = orig_step_getitem.args[1]
        orig_step_indices[group_idx].append(orig_step_idx)

    if not all(l for l in (orig_step_indices + orig_optim_indices)):
        raise ValueError("At least one split optimizer does not have input.")

    output = get_output(gm.graph)
    results: List[FusedOptimizerBlock] = []
    flatten_output_args, spec = tree_flatten((output.args, output.kwargs))
    flatten_output_args_indices: DefaultDict[
        fx.Node, Set[int]
    ] = collections.defaultdict(set)
    # 对于 flatten_output_args 中的每个元素进行遍历，获取其索引和值
    for idx, output_arg in enumerate(flatten_output_args):
        # 如果 output_arg 是 fx.Node 类型的对象
        if isinstance(output_arg, fx.Node):
            # 将 output_arg 对应的索引 idx 添加到 flatten_output_args_indices 中
            flatten_output_args_indices[output_arg].add(idx)

    # 定义函数 replace_flatten_output_args，用于替换 flatten_output_args 中的原始节点 orig_node 为新节点 new_node
    def replace_flatten_output_args(orig_node: fx.Node, new_node: fx.Node):
        # 遍历 flatten_output_args_indices 中 orig_node 对应的索引集合
        for idx in flatten_output_args_indices[orig_node]:
            # 将 flatten_output_args 中的第 idx 个元素替换为 new_node
            flatten_output_args[idx] = new_node

    # 创建新的步骤节点和优化节点以及块。
    for group_idx in range(2):
        # 初始化步骤参数列表和原始步骤输出列表
        step_args: List[fx.Node] = []
        orig_step_outputs: List[fx.Node] = []

        # 在 orig_optim_block.optim.optim_node 之后插入新的步骤节点和块
        with gm.graph.inserting_after(orig_optim_block.optim.optim_node):
            # 遍历 orig_step_indices[group_idx] 中的索引
            for idx in orig_step_indices[group_idx]:
                # 将 orig_optim_block.step.add_node.args[0][idx] 添加到 step_args 中
                step_args.append(
                    cast(Tuple[fx.Node, ...], orig_optim_block.step.add_node.args[0])[idx]
                )
                # 将 orig_optim_block.step.outputs[idx] 添加到 orig_step_outputs 中
                orig_step_outputs.append(orig_optim_block.step.outputs[idx])

            # 创建一个新的步骤节点 step，调用 aten._foreach_add.Scalar 函数
            step = gm.graph.call_function(
                aten._foreach_add.Scalar,
                (step_args, 1),
            )
        
        # 创建步骤块 step_block，并生成其输出
        step_block = ForeachAddBlock(step, generate_output=True)

        # 遍历步骤块的输出列表 step_block.outputs
        for i, step_output in enumerate(step_block.outputs):
            # 替换图输出节点中的原始步骤输出 orig_step_outputs[i] 为新的步骤输出 step_output
            orig_step_output = orig_step_outputs[i]
            replace_flatten_output_args(orig_step_output, step_output)

            # 同时需要将用于新优化器的步骤输出也替换为新的步骤输出 step_output
            assert optim_args[group_idx].state_steps[i] == orig_step_output, (
                f"The expected step output node mismatched, {orig_step_output} "
                f"{optim_args[group_idx].state_steps[i]}"
            )
            optim_args[group_idx].state_steps[i] = step_output

        # 在第一个步骤输出后插入优化器节点，因为其拓扑排序顺序是最后的
        with gm.graph.inserting_after(step_block.outputs[0]):
            # 创建一个新的优化器节点 optim，调用 aten._fused_adam.default 函数
            optim = gm.graph.call_function(
                aten._fused_adam.default,
                optim_args[group_idx],
                orig_optim_block.optim.optim_node.kwargs,
            )
        
        # 创建优化器块 optim_block，并生成其输出
        optim_block = FusedAdamBlock(optim, generate_output=True)

        # 替换 orig_optim_block.optim 中的列表（如 param_outputs、exp_avgs_outputs、exp_avg_sqs_outputs）
        for curr_idx, orig_idx in enumerate(orig_optim_indices[group_idx]):
            list_names = ("param_outputs", "exp_avgs_outputs", "exp_avg_sqs_outputs")
            for name in list_names:
                orig_list = getattr(orig_optim_block.optim, name)
                curr_list = getattr(optim_block, name)
                replace_flatten_output_args(orig_list[orig_idx], curr_list[curr_idx])

        # 将步骤块 step_block 和优化器块 optim_block 添加到结果列表 results 中
        results.append(FusedOptimizerBlock(step_block, optim_block))

    # 优化器被用作 train_step 的输出，因此必须更新图的输出节点。
    # 将扁平化后的输出参数按照规范重新组装成输出参数和输出关键字参数
    output_args, output_kwargs = tree_unflatten(flatten_output_args, spec)
    # 将输出节点的参数设置为重新组装的参数
    gm.graph.node_set_args(output, output_args)
    # 将输出节点的关键字参数设置为重新组装的关键字参数
    gm.graph.node_set_kwargs(output, output_kwargs)
    # 移除原始的 copy_ 节点，因为它们不会被 DCE（死代码消除）优化
    for copy_output in itertools.chain(
        orig_optim_block.optim.param_outputs,
        orig_optim_block.optim.exp_avgs_outputs,
        orig_optim_block.optim.exp_avg_sqs_outputs,
    ):
        gm.graph.erase_node(copy_output)
    # 调用 DCE 一次，以去除旧的优化器。这样做可以稍后擦除步骤的 copy_ 节点。
    gm.graph.eliminate_dead_code()
    # 擦除步骤的输出的 copy_ 节点
    for copy_output in orig_optim_block.step.outputs:
        gm.graph.erase_node(copy_output)
    # 为了保持一致性而调用这个，实际上不是必需的
    gm.graph.eliminate_dead_code()

    # 返回处理后的结果的元组
    return results[0], results[1]
def split_fused_optimizer(
    gm: IterGraphModule,
    optim_block: FusedOptimizerBlock,
    split_gradients: Set[fx.Node],
) -> Tuple[FusedOptimizerBlock, FusedOptimizerBlock]:
    # 如果 split_gradients 为空集，抛出数值错误异常
    if not split_gradients:
        raise ValueError("The given split_gradients is empty.")
    # 如果优化器块的目标以 "aten._fused_adam" 开头，调用 _split_fused_adam 函数处理
    if str(optim_block.optim.optim_node.target).startswith("aten._fused_adam"):
        return _split_fused_adam(gm, optim_block, split_gradients)
    else:
        # 否则抛出未实现错误异常，只支持 fused_adam
        raise NotImplementedError("Only fused_adam is supported now")


# TODO(fegin): API 目前仅支持 fused adam。应扩展以支持 foreach。
@graph_optimization_pass(
    prerequisites=[remove_copy_from_optimizer],
    apply_after=[schedule_comm_wait],
)
def iter_move_grads_and_optimizers(
    gm: IterGraphModule,
    target_comm_node: str,
    target_dest_node: str,
) -> None:
    """提取通信块并拆分出新的优化器和步骤，然后将此子图移动到前向图中。"""
    # 获取所有通信块，类型为 "all_reduce"
    for comm_block in get_all_comm_blocks(gm, "all_reduce"):
        if comm_block.comm_node.name == target_comm_node:
            break
    else:
        # 如果找不到目标通信节点，抛出值错误异常
        raise ValueError(f"Cannot find {target_comm_node}")

    # 获取所有类型为 "_fused_adam" 的融合优化器块
    optim_blocks = get_all_fused_optimizer_blocks(gm, "_fused_adam")
    for optim_block in optim_blocks:
        # 获取优化器参数
        optim_args = AdamArgs(*optim_block.optim.optim_node.args)
        # 检查 comm_block 的输出是否在优化器参数的梯度中
        one_output = next(iter(comm_block.outputs))
        if one_output in optim_args.grads:
            break
    else:
        # 如果 target_comm_node 没有被任何融合优化器使用，抛出值错误异常
        raise ValueError(f"{target_comm_node} is not used by any fused optimizer.")

    # 拆分融合优化器，获取移动的优化器块
    move_optim, _ = split_fused_optimizer(gm, optim_block, comm_block.outputs)

    # 查找所有后代节点，包括 comm_block 的通信节点和 move_optim 的步骤添加节点
    move_nodes = find_all_descendants(
        gm, [comm_block.comm_node, move_optim.step.add_node]
    )

    # 查找停止节点，该节点名称与 target_dest_node 匹配
    stop_node = find_node(gm.graph, lambda n: n.name == target_dest_node)[0]

    # 将 move_nodes 移动到下一个迭代之前的位置
    gm.graph.move_to_next_iter_before(move_nodes, stop_node)


def find_all_descendants(
    gm: IterGraphModule,
    parent_nodes: List[fx.Node],
) -> List[fx.Node]:
    """在 FX 图变换期间识别要移动的节点列表。"""
    # 断言：必须提供至少一个父节点
    assert len(parent_nodes) > 0, "No parent nodes are given."

    # 获取输出节点
    output = get_output(gm.graph)
    dq_parent_nodes = collections.deque(parent_nodes)
    move_node_set = set()
    while dq_parent_nodes:
        node = dq_parent_nodes.popleft()
        move_node_set.add(node)
        # 将所有用户是 fx.Node 类型且不是输出的节点添加到双端队列中
        dq_parent_nodes += [
            u for u in node.users if isinstance(u, fx.Node) and u != output
        ]
    # 返回移动节点列表，包括在 move_node_set 中的节点
    move_nodes = [node for node in gm.graph.nodes if node in move_node_set]

    return move_nodes
```