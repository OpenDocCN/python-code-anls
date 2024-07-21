# `.\pytorch\torch\distributed\_spmd\graph_utils.py`

```py
# 导入日志模块
import logging
# 导入操作系统相关功能模块
import os
# 导入临时文件相关功能模块
import tempfile
# 导入枚举类型支持模块
from enum import Enum
# 导入类型提示相关功能模块
from typing import Callable, cast, Dict, Iterable, List, Set

# 导入 PyTorch 的 FX 模块
import torch.fx as fx
# 导入 PyTorch FX 模块的形状属性传播相关功能
from torch.fx.passes.shape_prop import TensorMetadata
# 导入 PyTorch FX 模块中的私有树操作模块
from torch.utils import _pytree as pytree
# 导入 PyTorch FX 模块中的树操作相关功能
from torch.utils._pytree import tree_flatten, tree_unflatten

# 创建日志记录器对象，命名为 "graph_utils"
logger: logging.Logger = logging.getLogger("graph_utils")

# 定义操作类型的枚举类
class OP(str, Enum):
    CALL_FUNCTION = "call_function"
    CALL_MODULE = "call_module"
    CALL_METHOD = "call_method"
    GET_ATTR = "get_attr"
    OUTPUT = "output"
    PLACEHOLDER = "placeholder"

# 定义通信类型的枚举类
class CommType(str, Enum):
    ALLREDUCE = "allreduce_"
    ALLGATHER = "allgather_"
    BROADCAST = "broadcast_"
    REDUCESCATTER = "reduce_scatter_"
    SCATTER = "scatter_"

# 获取节点的张量元数据信息
def get_node_tensor_metadata(node: fx.Node, is_required: bool = True) -> TensorMetadata:
    # 获取节点的元数据中的张量元数据，若不存在且需要时抛出运行时异常
    metadata = node.meta.get("tensor_meta", None)
    if is_required and metadata is None:
        raise RuntimeError(
            f"Callsite expects that ``tensor_meta`` exists in ``{node.name}``, "
            f"but got None instead. Node: {node.op} {node.name} {node.target}"
        )
    return metadata

# 获取图的输出节点
def get_output(graph: fx.Graph) -> fx.Node:
    """Take a graphmodule and return the graph output node.

    We traverse in reverse to expedite it, with the idea that last node should be output
    """
    # 反向遍历图中的节点，返回第一个操作为 OP.OUTPUT 的节点
    for node in reversed(graph.nodes):
        if node.op == OP.OUTPUT:
            return node
    # 若未找到输出节点，则抛出运行时异常
    raise RuntimeError(f"Cannot find the output node in {graph}")

# 查找满足条件的节点
def find_node(
    graph: fx.Graph, predicate: Callable, reverse_order: bool = False
) -> List[fx.Node]:
    """Take a predicate and return all the nodes in the `graph` where the predicate holds."""
    # 获得图中所有节点的可迭代对象
    nodes = cast(Iterable[fx.Node], graph.nodes)
    # 若需逆序，则反转节点的迭代器
    if reverse_order:
        nodes = cast(Iterable[fx.Node], iter(reversed(nodes)))  # type: ignore[call-overload]
    # 返回满足条件的节点列表
    return [node for node in nodes if predicate(node)]

# 判断子图是否为叶子子图
def is_leaf_subgraph(graph: fx.Graph, subgraph: List[fx.Node]) -> bool:
    """Ensure nodes in ``subgraph`` satisfy one of the following rules.

    1. The user of the node is in ``subgraph``.
    2. The user of the node is output.
    3. There are no users -- the node is a side-effect node.
    """
    # 创建包含所有子图节点的集合
    all_nodes: Set[fx.Node] = set(subgraph)
    # 获取图的输出节点
    output = get_output(graph)
    # 遍历子图中的每个节点
    for node in subgraph:
        # 遍历节点的用户
        for user in node.users:
            # 若用户不是节点或者用户不在子图中且用户不是输出节点，则返回 False
            if not isinstance(user, fx.Node):
                continue
            if user not in all_nodes and user != output:
                return False
    # 若所有节点满足条件，则返回 True
    return True

# 克隆子图并插入到目标节点之前
def clone_subgraph(
    graph: fx.Graph, subgraph: List[fx.Node], target: fx.Node
) -> List[fx.Node]:
    """Clone the given subgraph and insert it before ``target``.

    This API currently does not support inserting after ``target``.
    """
    # 创建包含所有子图节点的集合
    all_nodes = set(subgraph)
    # 创建节点映射字典
    mapping: Dict[fx.Node, fx.Node] = dict()
    # 创建克隆后的子图列表
    cloned_subgraph = []
    with graph.inserting_before(target):
        # 在目标节点之前插入子图中的节点
        for node in subgraph:
            # 遍历子图中的每个节点
            cloned_node = graph.call_function(
                node.target, node.args, node.kwargs, node.type
            )
            # 调用图对象的函数，克隆当前节点
            # TODO: IterGraph 中有许多 flatten/unflatten 操作可以使用 tree_map 简化。
            # 将在后续的 PR 中简化此处代码。
            # TODO 注释：在 IterGraph 中有很多可以使用 tree_map 简化的 flatten/unflatten 操作。将在后续的 PR 中简化此处代码。
            original_input = pytree.arg_tree_leaves(*node.args, **node.kwargs)
            # 获取当前节点的原始输入参数
            cloned_input, spec = tree_flatten((cloned_node.args, cloned_node.kwargs))
            # 对克隆节点的参数进行扁平化处理，并返回扁平化后的参数列表及其结构规范
            mapped_cloned_input = []
            for original_input_node, cloned_input_node in zip(
                original_input, cloned_input
            ):
                # 遍历原始输入参数和克隆后的参数，进行映射
                if (
                    isinstance(original_input_node, fx.Node)
                    and original_input_node in all_nodes
                ):
                    assert original_input_node in mapping
                    mapped_cloned_input.append(mapping[original_input_node])
                else:
                    mapped_cloned_input.append(cloned_input_node)
            # 根据映射后的参数和结构规范，进行反扁平化操作
            cloned_node.args, cloned_node.kwargs = tree_unflatten(
                mapped_cloned_input, spec
            )
            # 将当前节点与其克隆节点的映射关系存入 mapping 字典
            mapping[node] = cloned_node
            # 将克隆后的节点添加到克隆子图列表中
            cloned_subgraph.append(cloned_node)

    # 返回克隆后的子图
    return cloned_subgraph
# 重建图形模型，使其达到生产就绪状态
def rebuild_graph(gm: fx.GraphModule, remove_dead_code: bool = True) -> None:
    """Run the required steps to ensure production-ready graph.

    Note - per the fx docs, elimination of dead code is not very precise.
    Hence, the flag to make this step optional.
    """
    # 对图形模型进行静态分析和检查
    gm.graph.lint()
    # 如果指定要移除死代码，则执行死代码消除操作
    if remove_dead_code:
        gm.graph.eliminate_dead_code()
    # 重新编译图形模型
    gm.recompile()


# 将图形模型字典写入文件系统中
def dump_graphs_to_files(graphs: Dict[str, fx.GraphModule], folder: str = "") -> str:
    # 如果未提供文件夹路径，则创建临时文件夹
    if not folder:
        folder = tempfile.mkdtemp()

    # 遍历图形模型字典中的每个前缀和对应的图形模型对象
    for prefix, gm in graphs.items():
        # 打开文件以写入图形模型对象的字符串表示，并写入文件
        with open(os.path.join(folder, f"{prefix}.graph"), "w") as fp:
            fp.write(str(gm))

    # 记录警告信息，指示图形模型已经被写入到指定文件夹
    logger.warning("Dump graphs to %s", folder)

    # 返回存储图形模型文件的文件夹路径
    return folder
```