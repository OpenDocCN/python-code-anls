# `.\pytorch\torch\fx\passes\utils\source_matcher_utils.py`

```py
# mypy: allow-untyped-defs
# 引入所需模块和类
from dataclasses import dataclass, field
from torch.fx.graph import Graph
from torch.fx.node import Node
from torch.fx._compatibility import compatibility
from typing import Dict, List, Any, Type, Optional, Callable
import logging
import os

# 模块公开的接口列表
__all__ = ['get_source_partitions', 'check_subgraphs_connected', 'SourcePartition']

# 设置日志记录级别和格式，根据环境变量 PYTORCH_MATCHER_LOGLEVEL 来调整
def _init_logger():
    logger = logging.getLogger(__name__)

    level = os.environ.get('PYTORCH_MATCHER_LOGLEVEL', 'WARNING').upper()
    logger.setLevel(level)
    
    # 设置日志输出到控制台，指定格式为“文件名 > 消息内容”
    console = logging.StreamHandler()
    formatter = logging.Formatter("%(filename)s > %(message)s")
    console.setFormatter(formatter)
    console.setLevel(level)
    
    # 将控制台处理器添加到日志记录器中
    logger.addHandler(console)
    logger.propagate = False
    return logger

# 初始化全局日志记录器
logger = _init_logger()

# 定义数据类 SourcePartition，用于表示源分区
@compatibility(is_backward_compatible=False)
@dataclass
class SourcePartition:
    # 分区中的节点列表
    nodes: List[Node]

    # 这些节点所分解出的源
    source: Any

    # 分区中作为输入的节点列表
    input_nodes: List[Node] = field(default_factory=list)

    # 分区中被外部节点使用的节点列表
    output_nodes: List[Node] = field(default_factory=list)

    # 正在使用的参数列表
    params: List[Node] = field(default_factory=list)

# 定义函数 get_source_partitions，用于从图中获取源分区
@compatibility(is_backward_compatible=False)
def get_source_partitions(
    graph: Graph,
    wanted_sources: List[Any],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Dict[Any, List[SourcePartition]]:
    """
    Args:
        graph: 要分区的图
        wanted_sources: 期望分解的节点的源列表，可以是函数或模块类型
        filter_fn: 可选参数，用于过滤节点的函数

    Returns:
        返回一个字典，将给定源映射到与给定源对应的 SourcePartition 列表
    """
    # 初始化模块字典
    modules: Dict[Type, Dict[str, List[Node]]] = {}

    # 遍历图中的每个节点
    for node in graph.nodes:
        # 获取节点的元数据中的 source_fn_stack
        if (source_fn_st := node.meta.get("source_fn_stack", None)) is None:
            continue

        # 获取源函数或模块类型
        source_fn = source_fn_st[-1]
        
        # 如果节点的源不在期望的源列表中，则跳过
        if source_fn[1] not in wanted_sources:
            continue

        # 将节点添加到对应源的分区列表中
        diff_modules = modules.setdefault(source_fn[1], {})
        partition = diff_modules.setdefault(source_fn[0], [])
        partition.append(node)
    def make_partition(nodes: List[Node], module_type: Type) -> SourcePartition:
        input_nodes = set()
        output_nodes = set()
        params = set()
        # 遍历节点列表
        for node in nodes:
            # 遍历节点的参数列表
            for arg in node.args:
                # 如果参数是节点类型并且不在当前节点列表中，则添加到输入节点集合中
                if isinstance(arg, Node) and arg not in nodes:
                    input_nodes.add(arg)

            # 如果节点操作为 "get_attr"，则将节点添加到参数集合中
            if node.op == "get_attr":
                params.add(node)

            # 遍历节点的用户（即使用当前节点作为参数的节点）
            for user in node.users.keys():
                # 如果用户节点不在当前节点列表中，则将当前节点添加到输出节点集合中
                if user not in nodes:
                    output_nodes.add(node)

        # 创建并返回 SourcePartition 对象，包括节点列表、模块类型、输入节点列表、输出节点列表和参数列表
        return SourcePartition(
            nodes,
            module_type,
            list(input_nodes),
            list(output_nodes),
            list(params),  # type: ignore[arg-type]
        )

    # 初始化返回结果字典
    ret: Dict[Type[Any], List[SourcePartition]] = {}

    # 如果有过滤函数 filter_fn
    if filter_fn:
        # 用于存储过滤后的模块字典
        filtered_modules = {}
        # 遍历模块字典 modules 中的每个模块类型 tp 及其对应的名称到分区的映射
        for tp, name_to_partition in modules.items():
            # 根据 filter_fn 过滤分区，只保留满足过滤条件的分区
            filtered_name_to_partition = {
                name: partition
                for name, partition in name_to_partition.items()
                if all(map(filter_fn, partition))
            }
            # 将过滤后的分区字典存入 filtered_modules 中
            filtered_modules[tp] = filtered_name_to_partition
        # 更新 modules 为过滤后的模块字典
        modules = filtered_modules

    # 遍历模块字典 modules，为每个模块类型创建其对应的分区列表并存入 ret 中
    for k, v in modules.items():
        ret[k] = [make_partition(partition, k) for partition in v.values()]

    # 返回结果字典 ret
    return ret
# 使用装饰器标记函数兼容性属性，指定该函数不向后兼容
@compatibility(is_backward_compatible=False)
# 定义函数，检查两个子图是否连接
def check_subgraphs_connected(subgraph1: SourcePartition, subgraph2: SourcePartition) -> bool:
    """
    给定两个子图 A 和 B（以节点列表的形式），检查是否存在节点连接到 B 中的至少一个节点，
    即存在一个节点在 B 中使用了 A 中的某个节点（而不是反过来）。
    """

    # 反向遍历 subgraph1 的节点列表
    for node in reversed(subgraph1.nodes):
        # 遍历当前节点的所有使用者（即使用当前节点作为输入的节点）
        for user in node.users.keys():
            # 如果某个使用者节点存在于 subgraph2 的节点列表中，则返回 True
            if user in subgraph2.nodes:
                return True
    # 若未找到连接，返回 False
    return False
```