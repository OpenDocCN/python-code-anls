# `.\pytorch\torch\fx\passes\infra\partitioner.py`

```
# 导入必要的模块和函数
# mypy: allow-untyped-defs
from torch.fx.passes.utils.fuser_utils import fuse_by_partitions
import collections
import itertools
import logging

from copy import copy
from typing import Dict, Iterable, List, Optional, Sequence, Set

from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, _get_qualified_name
from torch.fx.passes.operator_support import OperatorSupportBase

# 获取当前模块的日志记录器并设置日志级别为 WARNING
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# 定义一个分区类
class Partition:
    def __init__(self, id: Optional[int] = None, nodes: Optional[Iterable[Node]] = None):
        self.id = id
        self.nodes = {node: None for node in nodes} if nodes is not None else dict()

    def __repr__(self) -> str:
        return str(self.nodes)

    def add_node(self, node: Node):
        self.nodes.update({node: None})

    def remove_node(self, node: Node):
        del self.nodes[node]

    def size(self):
        return len(self.nodes)

# 定义一个依赖查看器类
class _DependencyViewer:
    def __init__(self, graph_module: GraphModule):
        self.upstreams = collections.defaultdict(set)
        self.downstreams = collections.defaultdict(set)

        # 遍历图中的节点，建立节点之间的上游和下游依赖关系
        for node in graph_module.graph.nodes:
            for input_node in node.all_input_nodes:
                # 添加输入节点和输入节点的上游依赖
                self.upstreams[node].add(input_node)
                self.upstreams[node].update(self.upstreams[input_node])

        for node in reversed(graph_module.graph.nodes):
            for output_node in node.users:
                # 添加输出节点和输出节点的下游依赖
                self.downstreams[node].add(output_node)
                self.downstreams[node].update(self.downstreams[output_node])

    def downstreams_of(self, node: Node) -> Set[Node]:
        return self.downstreams[node]

    def upstreams_of(self, node: Node) -> Set[Node]:
        return self.upstreams[node]

# 基于能力的分区器类
class CapabilityBasedPartitioner:

    def __init__(self,
                 graph_module: GraphModule,
                 operator_support: OperatorSupportBase,
                 allows_single_node_partition: bool = False,
                 non_compute_ops: Optional[Sequence[str]] = None,
                 allowed_single_node_partition_ops: Optional[Sequence[str]] = None,
                 ) -> None:
        self.graph_module = graph_module
        self.operator_support = operator_support
        self.allows_single_node_partition = allows_single_node_partition
        self.non_compute_ops = non_compute_ops if non_compute_ops is not None else []
        self.allowed_single_node_partition_ops = (
            allowed_single_node_partition_ops
            if allowed_single_node_partition_ops is not None
            else []
        )
        self.dependency_viewer = _DependencyViewer(graph_module)

    def __is_node_supported(self, node: Node) -> bool:
        return (
            self.operator_support.is_node_supported(dict(self.graph_module.named_modules()), node)
        )
    # 定义一个方法，用于融合给定的分区列表，返回融合后的图模块
    def fuse_partitions(self, partitions: List[Partition], prefix: str = "fused_") -> GraphModule:
        # 记录调试信息，表示正在进行分区融合操作
        logger.debug("Fusing partitions...")
        # 调用 fuse_by_partitions 函数，将分区转换为期望的格式 List[List[Node]]，然后进行融合
        return fuse_by_partitions(
            self.graph_module,
            [list(partition.nodes) for partition in partitions],
            prefix=prefix,
        )

    # 移除分区边界上的非计算操作。
    # 定义一个方法，用于移除分区中的非计算操作节点
    def remove_bookend_non_compute_ops(self, partitions: List[Partition]):
        # 将非计算操作的集合转换为集合类型，以便快速查找
        non_compute_ops = set(self.non_compute_ops)

        # 定义一个函数，用于判断节点是否为非计算节点
        def is_non_compute_node(node: Node):
            return node.op == "call_function" and \
                _get_qualified_name(node.target) in non_compute_ops  # type: ignore[arg-type]

        # 缓存透明输入节点和透明输出节点
        transparent_input_nodes: Dict[Node, bool] = {}
        transparent_output_nodes: Dict[Node, bool] = {}

        # 定义一个函数，用于判断节点是否为透明输入节点
        def is_transparent_input_node(node: Node, partition: Set[Node], removed_nodes: Set[Node]):
            # 如果节点是占位符或者不在当前分区中或者已被移除，则认为是透明输入节点
            if node.op == "placeholder" or (node not in partition) or (node in removed_nodes):
                return True
            # 如果节点已经在透明输入节点字典中，则直接返回缓存的结果
            if node in transparent_input_nodes:
                return transparent_input_nodes[node]
            # 如果节点是非计算节点，则遍历其所有输入节点，判断它们是否为透明输入节点
            if is_non_compute_node(node):
                for input_n in node.all_input_nodes:
                    if not is_transparent_input_node(input_n, partition, removed_nodes):
                        transparent_input_nodes[node] = False
                        return False
                transparent_input_nodes[node] = True
                return True
            transparent_input_nodes[node] = False
            return False

        # 定义一个函数，用于判断节点是否为透明输出节点
        def is_transparent_output_node(node: Node, partition: Set[Node], removed_nodes: Set[Node]):
            # 如果节点是占位符或者不在当前分区中或者已被移除，则认为是透明输出节点
            if node.op == "placeholder" or (node not in partition) or (node in removed_nodes):
                return True
            # 如果节点已经在透明输出节点字典中，则直接返回缓存的结果
            if node in transparent_output_nodes:
                return transparent_output_nodes[node]
            # 如果节点是非计算节点，则遍历其所有使用该节点的输出节点，判断它们是否为透明输出节点
            if is_non_compute_node(node):
                for output_n in node.users:
                    if not is_transparent_output_node(output_n, partition, removed_nodes):
                        transparent_output_nodes[node] = False
                        return False
                transparent_output_nodes[node] = True
                return True
            transparent_output_nodes[node] = False
            return False

        # 遍历每个分区
        for partition in partitions:
            # 注意这里使用集合来存储要移除的节点，因为只需要查询节点是否存在，不需要遍历集合中的节点
            remove_node: Set[Node] = set()
            # 遍历分区中的每个节点
            for node in partition.nodes:
                # 如果节点是非计算节点，并且同时满足以下条件之一：
                # 1. 被标记为透明输入节点
                # 2. 被标记为透明输出节点
                if is_non_compute_node(node) and \
                    (is_transparent_input_node(node, set(partition.nodes), remove_node) or
                     is_transparent_output_node(node, set(partition.nodes), remove_node)):
                    # 将该节点添加到要移除的集合中
                    remove_node.add(node)

            # 如果存在要移除的节点，则从分区中移除这些节点
            if len(remove_node) != 0:
                for node in remove_node:
                    partition.nodes.pop(node, None)

    # 定义一个方法，用于对图模块进行分区和融合操作
    def partition_and_fuse(self, prefix: str = "fused_") -> GraphModule:
        # 提出建议的分区方案
        partitions = self.propose_partitions()
        # 根据分区方案对图模块进行融合操作
        fused_gm = self.fuse_partitions(partitions, prefix=prefix)
        # 返回融合后的图模块
        return fused_gm
```