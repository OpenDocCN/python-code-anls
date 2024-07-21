# `.\pytorch\torch\fx\experimental\partitioner_utils.py`

```py
# mypy: allow-untyped-defs
# 引入必要的模块和类
from enum import Enum
from typing import NamedTuple, Dict, List, Set

# 导入torch.fx.node模块中的Node类和map_arg函数
from torch.fx.node import Node, map_arg


class Partition:
    """Partition class contains all the information about an individual partition.
    It also provides necessary methods for manipulation the partition.
    """

    def __init__(self, partition_id: int) -> None:
        # 初始化分区对象的属性
        self.nodes: Set[Node] = set()  # 存储分区内的节点集合
        self.partition_id = partition_id  # 分区的唯一标识符
        self.parents: Set[Partition] = set()  # 存储分区的父分区集合
        self.children: Set[Partition] = set()  # 存储分区的子分区集合
        self.bfs_level: int = -1  # 分区在广度优先搜索中的层级，默认为-1
        self.used_mem_bytes: int = 0  # 分区已使用的内存字节数，默认为0
        self.logical_device_ids: List[int] = []  # 分区逻辑设备的ID列表，初始化为空列表

    def __str__(self):
        return str(self.partition_id)  # 返回分区的字符串表示形式，即分区ID

    def recalculate_mem_size(self):
        self.used_mem_bytes = 0  # 重新计算分区已使用的内存字节数
        for node in self.nodes:
            self.used_mem_bytes += get_extra_size_of(node, self.nodes)  # 调用函数计算节点额外内存消耗

    def add_node(self, node):
        input_nodes: Dict[Node, None] = {}
        map_arg(node.args, input_nodes.setdefault)  # 映射节点参数到输入节点字典中
        map_arg(node.kwargs, input_nodes.setdefault)  # 映射节点关键字参数到输入节点字典中
        # 如果当前节点的输入节点是占位符或常量，则添加到分区节点集合中
        for n in input_nodes:
            if n.op in {"placeholder", "get_attr"}:
                self.nodes.add(n)
        self.nodes.add(node)  # 将当前节点添加到分区节点集合中
        self.recalculate_mem_size()  # 重新计算分区的内存消耗

    def remove_node(self, node):
        # 只有当节点在分区中时才移除该节点
        if node in self.nodes:
            self.nodes.remove(node)  # 从分区节点集合中移除节点
            # 收集节点的输入节点
            input_nodes: Dict[Node, None] = {}
            map_arg(node.args, input_nodes.setdefault)  # 映射节点参数到输入节点字典中
            map_arg(node.kwargs, input_nodes.setdefault)  # 映射节点关键字参数到输入节点字典中
            # 检查输入节点是否是占位符或get_attr，并且这些输入节点没有被分区中的其他节点使用时，则移除该输入节点
            for input_node in input_nodes:
                if all(
                    n not in self.nodes for n in input_node.users
                ) and input_node.op in {"placeholder", "get_attr"}:
                    self.nodes.remove(input_node)
            self.recalculate_mem_size()  # 重新计算分区的内存消耗


class Device(NamedTuple):
    name: str
    available_mem_bytes: int
    logical_id: int


class NodeLatency(NamedTuple):
    # 由于内存带宽引起的延迟
    mem_latency_sec: float
    # 由于计算引起的延迟
    computer_latency_sec: float


class PartitionLatency(NamedTuple):
    # 关键路径上所有节点的内存延迟之和
    mem_latency_sec: float
    # 关键路径上所有节点的计算延迟之和
    computer_latency_sec: float
    # 关键路径的总延迟
    overall_latency_sec: float


class PartitionMode(Enum):
    size_based = 0
    sparse_nn = 1
    cost_aware = 2
    kl_based = 3
    aot_based = 4


class PartitionerConfig(NamedTuple):
    devices: List[Device]
    mode: PartitionMode = PartitionMode.size_based
    # 定义一个浮点型变量，表示传输速率，初始值为 0.0
    transfer_rate_bytes_per_sec: float = 0.0
    
    # 定义一个字典，将节点映射到节点延迟的数据结构，初始为空字典
    node_to_latency_mapping: Dict[Node, NodeLatency] = {}
    
    # 定义一个字典，将节点映射到分区编号的数据结构，初始为空字典
    node_to_partition_mapping: Dict[Node, int] = {}
    
    # 定义一个字典，将分区编号映射到逻辑设备列表的数据结构，初始为空字典
    partition_to_logical_device_mapping: Dict[int, List[int]] = {}
    
    # 布尔型变量，表示是否通过将分区复制到其余空闲设备来使主机饱和，初始为 False
    saturate_host: bool = False
# 计算一个节点及其相关节点集合中额外所需的大小
def get_extra_size_of(node: Node, nodes: Set[Node]) -> int:
    """Given a node and a set of nodes,
    this function return the extra size that needed
    if this node is included in this set.
    """
    # 创建一个字典，用于存储所有输入节点
    input_nodes: Dict[Node, None] = {}
    # 将节点的位置参数和关键字参数映射到输入节点字典中
    map_arg(node.args, input_nodes.setdefault)
    map_arg(node.kwargs, input_nodes.setdefault)
    
    # 计算相关节点的总大小
    total_size_of_input_nodes = 0
    for n in input_nodes:
        # 确保这个节点尚未包含在给定的节点集合中
        if n not in nodes:
            # 获取节点的大小（以字节为单位）
            size_bytes = getattr(n, "size_bytes", None)
            if size_bytes:
                total_size_of_input_nodes += size_bytes.output_size
            else:
                raise RuntimeError("node has no size_bytes attr")
    
    # 加上当前节点本身的大小
    size_bytes = getattr(node, "size_bytes", None)
    if size_bytes:
        total_size_of_input_nodes += size_bytes.total_size
    else:
        raise RuntimeError("node has no size_bytes attr")
    
    return total_size_of_input_nodes


# 计算一个分区的延迟，根据节点到延迟的映射
def get_latency_of_one_partition(
    partition: Partition, node_to_latency_mapping: Dict[Node, NodeLatency]
) -> PartitionLatency:
    """Given a partition and its nodes' latency, return a PartitionLatency for this partition"""

    # 返回分区中顶层BFS级别的节点列表
    def get_top_nodes(partition: Partition) -> List[Node]:
        """Given a partition, return a list of nodes on the top bfs level"""
        top_nodes: List[Node] = []
        for node in partition.nodes:
            # 跳过占位符和get_attr节点
            if node.op in {"placeholder", "get_attr"}:
                continue
            input_nodes: Dict[Node, None] = {}
            map_arg(node.args, input_nodes.setdefault)
            map_arg(node.kwargs, input_nodes.setdefault)
            
            # 如果一个节点在这个分区中没有输入节点，
            # 或者其输入节点在这个分区中是占位符和get_attr节点，
            # 则该节点位于分区的顶层BFS级别
            if not any(
                n in partition.nodes and n.op not in {"placeholder", "get_attr"}
                for n in input_nodes
            ):
                top_nodes.append(node)
        return top_nodes
    def dfs_helper(node: Node, partition_latency) -> PartitionLatency:
        """Given a top node of a partition, this function returns
        the latency of the critical path in the partition
        """
        # 获取节点的延迟信息
        node_latency = node_to_latency_mapping[node]
        # 计算当前分区的整体延迟
        overall_latency_sec = partition_latency.overall_latency_sec + max(
            node_latency.computer_latency_sec, node_latency.mem_latency_sec
        )
        # 更新此路径的内存延迟
        mem_latency_sec = (
            partition_latency.mem_latency_sec + node_latency.mem_latency_sec
        )
        # 更新此路径的计算延迟
        computer_latency_sec = (
            partition_latency.computer_latency_sec + node_latency.computer_latency_sec
        )
        # 获取此节点在当前分区中的所有用户
        users = set(node.users).intersection(partition.nodes)
        if users:
            max_latency = PartitionLatency(
                mem_latency_sec=0.0, computer_latency_sec=0.0, overall_latency_sec=0.0
            )
            # 遍历所有用户节点，递归获取新的分区延迟
            for n in users:
                new_partition_latency = dfs_helper(
                    n,
                    PartitionLatency(
                        mem_latency_sec, computer_latency_sec, overall_latency_sec
                    ),
                )
                # 更新最大延迟
                if (
                    new_partition_latency.overall_latency_sec
                    > max_latency.overall_latency_sec
                ):
                    max_latency = new_partition_latency
            return max_latency
        # 如果没有用户，说明节点位于分区的底部
        return PartitionLatency(
            mem_latency_sec, computer_latency_sec, overall_latency_sec
        )

    # 主程序部分开始
    # 获取此分区的所有顶级节点
    top_nodes = get_top_nodes(partition)
    # 初始化关键路径延迟为初始值
    critical_path_latency = PartitionLatency(
        mem_latency_sec=0.0, computer_latency_sec=0.0, overall_latency_sec=0.0
    )
    # 遍历所有顶级节点，并找出最大延迟（关键路径延迟）
    for node in top_nodes:
        partition_latency = dfs_helper(
            node,
            PartitionLatency(
                mem_latency_sec=0.0, computer_latency_sec=0.0, overall_latency_sec=0.0
            ),
        )
        # 更新关键路径延迟为找到的最大延迟
        if (
            partition_latency.overall_latency_sec
            > critical_path_latency.overall_latency_sec
        ):
            critical_path_latency = partition_latency
    # 返回关键路径延迟
    return critical_path_latency
# 给定所有分区和节点到延迟映射的字典，返回每个分区到其总体延迟的映射字典
def get_partition_to_latency_mapping(
    partitions: List[Partition], node_to_latency_mapping: Dict[Node, NodeLatency]
) -> Dict[Partition, PartitionLatency]:
    # 初始化一个空字典，用于存储分区到延迟的映射关系
    partition_to_latency_mapping: Dict[Partition, PartitionLatency] = {}

    # 遍历每个分区并计算其延迟
    for partition in partitions:
        # 调用函数计算单个分区的延迟
        partition_latency = get_latency_of_one_partition(
            partition, node_to_latency_mapping
        )
        # 将分区及其计算得到的延迟添加到映射字典中
        partition_to_latency_mapping[partition] = partition_latency

    # 返回分区到延迟的映射字典
    return partition_to_latency_mapping


# 给定两个分区（父分区和子分区），计算它们之间的通信延迟
def get_comm_latency_between(
    parent_partition: Partition,
    child_partition: Partition,
    transfer_rate_bytes_per_sec: float,
):
    # 如果两个分区在同一个设备上，则通信延迟为 0
    if (
        parent_partition.logical_device_ids != []
        and child_partition.logical_device_ids != []
        and parent_partition.logical_device_ids == child_partition.logical_device_ids
    ):
        return 0.0

    # 初始化通信大小为 0
    comm_size = 0
    # 初始化已访问的节点集合
    visited_nodes = set()

    # 遍历子分区中的所有节点
    # 如果一个节点来自父分区的输入节点，则将其输出大小加入到通信大小中
    for node in child_partition.nodes:
        # 初始化一个空字典，用于存储节点的输入节点
        input_nodes: Dict[Node, None] = {}
        # 将节点的输入节点映射到 input_nodes 中
        map_arg(node.args, input_nodes.setdefault)
        map_arg(node.kwargs, input_nodes.setdefault)
        for n in input_nodes:
            # 如果输入节点在父分区中且未访问过，则处理其输出大小
            if n in parent_partition.nodes and n not in visited_nodes:
                # 获取节点的输出大小
                size_bytes = getattr(n, "size_bytes", None)
                if size_bytes is not None:
                    comm_size += size_bytes.output_size
                # 将节点标记为已访问
                visited_nodes.add(n)

    # 计算并返回通信延迟
    return comm_size / transfer_rate_bytes_per_sec


# 给定一个分区化图中的所有分区和分区到延迟映射的字典，找出所有分区中的关键路径，
# 并将其延迟作为整个图的延迟返回
def get_latency_of_partitioned_graph(
    partitions: List[Partition],
    partition_to_latency_mapping: Dict[Partition, PartitionLatency],
    transfer_rate_bytes_per_sec: float,
):
    """Given all partitions in a graph, find the critical path among all partitions
    and return its latency as the latency of the whole graph
    """
    def dfs_helper(partition: Partition, latency_so_far_sec: float) -> float:
        """递归辅助函数，计算分区路径的延迟"""
        # 加上当前分区的延迟，更新累计延迟
        latency_so_far_sec += partition_to_latency_mapping[
            partition
        ].overall_latency_sec
        children = partition.children
        if partition.children:
            max_latency_sec = 0.0
            for child in partition.children:
                # 计算当前分区与子分区之间的通信延迟
                comm_latency_sec = get_comm_latency_between(
                    partition, child, transfer_rate_bytes_per_sec
                )
                # 递归调用，计算子分区的延迟
                new_latency_sec = dfs_helper(
                    child, latency_so_far_sec + comm_latency_sec
                )
                if new_latency_sec > max_latency_sec:
                    max_latency_sec = new_latency_sec
            return max_latency_sec
        return latency_so_far_sec

    def get_top_partitions(partitions: List[Partition]) -> List[Partition]:
        """返回所有没有父分区的顶级分区"""
        top_partitions = []
        for partition in partitions:
            # 如果一个分区没有父分区，则它是顶级分区
            if len(partition.parents) == 0:
                top_partitions.append(partition)
        return top_partitions

    top_partitions = get_top_partitions(partitions)
    critical_path_latency_sec = 0.0
    for partition in top_partitions:
        # 计算每个顶级分区的最大路径延迟
        latency_sec = dfs_helper(partition, 0.0)
        if latency_sec > critical_path_latency_sec:
            critical_path_latency_sec = latency_sec
    return critical_path_latency_sec
```