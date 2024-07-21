# `.\pytorch\torch\fx\experimental\accelerator_partitioner.py`

```py
# mypy: allow-untyped-defs
# 导入必要的模块和类
import operator
from collections import deque
from typing import Dict, List, Set, NamedTuple, Tuple, Deque

import torch
from torch.fx.passes.graph_manipulation import get_size_of_all_nodes
from torch.fx.experimental.partitioner_utils import (
    Partition,
    Device,
    PartitionerConfig,
    get_partition_to_latency_mapping,
    get_latency_of_partitioned_graph,
    NodeLatency,
    get_extra_size_of,
    PartitionMode,
)
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, map_arg
from torch.fx.passes.split_module import split_module


class DAGNode:
    """DAGNode class maintains useful information for a partition (submodule),
    and its input submodules and output submodules.
    """

    def __init__(
        self,
        submodule_node: Node,
        input_nodes: List[Node],
        output_nodes: List[Node],
        logical_device_ids: List[int],
        size_bytes: int,
    ) -> None:
        # 初始化 DAG 节点对象
        self.submodule_node: Node = submodule_node
        self.input_nodes: List[Node] = input_nodes
        self.output_nodes: List[Node] = output_nodes
        self.logical_device_ids: List[int] = logical_device_ids
        self.size_bytes = size_bytes

    def __str__(self) -> str:
        return str(self.submodule_node)


class DAG:
    """DAG class contains all the DAG nodes"""

    def __init__(self) -> None:
        # 初始化 DAG 对象
        self.nodes: List[DAGNode] = []

    def create_node(
        self,
        submodule_node: Node,
        input_nodes: List[Node],
        output_nodes: List[Node],
        logical_devices: List[int],
        size_bytes: int,
    ) -> None:
        # 创建一个新的 DAG 节点并添加到 DAG 中
        node = DAGNode(
            submodule_node, input_nodes, output_nodes, logical_devices, size_bytes
        )
        self.nodes.append(node)


class PartitionResult(NamedTuple):
    """NameTuple used for returning DAG and a new fx module"""

    dag: DAG
    module_with_submodules: GraphModule


"""Followings are some helper functions for partition manipulation"""


def reset_partition_device(partitions):
    # 重置所有分区的逻辑设备列表
    for partition in partitions:
        partition.logical_device_ids = []


def combine_two_partitions(
    partition_0: Partition, partition_1: Partition, partitions: List[Partition]
) -> None:
    """Given a list of partitions and its two partitions,
    combine these two partitions into a new one appending to the partitions
    and remove the previous two partitions from the list of partitions
    """
    # 合并两个分区为一个新的分区对象，并从分区列表中移除旧的两个分区
    partition = Partition(len(partitions))
    partition.nodes = partition_0.nodes.union(partition_1.nodes)
    partition.recalculate_mem_size()
    partitions.append(partition)
    partitions.remove(partition_0)
    partitions.remove(partition_1)
    reorganize_partitions(partitions)
    return


def set_parents_and_children(partitions: List[Partition]) -> None:
    """Given a list of partitions, mark parents and children for each partition"""
    # 遍历所有分区中的节点。
    # 如果一个节点的使用者在其他分区中，
    # 遍历所有分区对象
    for partition in partitions:
        # 初始化当前分区的子节点集合和父节点集合为空集合
        partition.children = set()
        partition.parents = set()
    
    # 再次遍历所有分区对象
    for partition in partitions:
        # 遍历当前分区的所有节点
        for node in partition.nodes:
            # 获取当前节点的所有用户节点
            users = node.users
            # 遍历每个用户节点
            for n in users:
                # 查找用户节点属于哪个分区
                # 注意，如果用户节点本身也属于当前分区，则该分区不是当前分区的子分区
                for p in partitions:
                    # 如果找到了一个不是当前分区并且包含用户节点n但不包含当前节点node的分区p
                    if p != partition and n in p.nodes and node not in p.nodes:
                        # 将分区p添加到当前分区的子分区集合中，将当前分区添加到分区p的父分区集合中
                        partition.children.add(p)
                        p.parents.add(partition)
    
    # 函数执行完毕，返回到调用处
    return
# 根据给定的分区列表重新组织分区的 ID，将每个分区的 partition_id 设置为其在列表中的索引值
def reorganize_partitions(partitions: List[Partition]) -> None:
    # 重新分配分区的 partition_id
    for i, partition in enumerate(partitions):
        partition.partition_id = i
    # 设置每个分区的父分区和子分区关系
    set_parents_and_children(partitions)
    return


# 给定一个分区列表，为每个分区标记其在 BFS 中的层级
def get_bfs_level_partition(partitions: List[Partition]) -> None:
    # 当前层级的分区集合
    current_level: Set[Partition] = set()
    # 访问过的分区集合
    visited: Set[Partition] = set()
    
    # 遍历所有分区，将没有父分区的分区加入到根层级中
    for partition in partitions:
        if len(partition.parents) == 0:
            current_level.add(partition)
    
    # 下一个层级的分区集合
    next_level: Set[Partition] = set()
    level = 0
    
    # 广度优先搜索 (BFS)
    while current_level:
        partition = current_level.pop()
        # 设置分区的 BFS 层级
        partition.bfs_level = level
        visited.add(partition)
        children = partition.children
        # 将当前分区的子分区加入到下一个层级中
        for child in children:
            if child not in next_level:
                next_level.add(child)
        # 如果当前层级已经遍历完，则将下一个层级作为当前层级，层级加一
        if not current_level:
            current_level = next_level.copy()
            next_level = set()
            level += 1
    return


# 给定一个分区列表，返回节点到其所在分区的映射
def get_node_to_partition_mapping(partitions: List[Partition]) -> Dict[Node, int]:
    node_to_partition: Dict[Node, int] = {}
    # 遍历所有分区，为每个节点设置其所在的分区 ID
    for partition in partitions:
        for node in partition.nodes:
            node_to_partition[node] = partition.partition_id
    return node_to_partition


# 给定设备列表，返回逻辑设备 ID 到设备对象的映射
def get_logical_id_to_device(devices: List[Device]) -> Dict[int, Device]:
    logical_id_to_device: Dict[int, Device] = {}
    # 遍历所有设备，建立逻辑 ID 到设备对象的映射关系
    for d in devices:
        logical_id_to_device[d.logical_id] = d
    return logical_id_to_device


# 给定分区列表和设备列表，返回以下三个结果的元组：
# 1. 设备到其上的分区的映射；
# 2. 设备到其剩余内存大小的映射；
# 3. 没有分配到设备的分区列表。
def get_device_partition_stats(
    partitions: List[Partition], devices: List[Device]
) -> Tuple[Dict[Device, List[Partition]], Dict[Device, int], List[Partition]]:
    # 逻辑 ID 到设备对象的映射
    logical_id_to_device = get_logical_id_to_device(devices)
    
    # 设备到分区列表的映射
    device_to_partitions: Dict[Device, List[Partition]] = {}
    # 设备到剩余内存大小的映射
    device_to_left_mem_bytes: Dict[Device, int] = {}
    
    # 初始化设备到空列表和剩余内存大小
    for d in devices:
        device_to_partitions[d] = []
        device_to_left_mem_bytes[d] = d.available_mem_bytes
    
    # 未分配设备的分区列表
    no_device_partitions = []
    # 遍历 partitions 列表中的每一个分区对象
    for partition in partitions:
        # 检查当前分区的逻辑设备 ID 列表是否不为空
        if partition.logical_device_ids != []:
            # 遍历当前分区的每一个逻辑设备 ID
            for logical_id in partition.logical_device_ids:
                # 根据逻辑设备 ID 获取对应的实际设备对象
                device = logical_id_to_device[logical_id]
                # 将当前分区添加到设备到分区列表的映射中
                device_to_partitions[device].append(partition)
                # 更新设备剩余内存字节数，减去当前分区使用的内存字节数
                device_to_left_mem_bytes[device] -= partition.used_mem_bytes
        else:
            # 如果当前分区的逻辑设备 ID 列表为空，则将该分区添加到无设备分区列表中
            no_device_partitions.append(partition)

    # 返回三个结果：设备到分区列表的映射，设备到剩余内存字节数的映射，无设备分区列表
    return (
        device_to_partitions,
        device_to_left_mem_bytes,
        no_device_partitions,
    )
def get_device_to_partitions_mapping(
    partitions: List[Partition], devices: List[Device]
):
    """Given a list of partitions and a list of devices,
    map each partition into a device.
    """
    
    def calculate_extra_mem_bytes_needed_for(
        partition: Partition, partitions: List[Partition]
    ):
        """Calculate the extra memory bytes needed for placing
        a partition, considering all current partitions' nodes.
        """
        all_nodes: Set[Node] = set()
        # Collect all nodes from all partitions
        for p in partitions:
            all_nodes = all_nodes.union(p.nodes)
        if len(all_nodes) == 0:
            return partition.used_mem_bytes
        # Include nodes of the current partition
        all_nodes = all_nodes.union(partition.nodes)
        extra_size_needed = 0
        # Calculate extra size needed for each node in the partition
        for node in partition.nodes:
            extra_size_needed += get_extra_size_of(node, all_nodes)
        return extra_size_needed

    def find_device_for(partition: Partition):
        """Find a suitable device for placing a partition based
        on available memory and existing allocations.
        """
        for d in device_to_left_mem_bytes:
            # Calculate extra memory needed for placing the partition
            extra_size_needed = calculate_extra_mem_bytes_needed_for(
                partition, device_to_partitions[d]
            )
            # Check if the device has enough memory
            if extra_size_needed < device_to_left_mem_bytes[d]:
                device_to_partitions[d].append(partition)
                partition.logical_device_ids.append(d.logical_id)
                device_to_left_mem_bytes[d] -= extra_size_needed
                return True
        return False

    # Obtain initial partitioning statistics for devices
    (
        device_to_partitions,
        device_to_left_mem_bytes,
        no_device_partitions,
    ) = get_device_partition_stats(partitions, devices)

    # Find devices for partitions that currently have no assigned device
    found_device = True
    for partition in no_device_partitions:
        # Sort devices by remaining memory
        device_to_left_mem_bytes = dict(sorted(device_to_left_mem_bytes.items(), key=operator.itemgetter(1)))
        # Attempt to find a suitable device for the partition
        found_device = find_device_for(partition)
        if not found_device:
            break
    return found_device


def check_dependency(partition):
    """Check for circular dependencies starting from a given partition
    using breadth-first search (BFS).
    """
    visited: Set[Partition] = {partition}
    queue: Deque[Partition] = deque([partition])
    while queue:
        p = queue.popleft()
        for child in p.children:
            if child == partition:
                return True
            else:
                if child not in visited:
                    visited.add(child)
                    queue.append(child)
    return False


class Partitioner:
    """Class for partitioning a fx module into submodules (partitions)
    so they can be executed across different accelerators.
    """

    def partition_graph(self, fx_module, partition_config):
        """Partition the fx module according to the specified scheme
        in partition_config.
        """
        # Method implementation omitted for brevity
    # 定义一个 DAG 结构并返回一个带有子模块节点的新 fx 模块。
    class Partitioner:
        def __init__(self) -> None:
            # 初始化空的分区列表
            self.partitions: List[Partition] = []
            # 初始化空的节点到分区索引的映射字典
            self.node_to_partition: Dict[Node, int] = {}
            # 初始化空的设备列表
            self.devices: List[Device] = []

        def partition_graph(
            self,
            fx_module: GraphModule,
            torch_module: torch.nn.Module,
            partitioner_config: PartitionerConfig,
        ) -> None:
            # 对整个 fx 模块进行分区
            partition_0 = self.create_partition()
            # 遍历图中的每个节点
            for node in self.graph_module.graph.nodes:
                if node.op == "output":
                    # 跳过输出节点，但某些情况下可能会有输出节点之后还有节点。
                    continue
                partition_0.nodes.add(node)
            # 设置分区使用的内存大小为整个图的大小
            partition_0.used_mem_bytes = total_size_of_graph
            # 设置逻辑设备 ID 列表为指定的逻辑设备 ID
            partition_0.logical_device_ids = [logical_device_id]
            # 获取节点到分区的映射关系
            self.node_to_partition = get_node_to_partition_mapping(self.partitions)
            return

        def do_partition(self) -> GraphModule:
            # 返回一个包含子模块节点（分区）的新 fx 模块。
            module_with_submodules = split_module(
                self.graph_module,
                self.torch_module,
                lambda node: self.node_to_partition[node],
            )
            return module_with_submodules

        def dump_dag(self, module_with_submodules: GraphModule) -> DAG:
            # 返回 DAG 结构和带有子模块节点的新 fx 模块。
            dag = DAG()
            # 遍历模块中的每个节点
            for node in module_with_submodules.graph.nodes:
                if node.op == "output":
                    break
                # 跳过特定操作的节点
                if node.op in {"placeholder", "get_attr"}:
                    continue
                # 跳过特定目标操作的节点
                if node.target == operator.__getitem__:
                    continue
                # 初始化空的输入节点字典
                input_nodes: Dict[Node, None] = {}
                map_arg(node.args, input_nodes.setdefault)
                map_arg(node.kwargs, input_nodes.setdefault)
                # 当节点有两个或更多输出节点时，将其结果输出到 'getitem' 节点。
                # 这些 'getitem' 节点是此节点的输出节点。
                # 否则，输出节点就是此节点本身。
                if len(node.users) > 1:
                    output_nodes = list(node.users)
                else:
                    output_nodes = [node]
                # 从节点名称中解析分区 ID
                partition_id = int(node.name.rsplit("_", 1)[-1])
                # 获取分区中的逻辑设备 ID 列表
                device_ids = self.partitions[partition_id].logical_device_ids
                # 获取分区使用的内存大小
                size_bytes = self.partitions[partition_id].used_mem_bytes
                # 在 DAG 中创建节点
                dag.create_node(
                    node, list(input_nodes), output_nodes, device_ids, size_bytes
                )
            return dag
    # 创建一个分区并将其添加到 self.partitions 列表中
    def create_partition(self) -> Partition:
        """Create a partition and append it to self.partitions."""
        # 分区ID为当前 self.partitions 列表的长度
        partition_id = len(self.partitions)
        # 创建一个新的 Partition 对象
        partition = Partition(partition_id)
        # 将新创建的 partition 添加到 self.partitions 列表中
        self.partitions.append(partition)
        # 返回创建的 partition 对象
        return partition

    # 为单个节点创建一个分区
    def create_single_node_partition(self, node):
        """Create a partition for a single node"""
        # 创建一个分区，并添加节点到该分区中
        partition = self.create_partition()
        partition.add_node(node)
        # 函数没有返回值

    # 基于成本感知的分区算法
    def cost_aware_partition(
        self,
        transfer_rate_bytes_per_sec: float,
        node_to_latency_mapping: Dict[Node, NodeLatency],
    ):
        # 这个方法的具体实现在此省略

    # 基于 KL 散度的分区算法
    def kl_based_partition(
        self,
        transfer_rate_bytes_per_sec: float,
        node_to_latency_mapping: Dict[Node, NodeLatency],
    ):
        # 这个方法的具体实现在此省略

    # 基于 AOT 的分区算法
    def aot_based_partition(
        self, node_to_partition_mapping, partition_to_logical_device_mapping
    ):
        """This function helps to rebuild the partitions given the nodes and its
        corresponding partition id
        """
        # partition_id 到 Partition 对象的映射字典
        partition_id_to_partition_mapping: Dict[int, Partition] = {}
        # 将外部传入的 node_to_partition_mapping 赋值给 self.node_to_partition
        self.node_to_partition = node_to_partition_mapping
        # 遍历每个节点及其所属的分区
        for node in self.node_to_partition:
            partition_id = self.node_to_partition[node]
            # 如果请求的分区尚未创建，则创建该分区
            if partition_id not in partition_id_to_partition_mapping:
                partition = Partition(partition_id)
                # 将新创建的 partition 添加到 self.partitions 列表中
                self.partitions.append(partition)
                # 将 partition 添加到映射字典中
                partition_id_to_partition_mapping[partition_id] = partition
                # 设置 partition 的逻辑设备ID列表
                partition.logical_device_ids = partition_to_logical_device_mapping[
                    partition_id
                ]
            else:
                # 如果分区已经存在，则直接从映射字典中获取该分区
                partition = partition_id_to_partition_mapping[
                    self.node_to_partition[node]
                ]
            # 将当前节点添加到分区中
            partition.add_node(node)
```