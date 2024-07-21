# `.\pytorch\torch\fx\passes\utils\fuser_utils.py`

```
# mypy: allow-untyped-defs
# 导入必要的库和模块
import copy  # 导入copy模块，用于对象深复制
from queue import SimpleQueue  # 导入SimpleQueue类，用于实现队列数据结构
from typing import List, Dict, Tuple  # 导入类型提示相关模块

import torch.fx  # 导入torch.fx模块
from torch.fx.graph_module import GraphModule  # 导入GraphModule类
from torch.fx.graph import Graph  # 导入Graph类
from torch.fx.node import Node  # 导入Node类
from torch.fx.passes.tools_common import NodeList, NodeSet, legalize_graph  # 导入相关工具类和函数
from torch.fx.passes.utils import lift_subgraph_as_module  # 导入lift_subgraph_as_module函数
from torch.fx._compatibility import compatibility  # 导入compatibility装饰器

@compatibility(is_backward_compatible=False)
# 定义拓扑排序函数，返回节点列表
def topo_sort(nodes: NodeList) -> NodeList:
    # 根据拓扑顺序对节点进行排序
    indegree_map = dict.fromkeys(nodes, 0)  # 创建节点的入度映射，初始值为0
    candidates: SimpleQueue = SimpleQueue()  # 创建一个简单队列用于存放候选节点

    for node in nodes:
        for n in node.all_input_nodes:
            if n in indegree_map:
                indegree_map[node] += 1  # 增加节点的入度计数

        if indegree_map[node] == 0:
            candidates.put(node)  # 将入度为0的节点加入队列

    sorted_nodes: NodeList = list()  # 创建空列表存放排序后的节点
    while not candidates.empty():
        node = candidates.get()  # 从队列中取出节点
        sorted_nodes.append(node)  # 将节点加入排序列表

        for n in node.users:
            if n in indegree_map:
                indegree_map[n] -= 1  # 减少节点的入度计数
                if indegree_map[n] == 0:
                    candidates.put(n)  # 如果入度为0，则加入队列

    assert len(nodes) == len(sorted_nodes), "topological sorted nodes doesn't have same length as input nodes"
    # 断言排序后的节点数和输入节点数相等，用于验证排序结果的正确性

    return sorted_nodes  # 返回拓扑排序后的节点列表

@compatibility(is_backward_compatible=False)
# 验证分区是否在原始图中形成依赖循环
# 返回True表示分区有效，返回False表示分区无效
def validate_partition(partition: NodeList) -> bool:
    partition_set = set(partition)  # 将分区转换为集合类型

    outputs: NodeList = list()  # 创建空列表存放输出节点
    for node in partition_set:
        for user_node in node.users:
            if user_node not in partition_set:
                outputs.append(user_node)  # 将外部用户节点添加为输出节点

    # 对分区输出节点执行BFS
    # 如果到达分区内的节点，则表示存在循环依赖
    # 该函数接管`root_nodes`的所有权并可能对其进行修改。
    # 定义一个函数 bfs_find_cycle，用于在图中从给定的根节点集合 root_nodes 开始进行广度优先搜索，判断是否存在循环
    def bfs_find_cycle(root_nodes: NodeList) -> bool:
        # 用于存储已经访问过的节点的集合，用来排除已经检查过的节点及其子节点
        visited: NodeSet = set()

        # 初始时将 root_nodes 中的节点作为起点，通过它们的连接子图进行遍历。已经在 visited 中的节点不会再次添加到队列 queue 中。
        queue: NodeList = root_nodes
        while queue:
            # 从队列中取出当前节点 current 进行处理
            current = queue.pop()
            # 将当前节点标记为已访问
            visited.add(current)
            # 如果当前节点在 partition_set 中，表示从 partition 的输出节点开始，已经达到了另一个 partition 的节点，存在循环
            if current in partition_set:
                # 存在循环，返回 True
                return True
            # 遍历当前节点的所有用户节点 user_node
            for user_node in current.users:
                # 如果 user_node 已经访问过，则跳过
                if user_node in visited:
                    continue
                # 将 user_node 添加到队列中，继续遍历其子节点
                queue.append(user_node)
        # 如果从 root_nodes 开始的遍历没有发现循环，则返回 False
        return False

    # 使用所有输出节点作为根节点，进行图的遍历检查是否存在循环
    if bfs_find_cycle(outputs):
        # 如果存在循环，返回 False
        return False

    # 如果不存在循环，返回 True
    return True
# 声明一个装饰器函数，用于标记该函数的兼容性，指定其是否向后兼容
@compatibility(is_backward_compatible=False)
# 定义一个函数，将指定图模块中的节点融合成一个新的图模块
def fuse_as_graphmodule(gm: GraphModule,
                        nodes: NodeList,
                        module_name: str) -> Tuple[GraphModule, Tuple[Node, ...], Tuple[Node, ...]]:
    """
    Fuse nodes in graph_module into a GraphModule.

    Args:
        gm (GraphModule): 目标图模块

        nodes (List[Node]): 要融合的`gm`中的节点列表，这些节点必须按拓扑顺序排序

        module_name: 融合后的GraphModule的类名

    Returns:
        fused_gm (GraphModule): 融合后的图模块，其节点是`gm`中`nodes`的副本

        original_inputs (Tuple[Node, ...]): 原始`gm`中节点`nodes`的输入节点

        original_outputs (Tuple[Node, ...]): 原始`gm`中节点`nodes`的输出消费者节点
    """

    # 假设节点已按拓扑顺序排序

    # 验证每个节点是否符合预期
    for node in nodes:
        assert node.graph.owning_module is gm, f"{node} doesn't belong to passed in graph module {gm._get_name()}"
        assert not node._erased, f"{node} has been removed from owning graph"
        assert node in gm.graph.nodes, f"{node} is not found in graph module {gm._get_name()}"

    # 验证融合后的分区是否引入了图中的依赖循环
    assert validate_partition(nodes), "Invalid partition, found dependency cycles"

    # 创建一个新的子图
    subgraph = Graph()

    # node_to_placeholder: 将旧图中的节点映射到新图中的占位符节点的字典
    node_to_placeholder: Dict[Node, Node] = {}

    # node_map: 将旧图中的节点映射到新图中对应节点的字典
    node_map: Dict[Node, Node] = {}

    # 处理输入节点，通过graph.node_copy的arg_transform函数
    def remap_inputs(x):
        if x.op == "get_attr":
            # TODO: 是否真的需要将get_attr节点复制到图中？
            # 在这里执行相应操作
            pass

        if x in nodes:
            # x在子图中，返回复制后的节点
            # 由于按拓扑顺序复制图，该节点应已被复制
            return node_map[x]

        if x not in node_to_placeholder:
            # x不在子图中，为其创建一个新的占位符节点
            placeholder_node = subgraph.placeholder(x.name, type_expr=x.type)
            # 复制所有元数据字段，即使某些字段对占位符节点来说可能不相关
            placeholder_node.meta = copy.copy(x.meta)
            node_to_placeholder[x] = placeholder_node

        return node_to_placeholder[x]

    # 按拓扑顺序复制节点
    for node in nodes:
        new_node = subgraph.node_copy(node, remap_inputs)
        node_map[node] = new_node

    # 处理输出节点
    output_mapping: Dict[Node, Node] = {}  # 旧输出到新输出的映射字典

    for node in nodes:
        for user_node in node.users:
            if user_node not in nodes:
                # 外部用户节点，需要作为输出暴露出去
                output_mapping[node] = node_map[node]

    # outs包含新子图中的节点
    # 获取输出映射中的所有值，并转换为元组
    outs = tuple(output_mapping.values())

    # 处理 FX 输出节点的参数。如果只有一个输出，则输出节点的参数形式为 (output_single)；
    # 如果有多个输出，则输出节点的参数形式为 ((output_0, output_1, ...)).
    subgraph.output(outs[0] if len(outs) == 1 else outs)

    # 对子图进行静态检查以确保正确性
    subgraph.lint()

    # 声明变量 fused_gm 为 GraphModule 类型
    fused_gm: GraphModule

    # 将子图作为模块提升，并将返回的模块和额外的对象（在此处未使用）分配给 fused_gm
    fused_gm, _ = lift_subgraph_as_module(gm, subgraph, comp_name="", class_name=module_name)

    # 声明变量 original_inputs 为 Node 类型的元组，其中包含原始模块中的输入节点
    original_inputs: Tuple[Node, ...] = tuple(node_to_placeholder.keys())

    # 声明变量 original_outputs 为 Node 类型的元组，其中包含原始模块中的输出节点
    original_outputs: Tuple[Node, ...] = tuple(output_mapping.keys())

    # 返回融合后的模块 fused_gm、原始模块的输入节点 original_inputs 和输出节点 original_outputs
    return fused_gm, original_inputs, original_outputs
# 根据兼容性标记，在不向后兼容的情况下定义函数，用于将子图模块插入主图模块中
@compatibility(is_backward_compatible=False)
def insert_subgm(gm: GraphModule, sub_gm: GraphModule, orig_inputs: Tuple[Node, ...], orig_outputs: Tuple[Node, ...]):
    # 将子图模块添加到主图模块中，使用子图模块的类名作为模块名
    submodule_name = sub_gm.__class__.__name__
    gm.add_submodule(submodule_name, sub_gm)

    # 在主图模块的计算图中创建一个 call_module 节点
    module_node = gm.graph.call_module(
        submodule_name,
        args=orig_inputs,
        kwargs=None)

    if len(orig_outputs) == 1:
        # 如果原始输出只有一个节点，则替换所有使用该节点的地方为新创建的模块节点
        orig_outputs[0].replace_all_uses_with(module_node, propagate_meta=True)
    else:
        # 如果有多个原始输出节点，则逐个处理替换其使用
        for i, orig_output in enumerate(orig_outputs):
            # 使用 Proxy 记录 getitem 访问
            proxy_out = torch.fx.Proxy(module_node)[i].node  # type: ignore[index]
            orig_output.replace_all_uses_with(proxy_out, propagate_meta=True)

        # 在模块节点的元数据中添加值为原始输出元数据中值的元组
        module_node.meta["val"] = tuple(orig_output.meta.get("val", None) for orig_output in orig_outputs)
    # 返回更新后的主图模块
    return gm

# 根据兼容性标记，在不向后兼容的情况下定义函数，用于从计算图模块中擦除指定的节点
@compatibility(is_backward_compatible=False)
def erase_nodes(gm: GraphModule, nodes: NodeList):
    # 按照反向拓扑顺序擦除原始节点
    for node in reversed(nodes):
        gm.graph.erase_node(node)

# 根据兼容性标记，在不向后兼容的情况下定义函数，用于将一组节点列表根据拓扑顺序融合成子图模块，并插入到主图模块中
@compatibility(is_backward_compatible=False)
def fuse_by_partitions(gm: GraphModule, partitions: List[NodeList], prefix: str = "fused_") -> GraphModule:
    for partition_id, nodes in enumerate(partitions):
        # 对每个分区中的节点进行拓扑排序
        sorted_nodes = topo_sort(nodes)

        # 构建子图模块，并获取原始输入和输出节点
        submodule_name = prefix + str(partition_id)
        sub_gm, orig_inputs, orig_outputs = fuse_as_graphmodule(gm, sorted_nodes, submodule_name)

        # 将子图模块插入主图模块中
        insert_subgm(gm, sub_gm, orig_inputs, orig_outputs)

        # 擦除已经融合的节点
        erase_nodes(gm, sorted_nodes)

    # 对包含新创建的子图模块的主图模块进行拓扑排序
    legalize_graph(gm)

    # 返回更新后的主图模块
    return gm
```