# `.\graphrag\graphrag\index\verbs\graph\merge\merge_graphs.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing merge_graphs, merge_nodes, merge_edges, merge_attributes, apply_merge_operation and _get_detailed_attribute_merge_operation methods definitions."""

# 引入必要的库和模块
from typing import Any, cast

import networkx as nx  # 导入网络图库
import pandas as pd  # 导入Pandas数据处理库
from datashaper import TableContainer, VerbCallbacks, VerbInput, progress_iterable, verb  # 导入数据处理相关的类和函数

from graphrag.index.utils import load_graph  # 从本地导入load_graph函数

from .defaults import (
    DEFAULT_CONCAT_SEPARATOR,  # 导入默认连接符常量
    DEFAULT_EDGE_OPERATIONS,  # 导入默认边操作字典常量
    DEFAULT_NODE_OPERATIONS,  # 导入默认节点操作字典常量
)
from .typing import (
    BasicMergeOperation,  # 导入基本合并操作类型
    DetailedAttributeMergeOperation,  # 导入详细属性合并操作类型
    NumericOperation,  # 导入数值操作类型
    StringOperation,  # 导入字符串操作类型
)


@verb(name="merge_graphs")  # 使用verb装饰器声明merge_graphs函数作为可执行操作
def merge_graphs(
    input: VerbInput,  # 输入参数，用于处理输入数据的接口
    callbacks: VerbCallbacks,  # 回调接口，用于执行合并操作时的回调功能
    column: str,  # 指定包含图形的列名
    to: str,  # 指定输出合并图形的列名
    nodes: dict[str, Any] = DEFAULT_NODE_OPERATIONS,  # 节点操作字典，默认使用DEFAULT_NODE_OPERATIONS
    edges: dict[str, Any] = DEFAULT_EDGE_OPERATIONS,  # 边操作字典，默认使用DEFAULT_EDGE_OPERATIONS
    **_kwargs,  # 其他未明确指定的参数，以关键字方式接收
) -> TableContainer:
    """
    Merge multiple graphs together. The graphs are expected to be in graphml format. The verb outputs a new column containing the merged graph.

    > Note: This will merge all rows into a single graph.

    ## Usage
    ```yaml
    verb: merge_graph
    args:
        column: clustered_graph # The name of the column containing the graph, should be a graphml graph
        to: merged_graph # The name of the column to output the merged graph to
        nodes: <node operations> # See node operations section below
        edges: <edge operations> # See edge operations section below
    ```py

    ## Node Operations
    The merge graph verb can perform operations on the nodes of the graph.

    ### Usage
    ```yaml
    nodes:
        <attribute name>: <operation>
        ... for each attribute or use the special value "*" for all attributes
    ```py

    ## Edge Operations
    The merge graph verb can perform operations on the nodes of the graph.

    ### Usage
    ```yaml
    edges:
        <attribute name>: <operation>
        ... for each attribute or use the special value "*" for all attributes
    ```py

    ## Operations
    The merge graph verb can perform operations on the nodes and edges of the graph. The following operations are available:

    - __replace__: This operation replaces the attribute with the last value seen.
    - __skip__: This operation skips the attribute, and just uses the first value seen.
    - __concat__: This operation concatenates the attribute with the last value seen.
    - __sum__: This operation sums the attribute with the last value seen.
    - __max__: This operation takes the max of the attribute with the last value seen.
    max
    - __min__: This operation takes the min of the attribute with the last value seen.
    - __average__: This operation takes the mean of the attribute with the last value seen.
    - __multiply__: This operation multiplies the attribute with the last value seen.
    """
    input_df = input.get_input()  # 获取输入数据作为Pandas DataFrame
    # 创建一个空的 Pandas DataFrame 对象
    output = pd.DataFrame()

    # 创建节点操作的字典，将每个节点属性及其详细合并操作存储起来
    node_ops = {
        attrib: _get_detailed_attribute_merge_operation(value)
        for attrib, value in nodes.items()
    }
    
    # 创建边操作的字典，将每条边属性及其详细合并操作存储起来
    edge_ops = {
        attrib: _get_detailed_attribute_merge_operation(value)
        for attrib, value in edges.items()
    }

    # 创建一个空的 NetworkX 图对象
    mega_graph = nx.Graph()
    
    # 获取输入数据框中待处理列的长度
    num_total = len(input_df)
    
    # 遍历输入数据框中指定列的每个图形描述，同时显示进度条
    for graphml in progress_iterable(input_df[column], callbacks.progress, num_total):
        # 加载指定文件名的图形，并将其强制类型转换为字符串或 NetworkX 图对象
        graph = load_graph(cast(str | nx.Graph, graphml))
        
        # 合并图形的节点到主图中，并应用预定义的节点操作
        merge_nodes(mega_graph, graph, node_ops)
        
        # 合并图形的边到主图中，并应用预定义的边操作
        merge_edges(mega_graph, graph, edge_ops)

    # 生成主图的 GraphML 格式表示，并将其作为字符串存储在输出表格的指定列中
    output[to] = ["\n".join(nx.generate_graphml(mega_graph))]

    # 将输出表格作为 TableContainer 对象返回
    return TableContainer(table=output)
# 将子图中的节点合并到目标图中，使用给定的节点操作字典来定义操作。
def merge_nodes(
    target: nx.Graph,
    subgraph: nx.Graph,
    node_ops: dict[str, DetailedAttributeMergeOperation],
):
    """Merge nodes from subgraph into target using the operations defined in node_ops."""
    # 遍历子图中的每个节点
    for node in subgraph.nodes:
        # 如果目标图中没有该节点，则将节点添加到目标图中，并复制其属性（如果有）
        if node not in target.nodes:
            target.add_node(node, **(subgraph.nodes[node] or {}))
        else:
            # 否则，使用指定的节点操作字典来合并节点属性
            merge_attributes(target.nodes[node], subgraph.nodes[node], node_ops)


# 将子图中的边合并到目标图中，使用给定的边操作字典来定义操作。
def merge_edges(
    target_graph: nx.Graph,
    subgraph: nx.Graph,
    edge_ops: dict[str, DetailedAttributeMergeOperation],
):
    """Merge edges from subgraph into target using the operations defined in edge_ops."""
    # 遍历子图中的每条边（包括源节点、目标节点、边的数据）
    for source, target, edge_data in subgraph.edges(data=True):  # type: ignore
        # 如果目标图中没有这条边，则将边添加到目标图中，并复制其数据属性（如果有）
        if not target_graph.has_edge(source, target):
            target_graph.add_edge(source, target, **(edge_data or {}))
        else:
            # 否则，使用指定的边操作字典来合并边的数据属性
            merge_attributes(target_graph.edges[(source, target)], edge_data, edge_ops)


# 将源项中的属性合并到目标项中，使用给定的操作字典来定义操作。
def merge_attributes(
    target_item: dict[str, Any] | None,
    source_item: dict[str, Any] | None,
    ops: dict[str, DetailedAttributeMergeOperation],
):
    """Merge attributes from source_item into target_item using the operations defined in ops."""
    # 如果源项为空，则将其视为空字典
    source_item = source_item or {}
    # 如果目标项为空，则将其视为空字典
    target_item = target_item or {}
    # 遍历操作字典中的每个操作及其对应的属性
    for op_attrib, op in ops.items():
        # 如果操作属性为 "*", 表示要对所有属性执行相同的操作
        if op_attrib == "*":
            # 遍历源项的每个属性
            for attrib in source_item:
                # 如果操作字典中未定义该属性的特定处理方式，则使用默认操作
                if attrib not in ops:
                    # 应用合并操作到目标项的属性中
                    apply_merge_operation(target_item, source_item, attrib, op)
        else:
            # 如果操作属性在源项或目标项中存在
            if op_attrib in source_item or op_attrib in target_item:
                # 应用合并操作到目标项的属性中
                apply_merge_operation(target_item, source_item, op_attrib, op)


# 将合并操作应用到属性中
def apply_merge_operation(
    target_item: dict[str, Any] | None,
    source_item: dict[str, Any] | None,
    attrib: str,
    op: DetailedAttributeMergeOperation,
):
    """Apply the merge operation to the attribute."""
    # 如果源项为空，则将其视为空字典
    source_item = source_item or {}
    # 如果目标项为空，则将其视为空字典
    target_item = target_item or {}

    # 根据合并操作的类型进行相应的处理
    if (
        op.operation == BasicMergeOperation.Replace
        or op.operation == StringOperation.Replace
    ):
        # 替换目标项的属性为源项的属性值，如果源项的属性不存在则用空字符串替代
        target_item[attrib] = source_item.get(attrib, None) or ""
    elif (
        op.operation == BasicMergeOperation.Skip or op.operation == StringOperation.Skip
    ):
        # 如果操作是跳过，则保持目标项属性不变，如果目标项的属性不存在则用空字符串替代
        target_item[attrib] = target_item.get(attrib, None) or ""
    elif op.operation == StringOperation.Concat:
        # 如果操作是连接，则将目标项和源项的属性值按指定分隔符连接起来
        separator = op.separator or DEFAULT_CONCAT_SEPARATOR
        target_attrib = target_item.get(attrib, "") or ""
        source_attrib = source_item.get(attrib, "") or ""
        target_item[attrib] = f"{target_attrib}{separator}{source_attrib}"
        if op.distinct:
            # 如果需要去重，则将连接后的字符串按分隔符拆分、去重、再连接
            # TODO: Slow
            target_item[attrib] = separator.join(
                sorted(set(target_item[attrib].split(separator)))
            )
    # 如果操作是求和操作（NumericOperation.Sum）
    elif op.operation == NumericOperation.Sum:
        # 将目标项中的属性与源项中的属性相加，并存入目标项中
        target_item[attrib] = (target_item.get(attrib, 0) or 0) + (
            source_item.get(attrib, 0) or 0
        )
    # 如果操作是平均操作（NumericOperation.Average）
    elif op.operation == NumericOperation.Average:
        # 将目标项中的属性与源项中的属性相加后除以2，存入目标项中
        target_item[attrib] = (
            (target_item.get(attrib, 0) or 0) + (source_item.get(attrib, 0) or 0)
        ) / 2
    # 如果操作是最大值操作（NumericOperation.Max）
    elif op.operation == NumericOperation.Max:
        # 将目标项中的属性与源项中的属性取最大值，存入目标项中
        target_item[attrib] = max(
            (target_item.get(attrib, 0) or 0), (source_item.get(attrib, 0) or 0)
        )
    # 如果操作是最小值操作（NumericOperation.Min）
    elif op.operation == NumericOperation.Min:
        # 将目标项中的属性与源项中的属性取最小值，存入目标项中
        target_item[attrib] = min(
            (target_item.get(attrib, 0) or 0), (source_item.get(attrib, 0) or 0)
        )
    # 如果操作是乘法操作（NumericOperation.Multiply）
    elif op.operation == NumericOperation.Multiply:
        # 将目标项中的属性与源项中的属性相乘，存入目标项中
        target_item[attrib] = (target_item.get(attrib, 1) or 1) * (
            source_item.get(attrib, 1) or 1
        )
    else:
        # 如果操作不是上述定义的任何一种操作，抛出错误信息
        msg = f"Invalid operation {op.operation}"
        raise ValueError(msg)
# 定义一个函数，用于将属性合并操作标准化为详细属性合并操作
def _get_detailed_attribute_merge_operation(
    value: str | dict[str, Any],  # 函数参数value可以是字符串或字典，字典的值可以是任意类型
) -> DetailedAttributeMergeOperation:
    """Normalize the AttributeMergeOperation into a DetailedAttributeMergeOperation."""
    # 如果value是字符串类型，则将其作为操作值创建DetailedAttributeMergeOperation对象并返回
    if isinstance(value, str):
        return DetailedAttributeMergeOperation(operation=value)
    # 如果value是字典类型，则使用字典中的键值对作为参数创建DetailedAttributeMergeOperation对象并返回
    return DetailedAttributeMergeOperation(**value)
```