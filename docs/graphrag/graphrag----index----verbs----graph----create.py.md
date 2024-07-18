# `.\graphrag\graphrag\index\verbs\graph\create.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing create_graph, _get_node_attributes, _get_edge_attributes and _get_attribute_column_mapping methods definition."""

from typing import Any

import networkx as nx
import pandas as pd
from datashaper import TableContainer, VerbCallbacks, VerbInput, progress_iterable, verb

from graphrag.index.utils import clean_str

# 默认的节点属性列表
DEFAULT_NODE_ATTRIBUTES = ["label", "type", "id", "name", "description", "community"]
# 默认的边属性列表
DEFAULT_EDGE_ATTRIBUTES = ["label", "type", "name", "source", "target"]


@verb(name="create_graph")
def create_graph(
    input: VerbInput,
    callbacks: VerbCallbacks,
    to: str,
    type: str,  # noqa A002
    graph_type: str = "undirected",
    **kwargs,
) -> TableContainer:
    """
    Create a graph from a dataframe. The verb outputs a new column containing the graph.

    > Note: This will roll up all rows into a single graph.

    ## Usage
    ```yaml
    verb: create_graph
    args:
        type: node # The type of graph to create, one of: node, edge
        to: <column name> # The name of the column to output the graph to, this will be a graphml graph
        attributes: # The attributes for the nodes / edges
            # If using the node type, the following attributes are required:
            id: <id_column_name>

            # If using the edge type, the following attributes are required:
            source: <source_column_name>
            target: <target_column_name>

            # Other attributes can be added as follows:
            <attribute_name>: <column_name>
            ... for each attribute
    ```py
    """
    # 检查输入的图类型是否合法
    if type != "node" and type != "edge":
        msg = f"Unknown type {type}"
        raise ValueError(msg)

    # 获取输入数据集
    input_df = input.get_input()
    num_total = len(input_df)
    # 创建一个新的空图形对象
    out_graph: nx.Graph = _create_nx_graph(graph_type)

    # 根据类型选择节点或者边的属性
    in_attributes = (
        _get_node_attributes(kwargs) if type == "node" else _get_edge_attributes(kwargs)
    )

    # 获取用于唯一标识节点或者边的列名
    id_col = in_attributes.get(
        "id", in_attributes.get("label", in_attributes.get("name", None))
    )
    source_col = in_attributes.get("source", None)
    target_col = in_attributes.get("target", None)

    # 遍历输入数据集中的每一行
    for _, row in progress_iterable(input_df.iterrows(), callbacks.progress, num_total):
        # 提取每个项的属性，清理键和值
        item_attributes = {
            clean_str(key): _clean_value(row[value])
            for key, value in in_attributes.items()
            if value in row
        }
        # 根据类型添加节点或者边到图形对象中
        if type == "node":
            id = clean_str(row[id_col])
            out_graph.add_node(id, **item_attributes)
        elif type == "edge":
            source = clean_str(row[source_col])
            target = clean_str(row[target_col])
            out_graph.add_edge(source, target, **item_attributes)

    # 将图形对象转换为GraphML格式的字符串
    graphml_string = "".join(nx.generate_graphml(out_graph))
    # 创建包含输出图形的DataFrame
    output_df = pd.DataFrame([{to: graphml_string}])
    # 返回一个 TableContainer 对象，其包含了 output_df 中的数据表
    return TableContainer(table=output_df)
# 清理给定值并返回字符串表示，如果值为 None 则返回空字符串
def _clean_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return clean_str(value)

    msg = f"Value must be a string or None, got {type(value)}"
    raise TypeError(msg)


# 从参数中获取节点属性映射字典
def _get_node_attributes(args: dict[str, Any]) -> dict[str, Any]:
    # 获取属性列映射字典，如果未指定则使用默认节点属性
    mapping = _get_attribute_column_mapping(
        args.get("attributes", DEFAULT_NODE_ATTRIBUTES)
    )
    # 检查映射中是否包含"id"、"label"或"name"，否则抛出异常
    if "id" not in mapping and "label" not in mapping and "name" not in mapping:
        msg = "You must specify an id, label, or name column in the node attributes"
        raise ValueError(msg)
    return mapping


# 从参数中获取边属性映射字典
def _get_edge_attributes(args: dict[str, Any]) -> dict[str, Any]:
    # 获取属性列映射字典，如果未指定则使用默认边属性
    mapping = _get_attribute_column_mapping(
        args.get("attributes", DEFAULT_EDGE_ATTRIBUTES)
    )
    # 检查映射中是否包含"source"和"target"，否则抛出异常
    if "source" not in mapping or "target" not in mapping:
        msg = "You must specify a source and target column in the edge attributes"
        raise ValueError(msg)
    return mapping


# 根据输入的属性字典或列表生成属性列映射字典
def _get_attribute_column_mapping(
    in_attributes: dict[str, Any] | list[str],
) -> dict[str, str]:
    # 如果输入已经是字典，则直接返回
    if isinstance(in_attributes, dict):
        return {
            **in_attributes,
        }

    # 否则以属性名作为键和值生成映射字典
    return {attrib: attrib for attrib in in_attributes}


# 根据图的类型创建对应的 NetworkX 图对象
def _create_nx_graph(graph_type: str) -> nx.Graph:
    # 如果图类型是有向图，则返回有向图对象
    if graph_type == "directed":
        return nx.DiGraph()

    # 否则返回无向图对象
    return nx.Graph()
```