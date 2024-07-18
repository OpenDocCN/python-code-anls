# `.\graphrag\graphrag\index\verbs\graph\compute_edge_combined_degree.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing create_graph, _get_node_attributes, _get_edge_attributes and _get_attribute_column_mapping methods definition."""

# 导入所需的模块和类型提示
from typing import cast

import pandas as pd
from datashaper import TableContainer, VerbInput, verb

# 从本地模块导入函数
from graphrag.index.utils.ds_util import get_required_input_table


@verb(name="compute_edge_combined_degree")
def compute_edge_combined_degree(
    input: VerbInput,
    to: str = "rank",
    node_name_column: str = "title",
    node_degree_column: str = "degree",
    edge_source_column: str = "source",
    edge_target_column: str = "target",
    **_kwargs,
) -> TableContainer:
    """
    Compute the combined degree for each edge in a graph.

    Inputs Tables:
    - input: The edge table
    - nodes: The nodes table.

    Args:
    - to: The name of the column to output the combined degree to. Default="rank"
    """
    # 获取输入的边缘数据表
    edge_df: pd.DataFrame = cast(pd.DataFrame, input.get_input())
    # 如果指定输出列已存在于边缘数据表中，直接返回表格容器
    if to in edge_df.columns:
        return TableContainer(table=edge_df)
    # 获取节点度数数据表
    node_degree_df = _get_node_degree_table(input, node_name_column, node_degree_column)

    # 定义函数：将节点度数合并到边缘数据表中
    def join_to_degree(df: pd.DataFrame, column: str) -> pd.DataFrame:
        degree_column = _degree_colname(column)
        result = df.merge(
            # 重命名节点度数数据表的列，以便与边缘数据表进行合并
            node_degree_df.rename(
                columns={node_name_column: column, node_degree_column: degree_column}
            ),
            on=column,
            how="left",
        )
        # 将缺失的节点度数填充为0
        result[degree_column] = result[degree_column].fillna(0)
        return result

    # 将源节点的度数合并到边缘数据表中
    edge_df = join_to_degree(edge_df, edge_source_column)
    # 将目标节点的度数合并到边缘数据表中
    edge_df = join_to_degree(edge_df, edge_target_column)
    # 计算组合度并存储在指定的输出列中
    edge_df[to] = (
        edge_df[_degree_colname(edge_source_column)]
        + edge_df[_degree_colname(edge_target_column)]
    )

    return TableContainer(table=edge_df)


# 辅助函数：生成节点度数列名
def _degree_colname(column: str) -> str:
    return f"{column}_degree"


# 辅助函数：获取节点度数数据表
def _get_node_degree_table(
    input: VerbInput, node_name_column: str, node_degree_column: str
) -> pd.DataFrame:
    # 获取必需的节点数据表
    nodes_container = get_required_input_table(input, "nodes")
    nodes = cast(pd.DataFrame, nodes_container.table)
    # 返回包含指定节点名称和度数列的数据表
    return cast(pd.DataFrame, nodes[[node_name_column, node_degree_column]])
```