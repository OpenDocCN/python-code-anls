# `.\graphrag\graphrag\index\graph\extractors\community_reports\utils.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing community report generation utilities."""

# 引入必要的模块和库
from typing import cast  # 用于类型转换

import pandas as pd  # 引入 pandas 库，用于数据操作

# 导入相关的模块和类
import graphrag.index.graph.extractors.community_reports.schemas as schemas

# 从 graphrag.query.llm.text_utils 导入 num_tokens 函数，用于计算文本的 token 数量
from graphrag.query.llm.text_utils import num_tokens


# 定义函数，设置数据框中的 CONTEXT_SIZE 列，表示上下文中的 token 数量
def set_context_size(df: pd.DataFrame) -> None:
    """Measure the number of tokens in the context."""
    df[schemas.CONTEXT_SIZE] = df[schemas.CONTEXT_STRING].apply(lambda x: num_tokens(x))


# 定义函数，设置数据框中的 CONTEXT_EXCEED_FLAG 列，表示上下文是否超过最大限制
def set_context_exceeds_flag(df: pd.DataFrame, max_tokens: int) -> None:
    """Set a flag to indicate if the context exceeds the limit."""
    df[schemas.CONTEXT_EXCEED_FLAG] = df[schemas.CONTEXT_SIZE].apply(
        lambda x: x > max_tokens
    )


# 定义函数，获取数据框中 communities 的级别（level_column 列的唯一值列表，按降序排序）
def get_levels(df: pd.DataFrame, level_column: str = schemas.NODE_LEVEL) -> list[int]:
    """Get the levels of the communities."""
    result = sorted(df[level_column].fillna(-1).unique().tolist(), reverse=True)
    return [r for r in result if r != -1]


# 定义函数，根据给定的 level 过滤节点数据框，返回指定级别的节点数据框
def filter_nodes_to_level(node_df: pd.DataFrame, level: int) -> pd.DataFrame:
    """Filter nodes to level."""
    return cast(pd.DataFrame, node_df[node_df[schemas.NODE_LEVEL] == level])


# 定义函数，根据给定的节点列表过滤边数据框，返回连接给定节点的边数据框
def filter_edges_to_nodes(edge_df: pd.DataFrame, nodes: list[str]) -> pd.DataFrame:
    """Filter edges to nodes."""
    return cast(
        pd.DataFrame,
        edge_df[
            edge_df[schemas.EDGE_SOURCE].isin(nodes)
            & edge_df[schemas.EDGE_TARGET].isin(nodes)
        ],
    )


# 定义函数，根据给定的节点列表过滤声明数据框，返回主题与给定节点相关的声明数据框
def filter_claims_to_nodes(claims_df: pd.DataFrame, nodes: list[str]) -> pd.DataFrame:
    """Filter edges to nodes."""
    return cast(
        pd.DataFrame,
        claims_df[claims_df[schemas.CLAIM_SUBJECT].isin(nodes)],
    )
```