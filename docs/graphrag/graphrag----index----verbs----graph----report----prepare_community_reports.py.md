# `.\graphrag\graphrag\index\verbs\graph\report\prepare_community_reports.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing create_community_reports and load_strategy methods definition."""

import logging
from typing import cast

import pandas as pd
from datashaper import (
    TableContainer,
    VerbCallbacks,
    VerbInput,
    progress_iterable,
    verb,
)

import graphrag.index.graph.extractors.community_reports.schemas as schemas
from graphrag.index.graph.extractors.community_reports import (
    filter_claims_to_nodes,
    filter_edges_to_nodes,
    filter_nodes_to_level,
    get_levels,
    set_context_exceeds_flag,
    set_context_size,
    sort_context,
)
from graphrag.index.utils.ds_util import get_named_input_table, get_required_input_table

log = logging.getLogger(__name__)


@verb(name="prepare_community_reports")
def prepare_community_reports(
    input: VerbInput,
    callbacks: VerbCallbacks,
    max_tokens: int = 16_000,
    **_kwargs,
) -> TableContainer:
    """Generate entities for each row, and optionally a graph of those entities."""
    # Prepare Community Reports
    # 获取节点数据表
    node_df = cast(pd.DataFrame, get_required_input_table(input, "nodes").table)
    # 获取边数据表
    edge_df = cast(pd.DataFrame, get_required_input_table(input, "edges").table)
    # 获取声明数据表（如果存在）
    claim_df = get_named_input_table(input, "claims")
    if claim_df is not None:
        claim_df = cast(pd.DataFrame, claim_df.table)

    # 获取节点等级列表
    levels = get_levels(node_df, schemas.NODE_LEVEL)
    dfs = []

    # 遍历节点等级
    for level in progress_iterable(levels, callbacks.progress, len(levels)):
        # 准备特定等级的社区报告数据框
        communities_at_level_df = _prepare_reports_at_level(
            node_df, edge_df, claim_df, level, max_tokens
        )
        dfs.append(communities_at_level_df)

    # 构建所有社区的初始本地上下文
    return TableContainer(table=pd.concat(dfs))


def _prepare_reports_at_level(
    node_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    claim_df: pd.DataFrame | None,
    level: int,
    max_tokens: int = 16_000,
    community_id_column: str = schemas.COMMUNITY_ID,
    node_id_column: str = schemas.NODE_ID,
    node_name_column: str = schemas.NODE_NAME,
    node_details_column: str = schemas.NODE_DETAILS,
    node_level_column: str = schemas.NODE_LEVEL,
    node_degree_column: str = schemas.NODE_DEGREE,
    node_community_column: str = schemas.NODE_COMMUNITY,
    edge_id_column: str = schemas.EDGE_ID,
    edge_source_column: str = schemas.EDGE_SOURCE,
    edge_target_column: str = schemas.EDGE_TARGET,
    edge_degree_column: str = schemas.EDGE_DEGREE,
    edge_details_column: str = schemas.EDGE_DETAILS,
    claim_id_column: str = schemas.CLAIM_ID,
    claim_subject_column: str = schemas.CLAIM_SUBJECT,
    claim_details_column: str = schemas.CLAIM_DETAILS,
):
    def get_edge_details(node_df: pd.DataFrame, edge_df: pd.DataFrame, name_col: str):
        # 合并节点数据帧和边数据帧，基于指定的名称列
        return node_df.merge(
            cast(
                pd.DataFrame,
                edge_df[[name_col, schemas.EDGE_DETAILS]],
            ).rename(columns={name_col: schemas.NODE_NAME}),
            on=schemas.NODE_NAME,
            how="left",
        )

    level_node_df = filter_nodes_to_level(node_df, level)
    # 记录关键信息到日志，显示当前层级的节点数量
    log.info("Number of nodes at level=%s => %s", level, len(level_node_df))
    nodes = level_node_df[node_name_column].tolist()

    # 筛选包含目标节点的边和声明
    level_edge_df = filter_edges_to_nodes(edge_df, nodes)
    level_claim_df = (
        filter_claims_to_nodes(claim_df, nodes) if claim_df is not None else None
    )

    # 按节点连接的边详细信息进行合并
    merged_node_df = pd.concat(
        [
            get_edge_details(level_node_df, level_edge_df, edge_source_column),
            get_edge_details(level_node_df, level_edge_df, edge_target_column),
        ],
        axis=0,
    )
    merged_node_df = (
        merged_node_df.groupby([
            node_name_column,
            node_community_column,
            node_degree_column,
            node_level_column,
        ])
        .agg({node_details_column: "first", edge_details_column: list})
        .reset_index()
    )

    # 按节点连接的声明详细信息进行合并
    if level_claim_df is not None:
        merged_node_df = merged_node_df.merge(
            cast(
                pd.DataFrame,
                level_claim_df[[claim_subject_column, claim_details_column]],
            ).rename(columns={claim_subject_column: node_name_column}),
            on=node_name_column,
            how="left",
        )
    merged_node_df = (
        merged_node_df.groupby([
            node_name_column,
            node_community_column,
            node_level_column,
            node_degree_column,
        ])
        .agg({
            node_details_column: "first",
            edge_details_column: "first",
            **({claim_details_column: list} if level_claim_df is not None else {}),
        })
        .reset_index()
    )

    # 合并所有节点的详细信息，包括名称、度、节点详细、边详细和声明详细
    merged_node_df[schemas.ALL_CONTEXT] = merged_node_df.apply(
        lambda x: {
            node_name_column: x[node_name_column],
            node_degree_column: x[node_degree_column],
            node_details_column: x[node_details_column],
            edge_details_column: x[edge_details_column],
            claim_details_column: x[claim_details_column]
            if level_claim_df is not None
            else [],
        },
        axis=1,
    )

    # 按社区分组所有节点详细信息
    community_df = (
        merged_node_df.groupby(node_community_column)
        .agg({schemas.ALL_CONTEXT: list})
        .reset_index()
    )
    # 将所有社区数据的上下文字符串列映射到经过排序的上下文数据
    community_df[schemas.CONTEXT_STRING] = community_df[schemas.ALL_CONTEXT].apply(
        lambda x: sort_context(
            x,
            node_id_column=node_id_column,
            node_name_column=node_name_column,
            node_details_column=node_details_column,
            edge_id_column=edge_id_column,
            edge_details_column=edge_details_column,
            edge_degree_column=edge_degree_column,
            edge_source_column=edge_source_column,
            edge_target_column=edge_target_column,
            claim_id_column=claim_id_column,
            claim_details_column=claim_details_column,
            community_id_column=community_id_column,
        )
    )
    # 设置社区数据的上下文大小
    set_context_size(community_df)
    # 根据最大标记数设置社区数据是否超出的标志
    set_context_exceeds_flag(community_df, max_tokens)

    # 将社区数据的社区级别列设置为指定的级别
    community_df[schemas.COMMUNITY_LEVEL] = level
    # 返回更新后的社区数据帧
    return community_df
```