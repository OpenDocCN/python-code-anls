# `.\graphrag\graphrag\index\workflows\v1\create_final_nodes.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing build_steps method definition."""

from graphrag.index.config import PipelineWorkflowConfig, PipelineWorkflowStep

workflow_name = "create_final_nodes"

def build_steps(
    config: PipelineWorkflowConfig,
) -> list[PipelineWorkflowStep]:
    """
    Create the base table for the document graph.

    ## Dependencies
    * `workflow:create_base_entity_graph`
    """
    # 从配置中获取是否启用快照顶层节点的选项，默认为 False
    snapshot_top_level_nodes = config.get("snapshot_top_level_nodes", False)
    # 从配置中获取布局图是否启用的选项，默认为 True
    layout_graph_enabled = config.get("layout_graph_enabled", True)
    # 定义计算顶层节点位置的步骤序列
    _compute_top_level_node_positions = [
        {
            "verb": "unpack_graph",
            "args": {"column": "positioned_graph", "type": "nodes"},
            "input": {"source": "laid_out_entity_graph"},
        },
        {
            "verb": "filter",
            "args": {
                "column": "level",
                "criteria": [
                    {
                        "type": "value",
                        "operator": "equals",
                        # 获取配置中用于节点位置的级别值，默认为 0
                        "value": config.get("level_for_node_positions", 0),
                    }
                ],
            },
        },
        {
            "verb": "select",
            "args": {"columns": ["id", "x", "y"]},
        },
        {
            "verb": "snapshot",
            # 根据配置决定是否启用快照顶层节点
            "enabled": snapshot_top_level_nodes,
            "args": {
                "name": "top_level_nodes",
                "formats": ["json"],
            },
        },
        {
            "id": "_compute_top_level_node_positions",
            "verb": "rename",
            "args": {
                "columns": {
                    "id": "top_level_node_id",
                }
            },
        },
        {
            "verb": "convert",
            "args": {
                "column": "top_level_node_id",
                "to": "top_level_node_id",
                "type": "string",
            },
        },
    ]
    # 获取布局图配置，如果未指定，则使用默认策略类型为 "umap" 或 "zero"
    layout_graph_config = config.get(
        "layout_graph",
        {
            "strategy": {
                "type": "umap" if layout_graph_enabled else "zero",
            },
        },
    )
    # 返回一个包含多个操作步骤的列表，每个步骤表示为一个字典

        {
            "id": "laid_out_entity_graph",  # 操作步骤的唯一标识符
            "verb": "layout_graph",  # 操作动词，表示对图进行布局
            "args": {
                "embeddings_column": "embeddings",  # 使用的嵌入列名称
                "graph_column": "clustered_graph",  # 使用的聚类图列名称
                "to": "node_positions",  # 布局后节点位置的目标列名
                "graph_to": "positioned_graph",  # 布局后图的目标列名
                **layout_graph_config,  # 其他布局图配置参数
            },
            "input": {"source": "workflow:create_base_entity_graph"},  # 操作的输入源
        },

        {
            "verb": "unpack_graph",  # 操作动词，表示解包图
            "args": {"column": "positioned_graph", "type": "nodes"},  # 解包操作的参数：目标列和节点类型
        },

        {
            "id": "nodes_without_positions",  # 操作步骤的唯一标识符
            "verb": "drop",  # 操作动词，表示丢弃指定列
            "args": {"columns": ["x", "y"]},  # 要丢弃的列名称列表
        },

        *_compute_top_level_node_positions,  # 执行计算顶层节点位置的操作（可能是一个函数或变量）

        {
            "verb": "join",  # 操作动词，表示进行连接操作
            "args": {
                "on": ["id", "top_level_node_id"],  # 连接操作的键列表
            },
            "input": {
                "source": "nodes_without_positions",  # 连接的左侧输入源
                "others": ["_compute_top_level_node_positions"],  # 其他输入源（右侧）
            },
        },

        {
            "verb": "rename",  # 操作动词，表示重命名列
            "args": {"columns": {"label": "title", "cluster": "community"}},  # 列重命名的映射
        },
```