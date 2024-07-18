# `.\graphrag\graphrag\index\workflows\v1\create_base_entity_graph.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing build_steps method definition."""

# 定义工作流的名称为 "create_base_entity_graph"
workflow_name = "create_base_entity_graph"

# 定义 build_steps 方法，接收 PipelineWorkflowConfig 类型的参数 config，返回 PipelineWorkflowStep 类型的列表
def build_steps(
    config: PipelineWorkflowConfig,
) -> list[PipelineWorkflowStep]:
    """
    Create the base table for the entity graph.

    ## Dependencies
    * `workflow:create_base_extracted_entities`
    """

    # 从配置中获取聚类图的配置信息，默认使用 "leiden" 策略
    clustering_config = config.get(
        "cluster_graph",
        {"strategy": {"type": "leiden"}},
    )

    # 从配置中获取嵌入图的配置信息，默认使用 "node2vec" 策略，并设置相关参数
    embed_graph_config = config.get(
        "embed_graph",
        {
            "strategy": {
                "type": "node2vec",
                "num_walks": config.get("embed_num_walks", 10),
                "walk_length": config.get("embed_walk_length", 40),
                "window_size": config.get("embed_window_size", 2),
                "iterations": config.get("embed_iterations", 3),
                "random_seed": config.get("embed_random_seed", 86),
            }
        },
    )

    # 检查配置中是否启用了 graphml_snapshot，如果没有，默认为 False
    graphml_snapshot_enabled = config.get("graphml_snapshot", False) or False

    # 检查配置中是否启用了 embed_graph_enabled，如果没有，默认为 False
    embed_graph_enabled = config.get("embed_graph_enabled", False) or False

    # 返回构建好的工作流步骤列表
    return [
        {
            "verb": "cluster_graph",
            "args": {
                **clustering_config,
                "column": "entity_graph",
                "to": "clustered_graph",
                "level_to": "level",
            },
            "input": ({"source": "workflow:create_summarized_entities"}),
        },
        {
            "verb": "snapshot_rows",
            "enabled": graphml_snapshot_enabled,
            "args": {
                "base_name": "clustered_graph",
                "column": "clustered_graph",
                "formats": [{"format": "text", "extension": "graphml"}],
            },
        },
        {
            "verb": "embed_graph",
            "enabled": embed_graph_enabled,
            "args": {
                "column": "clustered_graph",
                "to": "embeddings",
                **embed_graph_config,
            },
        },
        {
            "verb": "snapshot_rows",
            "enabled": graphml_snapshot_enabled,
            "args": {
                "base_name": "embedded_graph",
                "column": "entity_graph",
                "formats": [{"format": "text", "extension": "graphml"}],
            },
        },
        {
            "verb": "select",
            "args": {
                # 仅用于文档，用以知道此工作流包含哪些内容
                "columns": (
                    ["level", "clustered_graph", "embeddings"]
                    if embed_graph_enabled
                    else ["level", "clustered_graph"]
                ),
            },
        },
    ]
```