# `.\graphrag\graphrag\index\workflows\v1\create_final_relationships.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing build_steps method definition."""

# 定义工作流名称为"create_final_relationships"
workflow_name = "create_final_relationships"

# 定义函数build_steps，接受一个PipelineWorkflowConfig类型的参数config，并返回PipelineWorkflowStep类型的列表
def build_steps(
    config: PipelineWorkflowConfig,
) -> list[PipelineWorkflowStep]:
    """
    Create the final relationships table.

    ## Dependencies
    * `workflow:create_base_entity_graph`
    """
    
    # 从config中获取"text_embed"的配置，如果没有则使用空字典
    base_text_embed = config.get("text_embed", {})
    
    # 从config中获取"relationship_description_embed"的配置，如果没有则使用base_text_embed的值
    relationship_description_embed_config = config.get(
        "relationship_description_embed", base_text_embed
    )
    
    # 从config中获取"skip_description_embedding"的配置，默认为False
    skip_description_embedding = config.get("skip_description_embedding", False)

    # 返回一个包含多个步骤的列表，每个步骤是一个字典表示
    return [
        {
            "verb": "unpack_graph",  # 解包图操作
            "args": {
                "column": "clustered_graph",  # 列名设置为"clustered_graph"
                "type": "edges",  # 操作类型为"edges"
            },
            "input": {"source": "workflow:create_base_entity_graph"},  # 输入来自"workflow:create_base_entity_graph"
        },
        {
            "verb": "rename",  # 重命名操作
            "args": {"columns": {"source_id": "text_unit_ids"}},  # 将"source_id"列重命名为"text_unit_ids"
        },
        {
            "verb": "filter",  # 过滤操作
            "args": {
                "column": "level",  # 根据"level"列进行过滤
                "criteria": [{"type": "value", "operator": "equals", "value": 0}],  # 条件为"level"值等于0
            },
        },
        {
            "verb": "text_embed",  # 文本嵌入操作
            "enabled": not skip_description_embedding,  # 根据"skip_description_embedding"判断是否启用
            "args": {
                "embedding_name": "relationship_description",  # 嵌入名称为"relationship_description"
                "column": "description",  # 待嵌入的列为"description"
                "to": "description_embedding",  # 嵌入结果存储在"description_embedding"列中
                **relationship_description_embed_config,  # 其他嵌入配置从relationship_description_embed_config获取
            },
        },
        {
            "id": "pruned_edges",  # 此步骤的唯一标识为"pruned_edges"
            "verb": "drop",  # 删除操作
            "args": {"columns": ["level"]},  # 删除"level"列
        },
        {
            "id": "filtered_nodes",  # 此步骤的唯一标识为"filtered_nodes"
            "verb": "filter",  # 过滤操作
            "args": {
                "column": "level",  # 根据"level"列进行过滤
                "criteria": [{"type": "value", "operator": "equals", "value": 0}],  # 条件为"level"值等于0
            },
            "input": "workflow:create_final_nodes",  # 输入来自"workflow:create_final_nodes"
        },
        {
            "verb": "compute_edge_combined_degree",  # 计算边的综合度量操作
            "args": {"to": "rank"},  # 将结果存储在"rank"列中
            "input": {
                "source": "pruned_edges",  # 输入的边来自"pruned_edges"
                "nodes": "filtered_nodes",  # 节点来自"filtered_nodes"
            },
        },
        {
            "verb": "convert",  # 类型转换操作
            "args": {
                "column": "human_readable_id",  # 待转换的列为"human_readable_id"
                "type": "string",  # 转换为字符串类型
                "to": "human_readable_id",  # 转换结果存储在"human_readable_id"列中
            },
        },
        {
            "verb": "convert",  # 类型转换操作
            "args": {
                "column": "text_unit_ids",  # 待转换的列为"text_unit_ids"
                "type": "array",  # 转换为数组类型
                "to": "text_unit_ids",  # 转换结果存储在"text_unit_ids"列中
            },
        },
    ]
```