# `.\graphrag\graphrag\index\workflows\v1\create_base_extracted_entities.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing build_steps method definition."""

# 导入必要的模块和类
from datashaper import AsyncType

# 从 graphrag.index.config 模块导入 PipelineWorkflowConfig 和 PipelineWorkflowStep 类
from graphrag.index.config import PipelineWorkflowConfig, PipelineWorkflowStep

# 定义工作流的名称
workflow_name = "create_base_extracted_entities"

# 定义 build_steps 函数，接受一个 PipelineWorkflowConfig 类型的参数 config
def build_steps(
    config: PipelineWorkflowConfig,
) -> list[PipelineWorkflowStep]:
    """
    Create the base table for extracted entities.

    ## Dependencies
    * `workflow:create_base_text_units`
    """

    # 从配置中获取实体提取相关的配置信息
    entity_extraction_config = config.get("entity_extract", {})
    
    # 获取配置中的 graphml_snapshot 和 raw_entity_snapshot 的启用状态，默认为 False
    graphml_snapshot_enabled = config.get("graphml_snapshot", False) or False
    raw_entity_snapshot_enabled = config.get("raw_entity_snapshot", False) or False
    # 返回一个包含多个操作描述的列表，每个操作描述为一个字典
    return [
        {
            "verb": "entity_extract",  # 操作类型为实体抽取
            "args": {
                **entity_extraction_config,  # 将entity_extraction_config的内容展开到args中
                "column": entity_extraction_config.get("text_column", "chunk"),  # 指定文本列，默认为"chunk"
                "id_column": entity_extraction_config.get("id_column", "chunk_id"),  # 指定ID列，默认为"chunk_id"
                "async_mode": entity_extraction_config.get(  # 指定异步模式，默认为AsyncIO
                    "async_mode", AsyncType.AsyncIO
                ),
                "to": "entities",  # 输出到"entities"
                "graph_to": "entity_graph",  # 输出到"entity_graph"
            },
            "input": {"source": "workflow:create_base_text_units"},  # 输入源为workflow:create_base_text_units
        },
        {
            "verb": "snapshot",  # 操作类型为快照
            "enabled": raw_entity_snapshot_enabled,  # 指示快照是否启用的标志
            "args": {
                "name": "raw_extracted_entities",  # 指定快照的名称为"raw_extracted_entities"
                "formats": ["json"],  # 指定快照的格式为JSON
            },
        },
        {
            "verb": "merge_graphs",  # 操作类型为合并图形
            "args": {
                "column": "entity_graph",  # 指定要操作的列为"entity_graph"
                "to": "entity_graph",  # 操作结果输出到"entity_graph"
                **config.get(  # 使用配置中指定的图形合并操作，若无配置则使用默认操作
                    "graph_merge_operations",
                    {
                        "nodes": {
                            "source_id": {  # 节点操作，将source_id连接起来，使用逗号和空格分隔
                                "operation": "concat",
                                "delimiter": ", ",
                                "distinct": True,
                            },
                            "description": ({  # 节点描述操作，将描述信息连接起来，使用换行符分隔
                                "operation": "concat",
                                "separator": "\n",
                                "distinct": False,
                            }),
                        },
                        "edges": {
                            "source_id": {  # 边操作，将source_id连接起来，使用逗号和空格分隔
                                "operation": "concat",
                                "delimiter": ", ",
                                "distinct": True,
                            },
                            "description": ({  # 边描述操作，将描述信息连接起来，不使用分隔符
                                "operation": "concat",
                                "separator": "\n",
                                "distinct": False,
                            }),
                            "weight": "sum",  # 边权重操作，求和
                        },
                    },
                ),
            },
        },
        {
            "verb": "snapshot_rows",  # 操作类型为行快照
            "enabled": graphml_snapshot_enabled,  # 指示行快照是否启用的标志
            "args": {
                "base_name": "merged_graph",  # 指定基础名称为"merged_graph"
                "column": "entity_graph",  # 指定要快照的列为"entity_graph"
                "formats": [{"format": "text", "extension": "graphml"}],  # 指定快照的格式为GraphML文本格式
            },
        },
    ]
```