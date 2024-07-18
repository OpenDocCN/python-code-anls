# `.\graphrag\graphrag\index\workflows\v1\create_summarized_entities.py`

```py
# 定义一个名为 build_steps 的函数，用于创建包含流程步骤的列表
def build_steps(
    config: PipelineWorkflowConfig,
) -> list[PipelineWorkflowStep]:
    """
    Create the base table for extracted entities.

    ## Dependencies
    * `workflow:create_base_text_units`
    """
    # 从配置中获取 summarize_descriptions 的设置，若没有则使用空字典
    summarize_descriptions_config = config.get("summarize_descriptions", {})
    # 从配置中获取 graphml_snapshot 的设置，若没有则默认为 False
    graphml_snapshot_enabled = config.get("graphml_snapshot", False) or False

    # 返回一个包含两个字典的列表，每个字典描述一个流程步骤
    return [
        {
            "verb": "summarize_descriptions",  # 第一个步骤的操作动词
            "args": {  # 第一个步骤的参数，使用 summarize_descriptions_config 扩展
                **summarize_descriptions_config,
                "column": "entity_graph",  # 指定列名为 'entity_graph'
                "to": "entity_graph",  # 结果输出到 'entity_graph'
                "async_mode": summarize_descriptions_config.get(  # 异步模式，默认为 AsyncIO
                    "async_mode", AsyncType.AsyncIO
                ),
            },
            "input": {"source": "workflow:create_base_extracted_entities"},  # 输入源
        },
        {
            "verb": "snapshot_rows",  # 第二个步骤的操作动词
            "enabled": graphml_snapshot_enabled,  # 是否启用快照功能
            "args": {  # 第二个步骤的参数
                "base_name": "summarized_graph",  # 基本名称为 'summarized_graph'
                "column": "entity_graph",  # 指定列名为 'entity_graph'
                "formats": [{"format": "text", "extension": "graphml"}],  # 输出格式为 graphml
            },
        },
    ]
```