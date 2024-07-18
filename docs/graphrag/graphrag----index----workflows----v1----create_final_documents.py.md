# `.\graphrag\graphrag\index\workflows\v1\create_final_documents.py`

```py
# 版权声明和许可信息
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 导入所需模块和类
"""A module containing build_steps method definition."""
from graphrag.index.config import PipelineWorkflowConfig, PipelineWorkflowStep

# 定义工作流名称
workflow_name = "create_final_documents"

# 定义函数 build_steps，接受一个 PipelineWorkflowConfig 类型的参数 config，
# 返回一个由 PipelineWorkflowStep 对象组成的列表
def build_steps(
    config: PipelineWorkflowConfig,
) -> list[PipelineWorkflowStep]:
    """
    Create the final documents table.

    ## Dependencies
    * `workflow:create_base_documents`
    * `workflow:create_base_document_nodes`
    """
    # 从配置中获取文本嵌入的基础配置
    base_text_embed = config.get("text_embed", {})
    # 获取文档原始内容嵌入的配置，可以从基础文本嵌入配置中继承
    document_raw_content_embed_config = config.get(
        "document_raw_content_embed", base_text_embed
    )
    # 获取是否跳过原始内容嵌入的标志
    skip_raw_content_embedding = config.get("skip_raw_content_embedding", False)
    # 返回包含两个步骤的列表，每个步骤用字典表示
    return [
        {
            "verb": "rename",
            "args": {"columns": {"text_units": "text_unit_ids"}},
            "input": {"source": "workflow:create_base_documents"},
        },
        {
            "verb": "text_embed",
            "enabled": not skip_raw_content_embedding,
            "args": {
                "column": "raw_content",
                "to": "raw_content_embedding",
                **document_raw_content_embed_config,
            },
        },
    ]
```