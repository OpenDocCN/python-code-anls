# `.\graphrag\graphrag\index\workflows\v1\join_text_units_to_relationship_ids.py`

```py
# 版权声明及许可信息
# 2024 年版权所有 Microsoft Corporation.
# 根据 MIT 许可证授权

"""定义了 build_steps 方法的模块。"""

# 导入所需的模块和类
from graphrag.index.config import PipelineWorkflowConfig, PipelineWorkflowStep

# 定义工作流名称
workflow_name = "join_text_units_to_relationship_ids"

def build_steps(
    _config: PipelineWorkflowConfig,
) -> list[PipelineWorkflowStep]:
    """
    创建从文本单元 ID 到关系 ID 的连接表。

    ## 依赖项
    * `workflow:create_final_relationships`
    """
    # 返回包含多个步骤的列表，每个步骤是一个字典形式的操作描述
    return [
        {
            "verb": "select",
            "args": {"columns": ["id", "text_unit_ids"]},
            "input": {"source": "workflow:create_final_relationships"},
        },
        {
            "verb": "unroll",
            "args": {
                "column": "text_unit_ids",
            },
        },
        {
            "verb": "aggregate_override",
            "args": {
                "groupby": ["text_unit_ids"],
                "aggregations": [
                    {
                        "column": "id",
                        "operation": "array_agg_distinct",
                        "to": "relationship_ids",
                    },
                    {
                        "column": "text_unit_ids",
                        "operation": "any",
                        "to": "id",
                    },
                ],
            },
        },
        {
            "id": "text_unit_id_to_relationship_ids",
            "verb": "select",
            "args": {"columns": ["id", "relationship_ids"]},
        },
    ]
```