# `.\graphrag\graphrag\index\workflows\v1\join_text_units_to_covariate_ids.py`

```py
# 定义了一个函数 build_steps，用于生成构建步骤列表
def build_steps(
    _config: PipelineWorkflowConfig,
) -> list[PipelineWorkflowStep]:
    """
    创建最终的文本单元表。

    ## Dependencies
    * `workflow:create_final_covariates`
    """
    # 返回一个包含两个字典的列表，每个字典描述一个处理步骤
    return [
        {
            "verb": "select",  # 操作动词为选择
            "args": {"columns": ["id", "text_unit_id"]},  # 参数指定选择的列
            "input": {"source": "workflow:create_final_covariates"},  # 输入源是另一个工作流步骤
        },
        {
            "verb": "aggregate_override",  # 操作动词为聚合覆盖
            "args": {
                "groupby": ["text_unit_id"],  # 按文本单元 ID 进行分组
                "aggregations": [
                    {
                        "column": "id",  # 聚合的列是 ID
                        "operation": "array_agg_distinct",  # 使用 array_agg_distinct 操作进行聚合
                        "to": "covariate_ids",  # 结果放入 covariate_ids 字段
                    },
                    {
                        "column": "text_unit_id",  # 聚合的列是 text_unit_id
                        "operation": "any",  # 使用 any 操作
                        "to": "id",  # 结果放入 id 字段
                    },
                ],
            },
        },
    ]
```