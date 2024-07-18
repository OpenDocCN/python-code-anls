# `.\graphrag\graphrag\index\workflows\v1\create_final_covariates.py`

```py
# 定义一个名为 build_steps 的函数，用于构建工作流步骤列表
def build_steps(
    config: PipelineWorkflowConfig,  # 接受一个 PipelineWorkflowConfig 类型的参数 config
) -> list[PipelineWorkflowStep]:  # 返回一个 PipelineWorkflowStep 对象的列表

    """
    Create the final covariates table.

    ## Dependencies
    * `workflow:create_base_text_units`
    * `workflow:create_base_extracted_entities`
    """

    # 从配置中获取 claim_extract 字段的配置信息，默认为空字典
    claim_extract_config = config.get("claim_extract", {})

    # 定义一个输入字典，包含来源为 'workflow:create_base_text_units' 的键值对
    input = {"source": "workflow:create_base_text_units"}

    # 返回一个包含多个步骤的列表
    return [
        {
            "verb": "extract_covariates",  # 第一个步骤：提取协变量
            "args": {
                "column": config.get("chunk_column", "chunk"),  # 设置列名，默认为 'chunk'
                "id_column": config.get("chunk_id_column", "chunk_id"),  # 设置 ID 列名，默认为 'chunk_id'
                "resolved_entities_column": "resolved_entities",  # 解析实体的列名为 'resolved_entities'
                "covariate_type": "claim",  # 协变量类型为 'claim'
                "async_mode": config.get("async_mode", AsyncType.AsyncIO),  # 异步模式，默认为 AsyncType.AsyncIO
                **claim_extract_config,  # 使用 claim_extract_config 中的额外参数
            },
            "input": input,  # 此步骤的输入为之前定义的 input 字典
        },
        {
            "verb": "window",  # 第二个步骤：窗口操作
            "args": {"to": "id", "operation": "uuid", "column": "covariate_type"},  # 设置窗口参数
        },
        {
            "verb": "genid",  # 第三个步骤：生成 ID
            "args": {
                "to": "human_readable_id",  # 生成到 human_readable_id 列
                "method": "increment",  # 使用增量方法生成
            },
        },
        {
            "verb": "convert",  # 第四个步骤：转换数据类型
            "args": {
                "column": "human_readable_id",  # 要转换的列为 human_readable_id
                "type": "string",  # 转换为字符串类型
                "to": "human_readable_id",  # 转换后的列名仍为 human_readable_id
            },
        },
        {
            "verb": "rename",  # 第五个步骤：重命名列
            "args": {
                "columns": {
                    "chunk_id": "text_unit_id",  # 将 chunk_id 列重命名为 text_unit_id
                }
            },
        },
        {
            "verb": "select",  # 第六个步骤：选择特定列
            "args": {
                "columns": [  # 选择以下列名的数据
                    "id",
                    "human_readable_id",
                    "covariate_type",
                    "type",
                    "description",
                    "subject_id",
                    "subject_type",
                    "object_id",
                    "object_type",
                    "status",
                    "start_date",
                    "end_date",
                    "source_text",
                    "text_unit_id",
                    "document_ids",
                    "n_tokens",
                ]
            },
        },
    ]
```