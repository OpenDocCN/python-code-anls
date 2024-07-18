# `.\graphrag\graphrag\index\workflows\v1\create_base_documents.py`

```py
# 版权声明和许可声明
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 导入所需模块和类
"""A module containing build_steps method definition."""
from datashaper import DEFAULT_INPUT_NAME  # 导入默认输入名称

from graphrag.index.config import PipelineWorkflowConfig, PipelineWorkflowStep  # 导入PipelineWorkflowConfig和PipelineWorkflowStep类

# 定义工作流名称变量
workflow_name = "create_base_documents"

# 定义函数 build_steps，接收一个 PipelineWorkflowConfig 对象作为参数，返回 PipelineWorkflowStep 对象列表
def build_steps(
    config: PipelineWorkflowConfig,
) -> list[PipelineWorkflowStep]:
    """
    Create the documents table.

    ## Dependencies
    * `workflow:create_final_text_units`
    """
    # 从配置中获取文档属性列，如果未设置则返回空列表
    document_attribute_columns = config.get("document_attribute_columns", [])
    # 返回一个列表，包含多个字典，每个字典描述一系列数据操作
    return [
        {
            "verb": "unroll",  # 操作：展开
            "args": {"column": "document_ids"},  # 参数：指定列名为 document_ids
            "input": {"source": "workflow:create_final_text_units"},  # 输入源：工作流的最终文本单元创建
        },
        {
            "verb": "select",  # 操作：选择
            "args": {
                # 参数：仅选择列 id, document_ids, text
                "columns": ["id", "document_ids", "text"]
            },
        },
        {
            "id": "rename_chunk_doc_id",  # ID：重命名为 rename_chunk_doc_id
            "verb": "rename",  # 操作：重命名
            "args": {
                "columns": {
                    "document_ids": "chunk_doc_id",  # 重命名列 document_ids 为 chunk_doc_id
                    "id": "chunk_id",  # 重命名列 id 为 chunk_id
                    "text": "chunk_text",  # 重命名列 text 为 chunk_text
                }
            },
        },
        {
            "verb": "join",  # 操作：连接
            "args": {
                # 参数：使用 chunk_doc_id 和 id 进行连接
                "on": ["chunk_doc_id", "id"]
            },
            "input": {"source": "rename_chunk_doc_id", "others": [DEFAULT_INPUT_NAME]},  # 输入源：rename_chunk_doc_id 和其他默认输入
        },
        {
            "id": "docs_with_text_units",  # ID：文档带文本单元
            "verb": "aggregate_override",  # 操作：覆盖聚合
            "args": {
                "groupby": ["id"],  # 按 id 分组
                "aggregations": [
                    {
                        "column": "chunk_id",  # 列：chunk_id
                        "operation": "array_agg",  # 操作：数组聚合
                        "to": "text_units",  # 结果存入 text_units
                    }
                ],
            },
        },
        {
            "verb": "join",  # 操作：连接
            "args": {
                "on": ["id", "id"],  # 参数：使用 id 进行连接
                "strategy": "right outer",  # 策略：右外连接
            },
            "input": {
                "source": "docs_with_text_units",  # 输入源：docs_with_text_units
                "others": [DEFAULT_INPUT_NAME],  # 其他输入源：默认输入名称
            },
        },
        {
            "verb": "rename",  # 操作：重命名
            "args": {"columns": {"text": "raw_content"}},  # 参数：重命名 text 为 raw_content
        },
        *[
            {
                "verb": "convert",  # 操作：转换
                "args": {
                    "column": column,  # 转换列
                    "to": column,  # 转换为相同列名
                    "type": "string",  # 转换类型为字符串
                },
            }
            for column in document_attribute_columns  # 遍历文档属性列列表
        ],
        {
            "verb": "merge_override",  # 操作：覆盖合并
            "enabled": len(document_attribute_columns) > 0,  # 参数：若文档属性列不为空则启用
            "args": {
                "columns": document_attribute_columns,  # 合并的列为文档属性列
                "strategy": "json",  # 合并策略为 JSON
                "to": "attributes",  # 结果存入 attributes
            },
        },
        {"verb": "convert", "args": {"column": "id", "to": "id", "type": "string"}},  # 操作：转换 id 列为字符串类型
    ]
```