# `.\graphrag\graphrag\index\workflows\v1\create_base_text_units.py`

```py
# 从 datashaper 模块导入 DEFAULT_INPUT_NAME 常量
from datashaper import DEFAULT_INPUT_NAME

# 用于定义工作流名称的字符串变量
workflow_name = "create_base_text_units"

# 定义 build_steps 函数，接受一个 PipelineWorkflowConfig 类型的参数 config
def build_steps(
    config: PipelineWorkflowConfig,
) -> list[PipelineWorkflowStep]:
    """
    Create the base table for text units.

    ## Dependencies
    None
    """

    # 从配置对象 config 中获取 chunk_column 的值，若不存在则使用默认值 "chunk"
    chunk_column_name = config.get("chunk_column", "chunk")

    # 从配置对象 config 中获取 chunk_by 的值，若不存在则使用空列表 []
    chunk_by_columns = config.get("chunk_by", []) or []

    # 从配置对象 config 中获取 n_tokens_column 的值，若不存在则使用默认值 "n_tokens"
    n_tokens_column_name = config.get("n_tokens_column", "n_tokens")
    return [
        {
            "verb": "orderby",
            "args": {
                # 按照 id 列进行升序排序，以确保排序的一致性
                "orders": [
                    {"column": "id", "direction": "asc"},
                ]
            },
            "input": {"source": DEFAULT_INPUT_NAME},
        },
        {
            "verb": "zip",
            "args": {
                # 将文档 id 和文本一起打包
                # 这样在解包分块时可以还原文档 id
                "columns": ["id", "text"],
                "to": "text_with_ids",
            },
        },
        {
            "verb": "aggregate_override",
            "args": {
                # 如果有指定的分块列，按这些列分组
                # 聚合操作将 text_with_ids 列的数据按照 array_agg 方法聚合到 texts 列中
                "groupby": [*chunk_by_columns] if len(chunk_by_columns) > 0 else None,
                "aggregations": [
                    {
                        "column": "text_with_ids",
                        "operation": "array_agg",
                        "to": "texts",
                    }
                ],
            },
        },
        {
            "verb": "chunk",
            "args": {
                # 按照 texts 列的内容进行分块，配置参考 config.get("text_chunk", {})
                "column": "texts",
                "to": "chunks",
                **config.get("text_chunk", {}),
            },
        },
        {
            "verb": "select",
            "args": {
                # 选择需要输出的列，包括分块列和可能的其他分块依据列
                "columns": [*chunk_by_columns, "chunks"],
            },
        },
        {
            "verb": "unroll",
            "args": {
                # 展开 chunks 列的内容
                "column": "chunks",
            },
        },
        {
            "verb": "rename",
            "args": {
                # 重命名 chunks 列为 chunk_column_name
                "columns": {
                    "chunks": chunk_column_name,
                }
            },
        },
        {
            "verb": "genid",
            "args": {
                # 为每个分块生成唯一的 id
                "to": "chunk_id",
                "method": "md5_hash",
                # 使用 chunk_column_name 列作为散列的依据
                "hash": [chunk_column_name],
            },
        },
        {
            "verb": "unzip",
            "args": {
                # 对指定的 chunk_column_name 列进行解包，结果包括 document_ids、chunk_column_name 和 n_tokens_column_name 列
                "column": chunk_column_name,
                "to": ["document_ids", chunk_column_name, n_tokens_column_name],
            },
        },
        {"verb": "copy", "args": {"column": "chunk_id", "to": "id"}},
        {
            # 过滤掉空的分块
            "verb": "filter",
            "args": {
                "column": chunk_column_name,
                "criteria": [
                    {
                        "type": "value",
                        "operator": "is not empty",
                    }
                ],
            },
        },
    ]
```