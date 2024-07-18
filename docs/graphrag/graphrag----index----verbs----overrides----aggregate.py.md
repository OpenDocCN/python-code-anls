# `.\graphrag\graphrag\index\verbs\overrides\aggregate.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing 'Aggregation' model."""

# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 引入必要的模块和类
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd
from datashaper import (
    FieldAggregateOperation,
    Progress,
    TableContainer,
    VerbCallbacks,
    VerbInput,
    aggregate_operation_mapping,
    verb,
)

# 定义数组类型的聚合操作列表
ARRAY_AGGREGATIONS = [
    FieldAggregateOperation.ArrayAgg,
    FieldAggregateOperation.ArrayAggDistinct,
]

# TODO: This thing is kinda gross
# Also, it diverges from the original aggregate verb, since it doesn't support the same syntax

# 定义名为'aggregate_override'的自定义聚合方法装饰器
@verb(name="aggregate_override")
def aggregate(
    input: VerbInput,
    callbacks: VerbCallbacks,
    aggregations: list[dict[str, Any]],
    groupby: list[str] | None = None,
    **_kwargs: dict,
) -> TableContainer:
    """Aggregate method definition."""
    # 载入并解析聚合操作定义
    aggregations_to_apply = _load_aggregations(aggregations)
    
    # 生成针对 Pandas 的聚合操作字典
    df_aggregations = {
        agg.column: _get_pandas_agg_operation(agg)
        for agg in aggregations_to_apply.values()
    }
    
    # 获取输入表格
    input_table = input.get_input()
    
    # 更新进度到回调函数，起始进度为 0%
    callbacks.progress(Progress(percent=0))

    # 根据是否有分组字段，进行数据分组
    if groupby is None:
        output_grouped = input_table.groupby(lambda _x: True)
    else:
        output_grouped = input_table.groupby(groupby, sort=False)
    
    # 应用 Pandas 的聚合操作到分组数据上
    output = cast(pd.DataFrame, output_grouped.agg(df_aggregations))
    
    # 重命名输出表格的列名，按照聚合操作的目标名
    output.rename(
        columns={agg.column: agg.to for agg in aggregations_to_apply.values()},
        inplace=True,
    )
    
    # 设置输出表格的列名为聚合操作的目标名
    output.columns = [agg.to for agg in aggregations_to_apply.values()]

    # 更新进度到回调函数，完成进度为 100%
    callbacks.progress(Progress(percent=1))

    # 返回结果作为一个 TableContainer 对象，重置索引
    return TableContainer(table=output.reset_index())


# 定义数据类 Aggregation，用于存储聚合操作的详细信息
@dataclass
class Aggregation:
    """Aggregation class method definition."""

    column: str | None
    operation: str
    to: str

    # Only useful for the concat operation
    separator: str | None = None


# 获取 Pandas 的聚合操作函数，根据传入的 Aggregation 对象
def _get_pandas_agg_operation(agg: Aggregation) -> Any:
    # TODO: Merge into datashaper
    # 根据操作类型返回相应的 Pandas 聚合函数
    if agg.operation == "string_concat":
        return (agg.separator or ",").join
    return aggregate_operation_mapping[FieldAggregateOperation(agg.operation)]


# 载入并解析聚合操作列表，返回字典形式的 Aggregation 对象
def _load_aggregations(
    aggregations: list[dict[str, Any]],
) -> dict[str, Aggregation]:
    return {
        aggregation["column"]: Aggregation(
            aggregation["column"], aggregation["operation"], aggregation["to"]
        )
        for aggregation in aggregations
    }
```