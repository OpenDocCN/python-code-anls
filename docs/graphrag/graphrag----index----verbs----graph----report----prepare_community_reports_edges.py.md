# `.\graphrag\graphrag\index\verbs\graph\report\prepare_community_reports_edges.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing create_graph, _get_node_attributes, _get_edge_attributes and _get_attribute_column_mapping methods definition."""

# 导入必要的模块和函数
from typing import cast  # 导入类型提示的cast函数

import pandas as pd  # 导入pandas库
from datashaper import TableContainer, VerbInput, verb  # 从datashaper模块导入TableContainer, VerbInput和verb函数

# 从指定路径导入相关的常量和变量
from graphrag.index.graph.extractors.community_reports.schemas import (
    EDGE_DEGREE,
    EDGE_DESCRIPTION,
    EDGE_DETAILS,
    EDGE_ID,
    EDGE_SOURCE,
    EDGE_TARGET,
)

# 未找到描述信息时的默认值
_MISSING_DESCRIPTION = "No Description"


@verb(name="prepare_community_reports_edges")  # 使用verb装饰器声明函数名为prepare_community_reports_edges
def prepare_community_reports_edges(
    input: VerbInput,  # 输入参数为VerbInput类型
    to: str = EDGE_DETAILS,  # 指定to参数默认值为EDGE_DETAILS
    id_column: str = EDGE_ID,  # 指定id_column参数默认值为EDGE_ID
    source_column: str = EDGE_SOURCE,  # 指定source_column参数默认值为EDGE_SOURCE
    target_column: str = EDGE_TARGET,  # 指定target_column参数默认值为EDGE_TARGET
    description_column: str = EDGE_DESCRIPTION,  # 指定description_column参数默认值为EDGE_DESCRIPTION
    degree_column: str = EDGE_DEGREE,  # 指定degree_column参数默认值为EDGE_DEGREE
    **_kwargs,  # 其余关键字参数
) -> TableContainer:
    """Merge edge details into an object."""
    # 获取输入数据，并转换为DataFrame格式，对缺失值进行填充
    edge_df: pd.DataFrame = cast(pd.DataFrame, input.get_input()).fillna(
        value={description_column: _MISSING_DESCRIPTION}
    )
    # 将边的详细信息合并为一个对象，存储到to参数指定的列中
    edge_df[to] = edge_df.apply(
        lambda x: {
            id_column: x[id_column],
            source_column: x[source_column],
            target_column: x[target_column],
            description_column: x[description_column],
            degree_column: x[degree_column],
        },
        axis=1,  # 按行进行操作
    )
    # 返回包含处理后数据的TableContainer对象
    return TableContainer(table=edge_df)
```