# `.\graphrag\graphrag\query\input\retrieval\text_units.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Util functions to retrieve text units from a collection."""

# 引入需要的类型提示和类型转换
from typing import Any, cast

# 引入 pandas 库，用于处理数据和创建 DataFrame
import pandas as pd

# 从 graphrag.model 模块中引入 Entity 和 TextUnit 类
from graphrag.model import Entity, TextUnit


# 定义函数：获取与选定实体关联的所有文本单元，并返回一个 DataFrame
def get_candidate_text_units(
    selected_entities: list[Entity],
    text_units: list[TextUnit],
) -> pd.DataFrame:
    """Get all text units that are associated to selected entities."""
    # 从选定的实体列表中提取文本单元的 ID
    selected_text_ids = [
        entity.text_unit_ids for entity in selected_entities if entity.text_unit_ids
    ]
    # 将嵌套的文本单元 ID 列表扁平化
    selected_text_ids = [item for sublist in selected_text_ids for item in sublist]
    # 根据选定的文本单元 ID 筛选出对应的文本单元对象
    selected_text_units = [unit for unit in text_units if unit.id in selected_text_ids]
    # 调用 to_text_unit_dataframe 函数将文本单元转换成 DataFrame 并返回
    return to_text_unit_dataframe(selected_text_units)


# 定义函数：将文本单元列表转换为 pandas 的 DataFrame
def to_text_unit_dataframe(text_units: list[TextUnit]) -> pd.DataFrame:
    """Convert a list of text units to a pandas dataframe."""
    # 如果文本单元列表为空，则返回空的 DataFrame
    if len(text_units) == 0:
        return pd.DataFrame()

    # 准备表头，初始包含 "id" 和 "text" 两列
    header = ["id", "text"]
    # 获取第一个文本单元的所有属性字段作为额外的列，排除 "id" 和 "text"
    attribute_cols = (
        list(text_units[0].attributes.keys()) if text_units[0].attributes else []
    )
    attribute_cols = [col for col in attribute_cols if col not in header]
    header.extend(attribute_cols)

    # 初始化记录列表
    records = []
    # 遍历文本单元列表，将每个文本单元的信息转换为一行记录
    for unit in text_units:
        # 构建新记录，包括文本单元的 ID 和文本内容，以及各属性字段的值
        new_record = [
            unit.short_id,
            unit.text,
            *[
                str(unit.attributes.get(field, ""))
                if unit.attributes and unit.attributes.get(field)
                else ""
                for field in attribute_cols
            ],
        ]
        # 将新记录加入记录列表
        records.append(new_record)
    
    # 使用记录列表和表头创建一个新的 pandas DataFrame，并指定列名
    return pd.DataFrame(records, columns=cast(Any, header))
```