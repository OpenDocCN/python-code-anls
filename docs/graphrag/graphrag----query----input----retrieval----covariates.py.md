# `.\graphrag\graphrag\query\input\retrieval\covariates.py`

```py
# 导入必要的库和模块
from typing import Any, cast  # 导入类型提示相关的模块

import pandas as pd  # 导入 pandas 库

from graphrag.model import Covariate, Entity  # 导入自定义的 Covariate 和 Entity 类


def get_candidate_covariates(
    selected_entities: list[Entity],
    covariates: list[Covariate],
) -> list[Covariate]:
    """获取与选定实体相关的所有协变量。

    Args:
        selected_entities: 选定的实体列表，包含 Entity 类的实例。
        covariates: Covariate 类的实例列表，表示所有可能的协变量。

    Returns:
        与选定实体相关的协变量列表，每个元素是 Covariate 类的实例。
    """
    # 提取选定实体的名称列表
    selected_entity_names = [entity.title for entity in selected_entities]
    # 返回符合条件的协变量列表
    return [
        covariate
        for covariate in covariates
        if covariate.subject_id in selected_entity_names
    ]


def to_covariate_dataframe(covariates: list[Covariate]) -> pd.DataFrame:
    """将协变量列表转换为 pandas 数据框。

    Args:
        covariates: Covariate 类的实例列表，表示要转换的协变量。

    Returns:
        包含协变量信息的 pandas 数据框，每行代表一个协变量的属性。
    """
    # 如果协变量列表为空，则返回空的 pandas 数据框
    if len(covariates) == 0:
        return pd.DataFrame()

    # 添加表头
    header = ["id", "entity"]  # 初始表头包括 id 和 entity
    # 获取第一个协变量的属性，如果存在则扩展表头
    attributes = covariates[0].attributes or {}
    attribute_cols = list(attributes.keys())
    attribute_cols = [col for col in attribute_cols if col not in header]
    header.extend(attribute_cols)

    # 构建记录列表
    records = []
    for covariate in covariates:
        new_record = [
            covariate.short_id if covariate.short_id else "",  # 添加 id 或空字符串
            covariate.subject_id,  # 添加实体的 subject_id
        ]
        # 添加每个属性的值到记录中
        for field in attribute_cols:
            field_value = (
                str(covariate.attributes.get(field)) if covariate.attributes and covariate.attributes.get(field) else ""
            )
            new_record.append(field_value)
        records.append(new_record)

    # 创建 pandas 数据框并返回
    return pd.DataFrame(records, columns=cast(Any, header))
```