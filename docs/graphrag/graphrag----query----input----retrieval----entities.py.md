# `.\graphrag\graphrag\query\input\retrieval\entities.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Util functions to get entities from a collection."""

import uuid
from collections.abc import Iterable
from typing import Any, cast

import pandas as pd

from graphrag.model import Entity


def get_entity_by_key(
    entities: Iterable[Entity], key: str, value: str | int
) -> Entity | None:
    """Get entity by key."""
    # 遍历实体集合中的每个实体
    for entity in entities:
        # 如果值为字符串且是有效的 UUID，则进行特殊处理
        if isinstance(value, str) and is_valid_uuid(value):
            # 检查实体的指定键是否等于原始值或去掉连字符后的值
            if getattr(entity, key) == value or getattr(entity, key) == value.replace(
                "-", ""
            ):
                return entity  # 返回找到的实体
        else:
            # 检查实体的指定键是否等于给定的值
            if getattr(entity, key) == value:
                return entity  # 返回找到的实体
    return None  # 若未找到匹配的实体，则返回 None


def get_entity_by_name(entities: Iterable[Entity], entity_name: str) -> list[Entity]:
    """Get entities by name."""
    # 使用列表推导式遍历实体集合，筛选出标题与给定实体名匹配的实体
    return [entity for entity in entities if entity.title == entity_name]


def get_entity_by_attribute(
    entities: Iterable[Entity], attribute_name: str, attribute_value: Any
) -> list[Entity]:
    """Get entities by attribute."""
    # 使用列表推导式遍历实体集合，筛选出具有指定属性名且属性值与给定值相匹配的实体
    return [
        entity
        for entity in entities
        if entity.attributes
        and entity.attributes.get(attribute_name) == attribute_value
    ]


def to_entity_dataframe(
    entities: list[Entity],
    include_entity_rank: bool = True,
    rank_description: str = "number of relationships",
) -> pd.DataFrame:
    """Convert a list of entities to a pandas dataframe."""
    # 若实体列表为空，则返回一个空的 pandas DataFrame
    if len(entities) == 0:
        return pd.DataFrame()
    
    # 定义 DataFrame 的列头，初始包含'id', 'entity', 'description'
    header = ["id", "entity", "description"]
    
    # 若需要包含实体排名信息，则添加对应列名到列头中
    if include_entity_rank:
        header.append(rank_description)
    
    # 筛选出实体的属性列名，并加入到列头中，确保没有重复列名
    attribute_cols = (
        list(entities[0].attributes.keys()) if entities[0].attributes else []
    )
    attribute_cols = [col for col in attribute_cols if col not in header]
    header.extend(attribute_cols)

    # 初始化记录列表
    records = []
    
    # 遍历实体列表，为每个实体创建一行记录
    for entity in entities:
        # 创建新的记录行，包括'id', 'entity', 'description'字段
        new_record = [
            entity.short_id if entity.short_id else "",
            entity.title,
            entity.description if entity.description else "",
        ]
        
        # 如果需要包含实体排名信息，则添加对应的排名值到记录行中
        if include_entity_rank:
            new_record.append(str(entity.rank))

        # 遍历属性列，将每个属性值添加到记录行中
        for field in attribute_cols:
            field_value = (
                str(entity.attributes.get(field))
                if entity.attributes and entity.attributes.get(field)
                else ""
            )
            new_record.append(field_value)
        
        # 将当前记录行添加到记录列表中
        records.append(new_record)
    
    # 使用记录列表和列头创建一个 pandas DataFrame，并返回结果
    return pd.DataFrame(records, columns=cast(Any, header))


def is_valid_uuid(value: str) -> bool:
    """Determine if a string is a valid UUID."""
    # 尝试将字符串转换为 UUID 对象，若成功则返回 True，否则返回 False
    try:
        uuid.UUID(str(value))
    except ValueError:
        return False
    else:
        return True
```