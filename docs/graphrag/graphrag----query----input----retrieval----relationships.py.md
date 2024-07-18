# `.\graphrag\graphrag\query\input\retrieval\relationships.py`

```py
# 版权声明和许可信息
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 导入必要的模块和类
"""Util functions to retrieve relationships from a collection."""
from typing import Any, cast  # 导入类型提示相关库

import pandas as pd  # 导入 pandas 库

from graphrag.model import Entity, Relationship  # 导入自定义的 Entity 和 Relationship 类


def get_in_network_relationships(
    selected_entities: list[Entity],
    relationships: list[Relationship],
    ranking_attribute: str = "rank",
) -> list[Relationship]:
    """Get all directed relationships between selected entities, sorted by ranking_attribute."""
    selected_entity_names = [entity.title for entity in selected_entities]  # 提取选定实体的名称列表
    selected_relationships = [
        relationship
        for relationship in relationships
        if relationship.source in selected_entity_names
        and relationship.target in selected_entity_names
    ]  # 筛选出源和目标均在选定实体中的关系
    if len(selected_relationships) <= 1:
        return selected_relationships  # 如果筛选后的关系数量小于等于1，则直接返回

    # 根据指定的排名属性对关系进行排序
    return sort_relationships_by_ranking_attribute(
        selected_relationships, selected_entities, ranking_attribute
    )


def get_out_network_relationships(
    selected_entities: list[Entity],
    relationships: list[Relationship],
    ranking_attribute: str = "rank",
) -> list[Relationship]:
    """Get relationships from selected entities to other entities that are not within the selected entities, sorted by ranking_attribute."""
    selected_entity_names = [entity.title for entity in selected_entities]  # 提取选定实体的名称列表
    source_relationships = [
        relationship
        for relationship in relationships
        if relationship.source in selected_entity_names
        and relationship.target not in selected_entity_names
    ]  # 筛选出源在选定实体中且目标不在选定实体中的关系
    target_relationships = [
        relationship
        for relationship in relationships
        if relationship.target in selected_entity_names
        and relationship.source not in selected_entity_names
    ]  # 筛选出目标在选定实体中且源不在选定实体中的关系
    selected_relationships = source_relationships + target_relationships  # 合并筛选出的关系列表
    return sort_relationships_by_ranking_attribute(
        selected_relationships, selected_entities, ranking_attribute
    )  # 根据指定的排名属性对关系进行排序


def get_candidate_relationships(
    selected_entities: list[Entity],
    relationships: list[Relationship],
) -> list[Relationship]:
    """Get all relationships that are associated with the selected entities."""
    selected_entity_names = [entity.title for entity in selected_entities]  # 提取选定实体的名称列表
    return [
        relationship
        for relationship in relationships
        if relationship.source in selected_entity_names
        or relationship.target in selected_entity_names
    ]  # 筛选出与选定实体相关联的所有关系


def get_entities_from_relationships(
    relationships: list[Relationship], entities: list[Entity]
) -> list[Entity]:
    """Get all entities that are associated with the selected relationships."""
    selected_entity_names = [relationship.source for relationship in relationships] + [
        relationship.target for relationship in relationships
    ]  # 提取所有与选定关系相关联的实体名称
    return [entity for entity in entities if entity.title in selected_entity_names]  # 筛选出与选定关系相关联的所有实体
# 计算基于源和目标实体的组合排名的关系默认排名
def calculate_relationship_combined_rank(
    relationships: list[Relationship],  # 输入参数：关系列表
    entities: list[Entity],  # 输入参数：实体列表
    ranking_attribute: str = "rank",  # 输入参数：排名属性，默认为 "rank"
) -> list[Relationship]:  # 返回类型：关系列表
    """Calculate default rank for a relationship based on the combined rank of source and target entities."""
    # 创建实体标题到实体对象的映射字典
    entity_mappings = {entity.title: entity for entity in entities}

    # 遍历每个关系对象
    for relationship in relationships:
        # 如果关系对象没有属性，则初始化为空字典
        if relationship.attributes is None:
            relationship.attributes = {}
        
        # 获取关系的源实体和目标实体对象
        source = entity_mappings.get(relationship.source)
        target = entity_mappings.get(relationship.target)
        
        # 获取源实体和目标实体的排名，若不存在则默认为0
        source_rank = source.rank if source and source.rank else 0
        target_rank = target.rank if target and target.rank else 0
        
        # 计算关系对象的排名属性为源实体排名与目标实体排名之和
        relationship.attributes[ranking_attribute] = source_rank + target_rank  # type: ignore
    
    # 返回更新后的关系列表
    return relationships


# 根据排名属性对关系进行排序
def sort_relationships_by_ranking_attribute(
    relationships: list[Relationship],  # 输入参数：关系列表
    entities: list[Entity],  # 输入参数：实体列表
    ranking_attribute: str = "rank",  # 输入参数：排名属性，默认为 "rank"
) -> list[Relationship]:  # 返回类型：关系列表
    """
    Sort relationships by a ranking_attribute.

    If no ranking attribute exists, sort by combined rank of source and target entities.
    """
    # 如果关系列表为空，则直接返回空列表
    if len(relationships) == 0:
        return relationships

    # 获取第一个关系对象的所有属性名作为排序依据
    attribute_names = (
        list(relationships[0].attributes.keys()) if relationships[0].attributes else []
    )
    
    # 如果指定的排名属性存在于关系对象的属性中，则按照排名属性进行降序排序
    if ranking_attribute in attribute_names:
        relationships.sort(
            key=lambda x: int(x.attributes[ranking_attribute]) if x.attributes else 0,
            reverse=True,
        )
    # 如果排名属性为 "weight"，则按照权重进行降序排序
    elif ranking_attribute == "weight":
        relationships.sort(key=lambda x: x.weight if x.weight else 0.0, reverse=True)
    else:
        # 如果排名属性不存在，则调用函数计算关系的组合排名，并按照该属性进行降序排序
        relationships = calculate_relationship_combined_rank(
            relationships, entities, ranking_attribute
        )
        relationships.sort(
            key=lambda x: int(x.attributes[ranking_attribute]) if x.attributes else 0,
            reverse=True,
        )
    
    # 返回排序后的关系列表
    return relationships


# 将关系列表转换为 pandas 数据框架
def to_relationship_dataframe(
    relationships: list[Relationship],  # 输入参数：关系列表
    include_relationship_weight: bool = True  # 输入参数：是否包含关系权重，默认为 True
) -> pd.DataFrame:  # 返回类型：Pandas 数据框架
    """Convert a list of relationships to a pandas dataframe."""
    # 如果关系列表为空，则返回一个空的数据框架
    if len(relationships) == 0:
        return pd.DataFrame()

    # 定义数据框架的列头，包括 id、source、target 和 description
    header = ["id", "source", "target", "description"]
    
    # 如果需要包含关系权重，则将 "weight" 列头加入
    if include_relationship_weight:
        header.append("weight")
    
    # 获取第一个关系对象的所有属性名作为数据框架的额外列头
    attribute_cols = (
        list(relationships[0].attributes.keys()) if relationships[0].attributes else []
    )
    # 筛选出不在基础列头中的属性列
    attribute_cols = [col for col in attribute_cols if col not in header]
    header.extend(attribute_cols)

    # 初始化记录列表
    records = []
    # 遍历relationships列表中的每个关系对象
    for rel in relationships:
        # 创建一个新的记录列表，包含关系对象的特定属性，如果属性为空则使用空字符串代替
        new_record = [
            rel.short_id if rel.short_id else "",  # 如果有short_id则添加到记录，否则添加空字符串
            rel.source,  # 添加关系的源
            rel.target,  # 添加关系的目标
            rel.description if rel.description else "",  # 如果有描述则添加到记录，否则添加空字符串
        ]
        # 如果需要包括关系的权重信息
        if include_relationship_weight:
            new_record.append(str(rel.weight if rel.weight else ""))  # 如果有权重则添加到记录，否则添加空字符串
        # 遍历属性列名列表，将关系对象的各个属性值添加到记录中，如果属性值为空则添加空字符串
        for field in attribute_cols:
            field_value = (
                str(rel.attributes.get(field))  # 获取属性值并转换为字符串
                if rel.attributes and rel.attributes.get(field)  # 检查属性存在且不为空
                else ""  # 否则添加空字符串
            )
            new_record.append(field_value)  # 添加属性值到记录
        records.append(new_record)  # 将新记录添加到记录列表中
    # 使用记录列表创建一个Pandas DataFrame，并指定列名为header
    return pd.DataFrame(records, columns=cast(Any, header))
```