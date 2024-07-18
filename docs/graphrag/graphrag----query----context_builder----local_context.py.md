# `.\graphrag\graphrag\query\context_builder\local_context.py`

```py
# 版权声明和许可证信息
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 导入所需模块和库
"""Local Context Builder."""
from collections import defaultdict
from typing import Any, cast

import pandas as pd
import tiktoken

# 导入数据模型相关的类
from graphrag.model import Covariate, Entity, Relationship
# 导入处理数据查询和转换的函数和类
from graphrag.query.input.retrieval.covariates import (
    get_candidate_covariates,
    to_covariate_dataframe,
)
from graphrag.query.input.retrieval.entities import to_entity_dataframe
from graphrag.query.input.retrieval.relationships import (
    get_candidate_relationships,
    get_entities_from_relationships,
    get_in_network_relationships,
    get_out_network_relationships,
    to_relationship_dataframe,
)
# 导入文本处理工具中的单词计数函数
from graphrag.query.llm.text_utils import num_tokens


def build_entity_context(
    selected_entities: list[Entity],
    token_encoder: tiktoken.Encoding | None = None,
    max_tokens: int = 8000,
    include_entity_rank: bool = True,
    rank_description: str = "number of relationships",
    column_delimiter: str = "|",
    context_name="Entities",
) -> tuple[str, pd.DataFrame]:
    """Prepare entity data table as context data for system prompt."""
    # 如果未选择任何实体，则返回空字符串和空的数据框
    if len(selected_entities) == 0:
        return "", pd.DataFrame()

    # 准备当前上下文文本的标题行
    current_context_text = f"-----{context_name}-----" + "\n"
    header = ["id", "entity", "description"]
    # 如果包括实体排名，则添加排名描述列
    if include_entity_rank:
        header.append(rank_description)
    
    # 获取第一个实体的属性作为列标题
    attribute_cols = (
        list(selected_entities[0].attributes.keys())
        if selected_entities[0].attributes
        else []
    )
    header.extend(attribute_cols)
    current_context_text += column_delimiter.join(header) + "\n"

    # 计算当前上下文文本中的单词数
    current_tokens = num_tokens(current_context_text, token_encoder)

    # 存储所有实体数据记录的列表，包括标题行
    all_context_records = [header]

    # 遍历选定的实体列表，构建每个实体的上下文数据行
    for entity in selected_entities:
        new_context = [
            entity.short_id if entity.short_id else "",  # 实体的短标识符
            entity.title,  # 实体的标题
            entity.description if entity.description else "",  # 实体的描述
        ]
        # 如果包括实体排名，则添加实体的排名值
        if include_entity_rank:
            new_context.append(str(entity.rank))
        
        # 添加每个属性字段的值到上下文数据行中
        for field in attribute_cols:
            field_value = (
                str(entity.attributes.get(field))  # 获取属性字段的值
                if entity.attributes and entity.attributes.get(field)
                else ""
            )
            new_context.append(field_value)

        # 将每行上下文数据转换为文本形式，并计算其单词数
        new_context_text = column_delimiter.join(new_context) + "\n"
        new_tokens = num_tokens(new_context_text, token_encoder)

        # 如果添加当前实体的上下文数据超出最大允许单词数，则停止添加更多实体数据
        if current_tokens + new_tokens > max_tokens:
            break
        
        # 更新当前上下文文本和单词数
        current_context_text += new_context_text
        all_context_records.append(new_context)
        current_tokens += new_tokens

    # 如果有有效的上下文数据记录，则创建数据框
    if len(all_context_records) > 1:
        record_df = pd.DataFrame(
            all_context_records[1:], columns=cast(Any, all_context_records[0])
        )
    else:
        record_df = pd.DataFrame()

    # 返回当前上下文文本和相应的数据框
    return current_context_text, record_df


def build_covariates_context(
    selected_entities: list[Entity],
    covariates: list[Covariate],
    # covariates 是一个列表，包含 Covariate 类型的元素，用于存储协变量数据

    token_encoder: tiktoken.Encoding | None = None,
    # token_encoder 是一个类型为 tiktoken.Encoding 或 None 的变量，用于编码令牌，缺省为 None

    max_tokens: int = 8000,
    # max_tokens 是一个整数变量，表示最大令牌数，默认为 8000

    column_delimiter: str = "|",
    # column_delimiter 是一个字符串变量，表示列分隔符，默认为竖线 "|"

    context_name: str = "Covariates",
    # context_name 是一个字符串变量，表示上下文名称，默认为 "Covariates"
    # 准备关系数据表作为系统提示的上下文数据
    selected_relationships = _filter_relationships(
        selected_entities=selected_entities,
        relationships=relationships,
        top_k_relationships=top_k_relationships,
        relationship_ranking_attribute=relationship_ranking_attribute,
    )

    # 如果选定的实体或关系为空，则返回空字符串和空的DataFrame
    if len(selected_entities) == 0 or len(selected_relationships) == 0:
        return "", pd.DataFrame()

    # 添加标题
    current_context_text = f"-----{context_name}-----" + "\n"
    # 定义表头，包括基本字段 "id", "source", "target", "description"
    header = ["id", "source", "target", "description"]
    
    # 如果需要包含关系权重信息，则添加 "weight" 到表头
    if include_relationship_weight:
        header.append("weight")
    
    # 根据所选关系的属性列生成属性列表
    attribute_cols = (
        list(selected_relationships[0].attributes.keys())  # 获取第一个关系的属性键列表
        if selected_relationships[0].attributes  # 如果第一个关系有属性
        else []  # 否则为空列表
    )
    
    # 筛选掉在表头中已有的属性列
    attribute_cols = [col for col in attribute_cols if col not in header]
    
    # 将筛选后的属性列添加到表头
    header.extend(attribute_cols)

    # 将表头转换为以列分隔符连接的字符串，并在末尾添加换行符，加入当前上下文文本
    current_context_text += column_delimiter.join(header) + "\n"

    # 计算当前上下文文本的token数量
    current_tokens = num_tokens(current_context_text, token_encoder)

    # 初始化存储所有上下文记录的列表，将表头作为第一个记录加入
    all_context_records = [header]
    
    # 遍历选定的关系列表
    for rel in selected_relationships:
        # 创建新的上下文记录
        new_context = [
            rel.short_id if rel.short_id else "",  # 如果存在短ID则添加，否则为空字符串
            rel.source,  # 添加关系源
            rel.target,  # 添加关系目标
            rel.description if rel.description else "",  # 如果存在描述则添加，否则为空字符串
        ]
        
        # 如果需要包含关系权重信息，则添加到新上下文记录中
        if include_relationship_weight:
            new_context.append(str(rel.weight if rel.weight else ""))
        
        # 遍历每个属性列，添加到新上下文记录中
        for field in attribute_cols:
            field_value = (
                str(rel.attributes.get(field))  # 获取属性值并转换为字符串
                if rel.attributes and rel.attributes.get(field)  # 如果属性存在且有对应值
                else ""  # 否则为空字符串
            )
            new_context.append(field_value)
        
        # 将新上下文记录转换为以列分隔符连接的字符串，并在末尾添加换行符
        new_context_text = column_delimiter.join(new_context) + "\n"
        
        # 计算新上下文记录的token数量
        new_tokens = num_tokens(new_context_text, token_encoder)
        
        # 如果加入新记录后超过了最大token数，则结束添加过程
        if current_tokens + new_tokens > max_tokens:
            break
        
        # 更新当前上下文文本，加入新上下文记录的文本表示
        current_context_text += new_context_text
        
        # 将新上下文记录添加到所有上下文记录列表中
        all_context_records.append(new_context)
        
        # 更新当前token数
        current_tokens += new_tokens
    
    # 如果存在多于一个记录，则基于所有上下文记录创建DataFrame，排除第一行表头
    if len(all_context_records) > 1:
        record_df = pd.DataFrame(
            all_context_records[1:],  # 使用除表头外的所有记录创建DataFrame
            columns=cast(Any, all_context_records[0])  # 指定列名为表头
        )
    else:
        record_df = pd.DataFrame()  # 否则创建空的DataFrame
    
    # 返回当前上下文文本和创建的DataFrame
    return current_context_text, record_df
# 根据一组选定的实体和排名属性筛选和排序关系
def _filter_relationships(
    selected_entities: list[Entity],                    # 选定的实体列表
    relationships: list[Relationship],                  # 所有关系的列表
    top_k_relationships: int = 10,                      # 返回的关系数目的上限，默认为10
    relationship_ranking_attribute: str = "rank",       # 用于排名的属性，默认为"rank"
) -> list[Relationship]:                               # 返回一个关系对象的列表

    """Filter and sort relationships based on a set of selected entities and a ranking attribute."""
    
    # 第一优先级：内部网络关系（即选定实体之间的关系）
    in_network_relationships = get_in_network_relationships(
        selected_entities=selected_entities,            # 选定的实体
        relationships=relationships,                    # 所有关系的列表
        ranking_attribute=relationship_ranking_attribute,  # 用于排名的属性
    )

    # 第二优先级：外部网络关系
    # （即选定实体与未包含在选定实体内的其他实体之间的关系）
    out_network_relationships = get_out_network_relationships(
        selected_entities=selected_entities,            # 选定的实体
        relationships=relationships,                    # 所有关系的列表
        ranking_attribute=relationship_ranking_attribute,  # 用于排名的属性
    )

    # 如果外部网络关系数量少于等于1，则直接返回内部网络关系和外部网络关系的结合
    if len(out_network_relationships) <= 1:
        return in_network_relationships + out_network_relationships

    # 在外部网络关系中，优先考虑共享关系（即与多个选定实体共享的外部实体之间的关系）
    selected_entity_names = [entity.title for entity in selected_entities]  # 选定实体的标题列表
    out_network_source_names = [
        relationship.source
        for relationship in out_network_relationships
        if relationship.source not in selected_entity_names  # 源实体不在选定实体列表中
    ]
    out_network_target_names = [
        relationship.target
        for relationship in out_network_relationships
        if relationship.target not in selected_entity_names  # 目标实体不在选定实体列表中
    ]
    out_network_entity_names = list(
        set(out_network_source_names + out_network_target_names)  # 外部网络实体的名称列表（去重）
    )
    out_network_entity_links = defaultdict(int)  # 创建一个默认值为int类型的空字典
    for entity_name in out_network_entity_names:
        # 统计与每个外部网络实体相关的链接数目
        targets = [
            relationship.target
            for relationship in out_network_relationships
            if relationship.source == entity_name  # 外部网络实体作为源实体的关系目标列表
        ]
        sources = [
            relationship.source
            for relationship in out_network_relationships
            if relationship.target == entity_name  # 外部网络实体作为目标实体的关系源列表
        ]
        out_network_entity_links[entity_name] = len(set(targets + sources))  # 计算链接数目并存储

    # 根据链接数目和排名属性对外部网络关系进行排序
    for rel in out_network_relationships:
        if rel.attributes is None:
            rel.attributes = {}  # 如果关系属性为空，则初始化为一个空字典
        rel.attributes["links"] = (
            out_network_entity_links[rel.source]   # 如果源实体在链接数目字典中，则使用其链接数目
            if rel.source in out_network_entity_links
            else out_network_entity_links[rel.target]  # 否则使用目标实体的链接数目
        )

    # 首先按照属性中的"links"排序，然后再按照排名属性排序
    # 如果关系排序属性为 "weight"，则按照链接数和权重降序排列外部网络关系列表
    if relationship_ranking_attribute == "weight":
        out_network_relationships.sort(
            key=lambda x: (x.attributes["links"], x.weight),  # 按照链接数和对象权重排序
            reverse=True,  # 降序排列
        )
    else:
        # 如果关系排序属性不是 "weight"，则按照链接数和指定属性降序排列外部网络关系列表
        out_network_relationships.sort(
            key=lambda x: (
                x.attributes["links"],  # 按照链接数排序
                x.attributes[relationship_ranking_attribute],  # 按照指定的关系排序属性排序
            ),  # 二级排序条件
            reverse=True,  # 降序排列
        )

    # 计算关系预算，为每个选定实体乘以前 k 个关系的数量
    relationship_budget = top_k_relationships * len(selected_entities)
    # 返回内部网络关系加上按照排序后的外部网络关系列表的前 relationship_budget 个元素
    return in_network_relationships + out_network_relationships[:relationship_budget]
# 定义函数，准备用于系统提示的实体、关系和协变量数据表的上下文数据
def get_candidate_context(
    selected_entities: list[Entity],                # 选定的实体列表
    entities: list[Entity],                         # 所有实体列表
    relationships: list[Relationship],              # 所有关系列表
    covariates: dict[str, list[Covariate]],         # 协变量字典，键为字符串，值为协变量列表
    include_entity_rank: bool = True,               # 是否包含实体排名，默认为True
    entity_rank_description: str = "number of relationships",  # 实体排名描述，默认为"number of relationships"
    include_relationship_weight: bool = False,      # 是否包含关系权重，默认为False
) -> dict[str, pd.DataFrame]:                      # 返回一个字典，键为字符串，值为DataFrame对象

    """Prepare entity, relationship, and covariate data tables as context data for system prompt."""
    
    # 初始化候选上下文字典
    candidate_context = {}

    # 获取候选关系列表
    candidate_relationships = get_candidate_relationships(
        selected_entities=selected_entities,
        relationships=relationships,
    )
    # 将候选关系转换为关系DataFrame，并存储在候选上下文中的"relationships"键下
    candidate_context["relationships"] = to_relationship_dataframe(
        relationships=candidate_relationships,
        include_relationship_weight=include_relationship_weight,
    )

    # 根据候选关系获取候选实体列表
    candidate_entities = get_entities_from_relationships(
        relationships=candidate_relationships,      # 使用候选关系列表
        entities=entities                           # 所有实体列表
    )
    # 将候选实体转换为实体DataFrame，并存储在候选上下文中的"entities"键下
    candidate_context["entities"] = to_entity_dataframe(
        entities=candidate_entities,
        include_entity_rank=include_entity_rank,     # 是否包含实体排名
        rank_description=entity_rank_description    # 实体排名描述
    )

    # 遍历协变量字典中的每一个键（协变量类型）
    for covariate in covariates:
        # 获取候选协变量列表
        candidate_covariates = get_candidate_covariates(
            selected_entities=selected_entities,     # 选定的实体列表
            covariates=covariates[covariate],       # 指定类型的协变量列表
        )
        # 将候选协变量转换为协变量DataFrame，并存储在候选上下文中，键名为协变量类型的小写形式
        candidate_context[covariate.lower()] = to_covariate_dataframe(
            candidate_covariates
        )

    # 返回组装好的候选上下文数据字典
    return candidate_context
```