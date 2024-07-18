# `.\graphrag\graphrag\query\context_builder\entity_extraction.py`

```py
# 版权声明和许可证信息
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Orchestration Context Builders."""

# 导入枚举类型
from enum import Enum

# 导入实体和关系模型
from graphrag.model import Entity, Relationship
# 导入实体检索相关功能
from graphrag.query.input.retrieval.entities import (
    get_entity_by_key,
    get_entity_by_name,
)
# 导入基础文本嵌入和基础向量存储
from graphrag.query.llm.base import BaseTextEmbedding
from graphrag.vector_stores import BaseVectorStore


class EntityVectorStoreKey(str, Enum):
    """Keys used as ids in the entity embedding vectorstores."""

    ID = "id"
    TITLE = "title"

    @staticmethod
    def from_string(value: str) -> "EntityVectorStoreKey":
        """Convert string to EntityVectorStoreKey."""
        # 根据字符串值转换为对应的 EntityVectorStoreKey 枚举
        if value == "id":
            return EntityVectorStoreKey.ID
        if value == "title":
            return EntityVectorStoreKey.TITLE

        # 如果值无效，抛出异常
        msg = f"Invalid EntityVectorStoreKey: {value}"
        raise ValueError(msg)


def map_query_to_entities(
    query: str,
    text_embedding_vectorstore: BaseVectorStore,
    text_embedder: BaseTextEmbedding,
    all_entities: list[Entity],
    embedding_vectorstore_key: str = EntityVectorStoreKey.ID,
    include_entity_names: list[str] | None = None,
    exclude_entity_names: list[str] | None = None,
    k: int = 10,
    oversample_scaler: int = 2,
) -> list[Entity]:
    """Extract entities that match a given query using semantic similarity of text embeddings of query and entity descriptions."""
    # 如果未提供包含的实体名列表，则设为空列表
    if include_entity_names is None:
        include_entity_names = []
    # 如果未提供排除的实体名列表，则设为空列表
    if exclude_entity_names is None:
        exclude_entity_names = []
    # 初始化匹配到的实体列表
    matched_entities = []
    # 如果查询字符串不为空
    if query != "":
        # 使用文本嵌入向量存储进行语义相似度搜索
        # 超采样以考虑排除的实体
        search_results = text_embedding_vectorstore.similarity_search_by_text(
            text=query,
            text_embedder=lambda t: text_embedder.embed(t),
            k=k * oversample_scaler,
        )
        # 遍历搜索结果
        for result in search_results:
            # 根据实体的键值从所有实体中获取匹配的实体
            matched = get_entity_by_key(
                entities=all_entities,
                key=embedding_vectorstore_key,
                value=result.document.id,
            )
            # 如果找到匹配的实体，则添加到匹配实体列表中
            if matched:
                matched_entities.append(matched)
    else:
        # 如果查询字符串为空，则按排名对所有实体进行排序
        all_entities.sort(key=lambda x: x.rank if x.rank else 0, reverse=True)
        # 选取排名靠前的 k 个实体作为匹配实体
        matched_entities = all_entities[:k]

    # 过滤掉在排除实体名列表中的实体
    if exclude_entity_names:
        matched_entities = [
            entity
            for entity in matched_entities
            if entity.title not in exclude_entity_names
        ]

    # 添加在包含实体名列表中的实体
    included_entities = []
    for entity_name in include_entity_names:
        included_entities.extend(get_entity_by_name(all_entities, entity_name))
    # 返回包含了包含实体和匹配实体的列表
    return included_entities + matched_entities


def find_nearest_neighbors_by_graph_embeddings(
    entity_id: str,
    graph_embedding_vectorstore: BaseVectorStore,
    all_entities: list[Entity],
    ...
):
    # 这里应该继续完成函数定义和注释
    # exclude_entity_names 是一个可选参数，用于指定要排除的实体名称列表，类型为 list[str] 或者 None
    # embedding_vectorstore_key 是一个参数，用于指定嵌入向量存储的键，默认为 EntityVectorStoreKey.ID
    # k 是一个参数，用于指定要返回的最近邻实体的数量，默认为 10
    # oversample_scaler 是一个参数，用于指定过采样的倍数，默认为 2
# 根据实体名称查找与其相关的最近邻实体，按照实体排名排序返回列表
def find_nearest_neighbors_by_entity_rank(
    entity_name: str,
    all_entities: list[Entity],
    all_relationships: list[Relationship],
    exclude_entity_names: list[str] | None = None,
    k: int | None = 10,
) -> list[Entity]:
    # 如果未指定要排除的实体名称列表，将其设为空列表
    if exclude_entity_names is None:
        exclude_entity_names = []

    # 获取与目标实体直接关联的所有关系
    entity_relationships = [
        rel
        for rel in all_relationships
        if rel.source == entity_name or rel.target == entity_name
    ]

    # 提取所有与目标实体直接关联的源实体和目标实体名称
    source_entity_names = {rel.source for rel in entity_relationships}
    target_entity_names = {rel.target for rel in entity_relationships}

    # 合并源实体和目标实体名称集合，去除需要排除的实体名称
    related_entity_names = (source_entity_names.union(target_entity_names)).difference(
        set(exclude_entity_names)
    )

    # 获取所有与相关实体名称匹配的实体对象
    top_relations = [
        entity for entity in all_entities if entity.title in related_entity_names
    ]

    # 根据实体的排名（如果存在）降序排序匹配的实体对象
    top_relations.sort(key=lambda x: x.rank if x.rank else 0, reverse=True)

    # 如果指定了返回的实体数量 k，则返回前 k 个实体；否则返回所有匹配的实体
    if k:
        return top_relations[:k]
    return top_relations
```