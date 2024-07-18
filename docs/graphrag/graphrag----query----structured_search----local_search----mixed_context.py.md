# `.\graphrag\graphrag\query\structured_search\local_search\mixed_context.py`

```py
# 版权声明和许可证声明
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 导入日志模块
import logging
# 引入类型提示模块中的 Any 类型
from typing import Any

# 导入 pandas 库，用于数据处理
import pandas as pd
# 导入 tiktoken 模块，用于处理 token 编码
import tiktoken

# 从 graphrag.model 模块中导入多个类
from graphrag.model import (
    CommunityReport,
    Covariate,
    Entity,
    Relationship,
    TextUnit,
)
# 从 graphrag.query.context_builder.community_context 模块导入 build_community_context 函数
from graphrag.query.context_builder.community_context import (
    build_community_context,
)
# 从 graphrag.query.context_builder.conversation_history 模块导入 ConversationHistory 类
from graphrag.query.context_builder.conversation_history import (
    ConversationHistory,
)
# 从 graphrag.query.context_builder.entity_extraction 模块导入 EntityVectorStoreKey 类和 map_query_to_entities 函数
from graphrag.query.context_builder.entity_extraction import (
    EntityVectorStoreKey,
    map_query_to_entities,
)
# 从 graphrag.query.context_builder.local_context 模块导入多个函数
from graphrag.query.context_builder.local_context import (
    build_covariates_context,
    build_entity_context,
    build_relationship_context,
    get_candidate_context,
)
# 从 graphrag.query.context_builder.source_context 模块导入 build_text_unit_context 函数和 count_relationships 函数
from graphrag.query.context_builder.source_context import (
    build_text_unit_context,
    count_relationships,
)
# 从 graphrag.query.input.retrieval.community_reports 模块导入 get_candidate_communities 函数
from graphrag.query.input.retrieval.community_reports import (
    get_candidate_communities,
)
# 从 graphrag.query.input.retrieval.text_units 模块导入 get_candidate_text_units 函数
from graphrag.query.input.retrieval.text_units import get_candidate_text_units
# 从 graphrag.query.llm.base 模块导入 BaseTextEmbedding 类
from graphrag.query.llm.base import BaseTextEmbedding
# 从 graphrag.query.llm.text_utils 模块导入 num_tokens 函数
from graphrag.query.llm.text_utils import num_tokens
# 从 graphrag.query.structured_search.base 模块导入 LocalContextBuilder 类
from graphrag.query.structured_search.base import LocalContextBuilder
# 从 graphrag.vector_stores 模块导入 BaseVectorStore 类
from graphrag.vector_stores import BaseVectorStore

# 获取当前模块的日志记录器对象
log = logging.getLogger(__name__)

# 定义 LocalSearchMixedContext 类，继承自 LocalContextBuilder 类
class LocalSearchMixedContext(LocalContextBuilder):
    """Build data context for local search prompt combining community reports and entity/relationship/covariate tables."""

    # 初始化方法，接受多个参数来构建数据上下文
    def __init__(
        self,
        entities: list[Entity],  # 实体列表
        entity_text_embeddings: BaseVectorStore,  # 实体文本嵌入向量存储
        text_embedder: BaseTextEmbedding,  # 文本嵌入对象
        text_units: list[TextUnit] | None = None,  # 文本单元列表或空
        community_reports: list[CommunityReport] | None = None,  # 社区报告列表或空
        relationships: list[Relationship] | None = None,  # 关系列表或空
        covariates: dict[str, list[Covariate]] | None = None,  # 协变量字典或空
        token_encoder: tiktoken.Encoding | None = None,  # token 编码对象或空
        embedding_vectorstore_key: str = EntityVectorStoreKey.ID,  # 嵌入向量存储键名
    ):
        # 如果 community_reports 为空，则设置为一个空列表
        if community_reports is None:
            community_reports = []
        # 如果 relationships 为空，则设置为一个空列表
        if relationships is None:
            relationships = []
        # 如果 covariates 为空，则设置为一个空字典
        if covariates is None:
            covariates = {}
        # 如果 text_units 为空，则设置为一个空列表
        if text_units is None:
            text_units = []

        # 使用列表推导式创建实体字典，键为实体的 id，值为实体对象本身
        self.entities = {entity.id: entity for entity in entities}
        # 创建社区报告字典，键为报告的 id，值为报告对象本身
        self.community_reports = {
            community.id: community for community in community_reports
        }
        # 创建文本单元字典，键为单元的 id，值为单元对象本身
        self.text_units = {unit.id: unit for unit in text_units}
        # 创建关系字典，键为关系的 id，值为关系对象本身
        self.relationships = {
            relationship.id: relationship for relationship in relationships
        }
        # 设置协变量属性为传入的 covariates 参数
        self.covariates = covariates
        # 设置实体文本嵌入向量存储属性为传入的 entity_text_embeddings 参数
        self.entity_text_embeddings = entity_text_embeddings
        # 设置文本嵌入对象属性为传入的 text_embedder 参数
        self.text_embedder = text_embedder
        # 设置 token 编码对象属性为传入的 token_encoder 参数
        self.token_encoder = token_encoder
        # 设置嵌入向量存储键名属性为传入的 embedding_vectorstore_key 参数
        self.embedding_vectorstore_key = embedding_vectorstore_key
    # 根据实体键列表过滤实体文本嵌入
    def filter_by_entity_keys(self, entity_keys: list[int] | list[str]):
        """Filter entity text embeddings by entity keys."""
        self.entity_text_embeddings.filter_by_id(entity_keys)

    # 构建上下文信息
    def build_context(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
        include_entity_names: list[str] | None = None,
        exclude_entity_names: list[str] | None = None,
        conversation_history_max_turns: int | None = 5,
        conversation_history_user_turns_only: bool = True,
        max_tokens: int = 8000,
        text_unit_prop: float = 0.5,
        community_prop: float = 0.25,
        top_k_mapped_entities: int = 10,
        top_k_relationships: int = 10,
        include_community_rank: bool = False,
        include_entity_rank: bool = False,
        rank_description: str = "number of relationships",
        include_relationship_weight: bool = False,
        relationship_ranking_attribute: str = "rank",
        return_candidate_context: bool = False,
        use_community_summary: bool = False,
        min_community_rank: int = 0,
        community_context_name: str = "Reports",
        column_delimiter: str = "|",
        **kwargs: dict[str, Any],
    ):
        pass

    # 构建社区上下文信息
    def _build_community_context(
        self,
        selected_entities: list[Entity],
        max_tokens: int = 4000,
        use_community_summary: bool = False,
        column_delimiter: str = "|",
        include_community_rank: bool = False,
        min_community_rank: int = 0,
        return_candidate_context: bool = False,
        context_name: str = "Reports",
    ):
        pass

    # 构建文本单元上下文信息
    def _build_text_unit_context(
        self,
        selected_entities: list[Entity],
        max_tokens: int = 8000,
        return_candidate_context: bool = False,
        column_delimiter: str = "|",
        context_name: str = "Sources",
    ):
        pass

    # 构建本地上下文信息
    def _build_local_context(
        self,
        selected_entities: list[Entity],
        max_tokens: int = 8000,
        include_entity_rank: bool = False,
        rank_description: str = "relationship count",
        include_relationship_weight: bool = False,
        top_k_relationships: int = 10,
        relationship_ranking_attribute: str = "rank",
        return_candidate_context: bool = False,
        column_delimiter: str = "|",
    ):
        pass
```