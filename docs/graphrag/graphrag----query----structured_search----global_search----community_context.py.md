# `.\graphrag\graphrag\query\structured_search\global_search\community_context.py`

```py
# 版权声明和许可证信息
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 导入必要的模块和库
"""Contains algorithms to build context data for global search prompt."""

from typing import Any

import pandas as pd
import tiktoken

# 导入相关模块和类
from graphrag.model import CommunityReport, Entity
from graphrag.query.context_builder.community_context import (
    build_community_context,
)
from graphrag.query.context_builder.conversation_history import (
    ConversationHistory,
)
from graphrag.query.structured_search.base import GlobalContextBuilder


# 定义一个用于构建全局社区上下文的类，继承自GlobalContextBuilder
class GlobalCommunityContext(GlobalContextBuilder):
    """GlobalSearch community context builder."""

    # 初始化方法，接收多个参数来构建对象
    def __init__(
        self,
        community_reports: list[CommunityReport],  # 社区报告的列表
        entities: list[Entity] | None = None,  # 实体的列表（可选）
        token_encoder: tiktoken.Encoding | None = None,  # 令牌编码器（可选）
        random_state: int = 86,  # 随机状态，默认为86
    ):
        self.community_reports = community_reports  # 初始化社区报告列表
        self.entities = entities  # 初始化实体列表
        self.token_encoder = token_encoder  # 初始化令牌编码器
        self.random_state = random_state  # 初始化随机状态

    # 方法用于构建上下文数据
    def build_context(
        self,
        conversation_history: ConversationHistory | None = None,  # 对话历史（可选）
        use_community_summary: bool = True,  # 是否使用社区摘要，默认为True
        column_delimiter: str = "|",  # 列分隔符，默认为竖线"|"
        shuffle_data: bool = True,  # 是否对数据进行洗牌，默认为True
        include_community_rank: bool = False,  # 是否包含社区排名，默认为False
        min_community_rank: int = 0,  # 最小社区排名，默认为0
        community_rank_name: str = "rank",  # 社区排名的名称，默认为"rank"
        include_community_weight: bool = True,  # 是否包含社区权重，默认为True
        community_weight_name: str = "occurrence",  # 社区权重的名称，默认为"occurrence"
        normalize_community_weight: bool = True,  # 是否归一化社区权重，默认为True
        max_tokens: int = 8000,  # 最大令牌数，默认为8000
        context_name: str = "Reports",  # 上下文名称，默认为"Reports"
        conversation_history_user_turns_only: bool = True,  # 是否仅使用用户转换，默认为True
        conversation_history_max_turns: int | None = 5,  # 对话历史的最大轮数（可选，默认为5）
        **kwargs: Any,  # 其他参数
    ) -> tuple[str | list[str], dict[str, pd.DataFrame]]:
        """Prepare batches of community report data table as context data for global search."""
        # 初始化空字符串，用于存储对话历史上下文
        conversation_history_context = ""
        # 初始化空字典，用于存储最终的上下文数据
        final_context_data = {}

        # 如果存在对话历史数据
        if conversation_history:
            # 构建对话历史上下文
            (
                conversation_history_context,
                conversation_history_context_data,
            ) = conversation_history.build_context(
                include_user_turns_only=conversation_history_user_turns_only,
                max_qa_turns=conversation_history_max_turns,
                column_delimiter=column_delimiter,
                max_tokens=max_tokens,
                recency_bias=False,
            )
            # 如果构建的对话历史上下文不为空，则更新最终上下文数据
            if conversation_history_context != "":
                final_context_data = conversation_history_context_data

        # 构建社区上下文数据
        community_context, community_context_data = build_community_context(
            community_reports=self.community_reports,
            entities=self.entities,
            token_encoder=self.token_encoder,
            use_community_summary=use_community_summary,
            column_delimiter=column_delimiter,
            shuffle_data=shuffle_data,
            include_community_rank=include_community_rank,
            min_community_rank=min_community_rank,
            community_rank_name=community_rank_name,
            include_community_weight=include_community_weight,
            community_weight_name=community_weight_name,
            normalize_community_weight=normalize_community_weight,
            max_tokens=max_tokens,
            single_batch=False,
            context_name=context_name,
            random_state=self.random_state,
        )

        # 如果社区上下文是一个列表，则将每个上下文与对话历史上下文组合
        if isinstance(community_context, list):
            final_context = [
                f"{conversation_history_context}\n\n{context}"
                for context in community_context
            ]
        else:
            # 否则，直接将对话历史上下文与社区上下文结合
            final_context = f"{conversation_history_context}\n\n{community_context}"

        # 更新最终的上下文数据
        final_context_data.update(community_context_data)

        # 返回最终的上下文和上下文数据
        return (final_context, final_context_data)
```