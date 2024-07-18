# `.\graphrag\graphrag\index\graph\extractors\community_reports\__init__.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""The Indexing Engine community reports package root."""

# 导入图形提取器社区报告模式的架构
import graphrag.index.graph.extractors.community_reports.schemas as schemas

# 导入以下模块和函数
from .build_mixed_context import build_mixed_context
from .community_reports_extractor import CommunityReportsExtractor
from .prep_community_report_context import prep_community_report_context
from .prompts import COMMUNITY_REPORT_PROMPT
from .sort_context import sort_context
from .utils import (
    filter_claims_to_nodes,
    filter_edges_to_nodes,
    filter_nodes_to_level,
    get_levels,
    set_context_exceeds_flag,
    set_context_size,
)

# __all__ 列表，指定了在使用 from module import * 时导入的符号
__all__ = [
    "COMMUNITY_REPORT_PROMPT",         # 导出社区报告提示
    "CommunityReportsExtractor",       # 导出社区报告提取器类
    "build_mixed_context",             # 导出混合上下文构建函数
    "filter_claims_to_nodes",          # 导出筛选声明到节点的函数
    "filter_edges_to_nodes",           # 导出筛选边到节点的函数
    "filter_nodes_to_level",           # 导出筛选节点到级别的函数
    "get_levels",                      # 导出获取级别的函数
    "prep_community_report_context",   # 导出准备社区报告上下文的函数
    "schemas",                         # 导出社区报告模式的架构
    "set_context_exceeds_flag",        # 导出设置上下文超出标志的函数
    "set_context_size",                # 导出设置上下文大小的函数
    "sort_context",                    # 导出排序上下文的函数
]
```