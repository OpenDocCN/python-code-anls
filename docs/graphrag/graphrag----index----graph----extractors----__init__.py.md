# `.\graphrag\graphrag\index\graph\extractors\__init__.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""The Indexing Engine graph extractors package root."""

# 导入索赔（claims）相关的常量和类
from .claims import CLAIM_EXTRACTION_PROMPT, ClaimExtractor
# 导入社区报告（community_reports）相关的常量和类
from .community_reports import (
    COMMUNITY_REPORT_PROMPT,
    CommunityReportsExtractor,
)
# 导入图提取结果类和图提取器类
from .graph import GraphExtractionResult, GraphExtractor

# 将所有公开的类和常量列在 __all__ 中，使它们在被 from package import * 导入时可见
__all__ = [
    "CLAIM_EXTRACTION_PROMPT",
    "COMMUNITY_REPORT_PROMPT",
    "ClaimExtractor",
    "CommunityReportsExtractor",
    "GraphExtractionResult",
    "GraphExtractor",
]
```