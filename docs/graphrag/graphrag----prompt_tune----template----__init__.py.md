# `.\graphrag\graphrag\prompt_tune\template\__init__.py`

```py
# 导入与实体提取、实体总结和社区报告总结相关的文本提示。
# 从相应模块导入社区报告总结的提示内容
from .community_report_summarization import COMMUNITY_REPORT_SUMMARIZATION_PROMPT
# 从实体提取模块导入示例提取模板和未分类实体提取模板
from .entity_extraction import (
    EXAMPLE_EXTRACTION_TEMPLATE,
    UNTYPED_EXAMPLE_EXTRACTION_TEMPLATE,
)
# 从实体提取模块导入图提取的 JSON 和普通文本提示
from .entity_extraction import (
    GRAPH_EXTRACTION_JSON_PROMPT,
    GRAPH_EXTRACTION_PROMPT,
    UNTYPED_GRAPH_EXTRACTION_PROMPT,
)
# 从实体总结模块导入实体总结的提示内容
from .entity_summarization import ENTITY_SUMMARIZATION_PROMPT

# __all__ 是一个列表，包含了当前模块中可以导出的公共接口
__all__ = [
    "COMMUNITY_REPORT_SUMMARIZATION_PROMPT",  # 社区报告总结的提示
    "ENTITY_SUMMARIZATION_PROMPT",  # 实体总结的提示
    "EXAMPLE_EXTRACTION_TEMPLATE",  # 示例提取的模板
    "GRAPH_EXTRACTION_JSON_PROMPT",  # 图提取的 JSON 提示
    "GRAPH_EXTRACTION_PROMPT",  # 图提取的普通提示
    "UNTYPED_EXAMPLE_EXTRACTION_TEMPLATE",  # 未分类示例提取的模板
    "UNTYPED_GRAPH_EXTRACTION_PROMPT",  # 未分类图提取的提示
]
```