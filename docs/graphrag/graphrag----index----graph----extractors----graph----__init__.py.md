# `.\graphrag\graphrag\index\graph\extractors\graph\__init__.py`

```py
# 版权声明和许可证信息，说明代码版权归 Microsoft Corporation 所有，使用 MIT 许可证进行授权
# 这是索引引擎单部图包的根目录

# 导入 graph_extractor 模块中的特定内容：
# - DEFAULT_ENTITY_TYPES: 默认实体类型列表
# - GraphExtractionResult: 图提取结果类
# - GraphExtractor: 图提取器类
from .graph_extractor import (
    DEFAULT_ENTITY_TYPES,
    GraphExtractionResult,
    GraphExtractor,
)

# 导入 prompts 模块中的 GRAPH_EXTRACTION_PROMPT 常量
from .prompts import GRAPH_EXTRACTION_PROMPT

# 定义一个列表 __all__，包含需要在模块外部导出的名称列表
__all__ = [
    "DEFAULT_ENTITY_TYPES",      # 默认实体类型列表
    "GRAPH_EXTRACTION_PROMPT",   # 图提取提示常量
    "GraphExtractionResult",     # 图提取结果类
    "GraphExtractor",            # 图提取器类
]
```