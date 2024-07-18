# `.\graphrag\graphrag\index\verbs\graph\embed\__init__.py`

```py
# 版权声明，指出代码版权归 Microsoft Corporation 所有，并使用 MIT 许可证授权
# Licensed under the MIT License

# 导入 embed_graph 模块的 EmbedGraphStrategyType 类和 embed_graph 函数
"""The Indexing Engine graph embed package root."""
# 这是索引引擎图嵌入包的根目录说明

# 将 EmbedGraphStrategyType 类和 embed_graph 函数添加到模块的公开接口中
from .embed_graph import EmbedGraphStrategyType, embed_graph

# 声明该模块导出的所有公共接口，即 EmbedGraphStrategyType 和 embed_graph
__all__ = ["EmbedGraphStrategyType", "embed_graph"]
```