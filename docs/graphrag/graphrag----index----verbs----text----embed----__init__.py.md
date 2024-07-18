# `.\graphrag\graphrag\index\verbs\text\embed\__init__.py`

```py
# 版权声明，指明此代码版权归 2024 年的 Microsoft 公司所有，采用 MIT 许可证授权

# 导入当前包的 text_embed 模块和相关的枚举类型 TextEmbedStrategyType
"""The Indexing Engine text embed package root."""

# 导入 text_embed 模块和 TextEmbedStrategyType 枚举类型，作为此包的公开接口
from .text_embed import TextEmbedStrategyType, text_embed

# 声明此模块公开的接口列表，包括 TextEmbedStrategyType 和 text_embed 两个符号
__all__ = ["TextEmbedStrategyType", "text_embed"]
```