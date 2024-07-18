# `.\graphrag\graphrag\index\verbs\entities\__init__.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
# 版权声明：版权归 Microsoft 公司所有，遵循 MIT 许可证

"""The Indexing Engine entities package root."""
# 模块说明：索引引擎实体包的根目录

# 从 extraction 模块导入 entity_extract 函数
from .extraction import entity_extract
# 从 summarize 模块导入 summarize_descriptions 函数
from .summarize import summarize_descriptions

# 将 entity_extract 和 summarize_descriptions 函数添加到 __all__ 列表中，表示它们是这个模块的公开接口
__all__ = ["entity_extract", "summarize_descriptions"]
```