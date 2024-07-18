# `.\graphrag\graphrag\index\verbs\overrides\__init__.py`

```py
# 版权声明和许可信息，指明代码版权属于 2024 年的 Microsoft Corporation，使用 MIT 许可证
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 导入模块中的函数和类
"""The Indexing Engine overrides package root."""
# 这个模块是索引引擎，用于覆盖包的根功能。

# 导入 aggregate 模块中的所有内容
from .aggregate import aggregate
# 导入 concat 模块中的所有内容
from .concat import concat
# 导入 merge 模块中的所有内容
from .merge import merge

# 声明该模块中可以被导出的符号列表，包括 aggregate, concat, merge
__all__ = ["aggregate", "concat", "merge"]
```