# `.\graphrag\graphrag\index\graph\utils\__init__.py`

```py
# 声明代码文件的版权和许可信息
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 引入当前包中的模块以及函数
"""The Indexing Engine graph utils package root."""
from .normalize_node_names import normalize_node_names
from .stable_lcc import stable_largest_connected_component

# 定义所有公开的模块和函数的列表，用于在使用 `from package import *` 时指定可导入的内容
__all__ = ["normalize_node_names", "stable_largest_connected_component"]
```