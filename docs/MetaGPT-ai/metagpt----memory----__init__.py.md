# `MetaGPT\metagpt\memory\__init__.py`

```

#!/usr/bin/env python
# 指定解释器为 Python
# -*- coding: utf-8 -*-
# 指定文件编码格式为 UTF-8
"""
@Time    : 2023/4/30 20:57
@Author  : alexanderwu
@File    : __init__.py
"""
# 导入 metagpt.memory.memory 模块中的 Memory 类
from metagpt.memory.memory import Memory

# 导入 metagpt.memory.longterm_memory 模块中的 LongTermMemory 类
# from metagpt.memory.longterm_memory import LongTermMemory

# 定义 __all__ 列表，包含需要导出的模块成员
__all__ = [
    "Memory",
    # "LongTermMemory",
]

```