# `MetaGPT\metagpt\utils\__init__.py`

```

#!/usr/bin/env python
# 指定解释器为 Python，并且使用环境变量中的 python 解释器

# -*- coding: utf-8 -*-
# 指定文件编码格式为 UTF-8

"""
@Time    : 2023/4/29 15:50
@Author  : alexanderwu
@File    : __init__.py
"""
# 文件的注释信息，包括时间、作者和文件名

# 导入 read_docx 函数
from metagpt.utils.read_document import read_docx
# 导入 Singleton 类
from metagpt.utils.singleton import Singleton
# 导入 TOKEN_COSTS 常量
from metagpt.utils.token_counter import (
    TOKEN_COSTS,
    # 导入 count_message_tokens 函数
    count_message_tokens,
    # 导入 count_string_tokens 函数
    count_string_tokens,
)

# 模块中可导出的内容
__all__ = [
    "read_docx",
    "Singleton",
    "TOKEN_COSTS",
    "count_message_tokens",
    "count_string_tokens",
]

```