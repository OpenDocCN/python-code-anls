# `MetaGPT\metagpt\learn\__init__.py`

```py

#!/usr/bin/env python
# 指定解释器为 Python
# -*- coding: utf-8 -*-
# 指定文件编码格式为 UTF-8
"""
@Time    : 2023/4/30 20:57
@Author  : alexanderwu
@File    : __init__.py
"""
# 文件的注释信息，包括时间、作者、文件名等

# 从 metagpt.learn.text_to_image 模块中导入 text_to_image 函数
# 从 metagpt.learn.text_to_speech 模块中导入 text_to_speech 函数
# 从 metagpt.learn.google_search 模块中导入 google_search 函数
from metagpt.learn.text_to_image import text_to_image
from metagpt.learn.text_to_speech import text_to_speech
from metagpt.learn.google_search import google_search

# 模块的公开接口，包括 text_to_image、text_to_speech、google_search 函数
__all__ = ["text_to_image", "text_to_speech", "google_search"]

```