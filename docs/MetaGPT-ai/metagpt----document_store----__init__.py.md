# `MetaGPT\metagpt\document_store\__init__.py`

```

#!/usr/bin/env python
# 指定解释器为 Python，并且使用环境变量中的第一个可用的 Python 解释器
# -*- coding: utf-8 -*-
# 指定文件编码格式为 UTF-8
"""
@Time    : 2023/5/25 10:20
@Author  : alexanderwu
@File    : __init__.py
"""
# 文件的注释信息，包括时间、作者和文件名

# 从 metagpt.document_store.faiss_store 模块中导入 FaissStore 类
from metagpt.document_store.faiss_store import FaissStore

# 导出 FaissStore 类，使其可以被其他模块导入和使用
__all__ = ["FaissStore"]

```