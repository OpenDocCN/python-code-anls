# `MetaGPT\metagpt\actions\rebuild_class_view.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/19
@Author  : mashenquan
@File    : rebuild_class_view.py
@Desc    : Rebuild class view info
"""
# 导入所需的模块
import re
from pathlib import Path
import aiofiles  # 异步文件操作模块

# 导入自定义模块
from metagpt.actions import Action
from metagpt.config import CONFIG
from metagpt.const import (
    AGGREGATION,
    COMPOSITION,
    DATA_API_DESIGN_FILE_REPO,
    GENERALIZATION,
    GRAPH_REPO_FILE_REPO,
)
from metagpt.logs import logger  # 导入日志模块
from metagpt.repo_parser import RepoParser  # 导入仓库解析模块
from metagpt.schema import ClassAttribute, ClassMethod, ClassView  # 导入类属性、类方法、类视图模块
from metagpt.utils.common import split_namespace  # 导入命名空间分割函数
from metagpt.utils.di_graph_repository import DiGraphRepository  # 导入有向图仓库模块
from metagpt.utils.graph_repository import GraphKeyword, GraphRepository  # 导入图关键词、图仓库模块

```