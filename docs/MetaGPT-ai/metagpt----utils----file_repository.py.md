# `MetaGPT\metagpt\utils\file_repository.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/11/20
@Author  : mashenquan
@File    : git_repository.py
@Desc: File repository management. RFC 135 2.2.3.2, 2.2.3.4 and 2.2.3.13.
"""
# 导入未来的注解特性
from __future__ import annotations

# 导入所需的模块和库
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

# 异步文件操作库
import aiofiles

# 导入自定义的配置和日志模块
from metagpt.config import CONFIG
from metagpt.logs import logger

# 导入自定义的文档模块和通用工具模块
from metagpt.schema import Document
from metagpt.utils.common import aread
from metagpt.utils.json_to_markdown import json_to_markdown

```