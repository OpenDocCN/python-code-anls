# `MetaGPT\metagpt\utils\mmdc_pyppeteer.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/9/4 16:12
@Author  : alitrack
@File    : mmdc_pyppeteer.py
"""
# 导入所需的模块
import os
from urllib.parse import urljoin
# 从 pyppeteer 模块中导入 launch 函数
from pyppeteer import launch
# 从 metagpt.config 模块中导入 CONFIG 变量
from metagpt.config import CONFIG
# 从 metagpt.logs 模块中导入 logger 变量
from metagpt.logs import logger

```