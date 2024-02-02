# `MetaGPT\metagpt\utils\mmdc_playwright.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/9/4 16:12
@Author  : Steven Lee
@File    : mmdc_playwright.py
"""

# 导入所需的模块
import os  # 导入操作系统模块
from urllib.parse import urljoin  # 导入用于处理 URL 的模块

from playwright.async_api import async_playwright  # 导入异步 Playwright 模块

from metagpt.logs import logger  # 从 metagpt.logs 模块中导入 logger 对象

```