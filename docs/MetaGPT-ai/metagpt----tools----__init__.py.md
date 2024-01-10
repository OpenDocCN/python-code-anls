# `MetaGPT\metagpt\tools\__init__.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/4/29 15:35
@Author  : alexanderwu
@File    : __init__.py
"""

# 导入枚举类型
from enum import Enum

# 定义搜索引擎类型枚举
class SearchEngineType(Enum):
    SERPAPI_GOOGLE = "serpapi"
    SERPER_GOOGLE = "serper"
    DIRECT_GOOGLE = "google"
    DUCK_DUCK_GO = "ddg"
    CUSTOM_ENGINE = "custom"

# 定义网络浏览器引擎类型枚举
class WebBrowserEngineType(Enum):
    PLAYWRIGHT = "playwright"
    SELENIUM = "selenium"
    CUSTOM = "custom"

    # 定义缺失枚举类型的默认处理方法
    @classmethod
    def __missing__(cls, key):
        """Default type conversion"""
        return cls.CUSTOM

```