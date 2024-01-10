# `MetaGPT\metagpt\actions\search_and_summarize.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/23 17:26
@Author  : alexanderwu
@File    : search_google.py
"""
# 导入必要的模块
from typing import Any, Optional
import pydantic
from pydantic import Field, model_validator
from metagpt.actions import Action
from metagpt.config import CONFIG, Config
from metagpt.logs import logger
from metagpt.schema import Message
from metagpt.tools import SearchEngineType
from metagpt.tools.search_engine import SearchEngine

# 定义常量
SEARCH_AND_SUMMARIZE_SYSTEM = """### Requirements
... (此处省略)
"""

SEARCH_AND_SUMMARIZE_SYSTEM_EN_US = SEARCH_AND_SUMMARIZE_SYSTEM.format(LANG="en-us")

SEARCH_AND_SUMMARIZE_PROMPT = """
... (此处省略)
"""

SEARCH_AND_SUMMARIZE_SALES_SYSTEM = """## Requirements
... (此处省略)
"""

SEARCH_AND_SUMMARIZE_SALES_PROMPT = """
... (此处省略)
"""

SEARCH_FOOD = """
# User Search Request
... (此处省略)
"""

# 定义一个类
class SearchAndSummarize(Action):
    name: str = ""
    content: Optional[str] = None
    config: None = Field(default_factory=Config)
    engine: Optional[SearchEngineType] = CONFIG.search_engine
    search_func: Optional[Any] = None
    search_engine: SearchEngine = None
    result: str = ""

    # 模型验证器
    @model_validator(mode="before")
    @classmethod
    def validate_engine_and_run_func(cls, values):
        ... (此处省略)

    # 异步运行方法
    async def run(self, context: list[Message], system_text=SEARCH_AND_SUMMARIZE_SYSTEM) -> str:
        ... (此处省略)


```