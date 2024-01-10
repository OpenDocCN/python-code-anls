# `MetaGPT\metagpt\tools\search_engine_serpapi.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/23 18:27
@Author  : alexanderwu
@File    : search_engine_serpapi.py
"""
# 导入所需的模块
from typing import Any, Dict, Optional, Tuple
import aiohttp
from pydantic import BaseModel, ConfigDict, Field, field_validator
# 从metagpt.config模块中导入CONFIG变量
from metagpt.config import CONFIG

# 如果当前脚本被直接执行，则执行以下代码
if __name__ == "__main__":
    # 导入fire模块
    import fire
    # 调用SerpAPIWrapper类的run方法
    fire.Fire(SerpAPIWrapper().run)

```