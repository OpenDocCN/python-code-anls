# `MetaGPT\metagpt\tools\search_engine_serper.py`

```py

#!/usr/bin/env python
# 指定解释器为 python，使得脚本可以直接执行

# -*- coding: utf-8 -*-
# 指定文件编码格式为 UTF-8

"""
@Time    : 2023/5/23 18:27
@Author  : alexanderwu
@File    : search_engine_serpapi.py
"""
# 文件的注释信息，包括时间、作者和文件名

import json
# 导入 json 模块，用于处理 JSON 数据
from typing import Any, Dict, Optional, Tuple
# 导入 typing 模块，用于类型提示

import aiohttp
# 导入 aiohttp 模块，用于异步 HTTP 请求
from pydantic import BaseModel, ConfigDict, Field, field_validator
# 导入 pydantic 模块，用于数据验证和设置字段

from metagpt.config import CONFIG
# 从 metagpt.config 模块中导入 CONFIG 变量

if __name__ == "__main__":
    # 如果当前脚本作为主程序运行
    import fire
    # 导入 fire 模块，用于命令行接口

    fire.Fire(SerperWrapper().run)
    # 使用 fire 模块创建命令行接口，调用 SerperWrapper 类的 run 方法

```