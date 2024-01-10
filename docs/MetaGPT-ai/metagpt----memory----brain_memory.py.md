# `MetaGPT\metagpt\memory\brain_memory.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/8/18
@Author  : mashenquan
@File    : brain_memory.py
@Desc    : Used by AgentStore. Used for long-term storage and automatic compression.
@Modified By: mashenquan, 2023/9/4. + redis memory cache.
@Modified By: mashenquan, 2023/12/25. Simplify Functionality.
"""
# 导入所需的模块
import json
import re
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from metagpt.config import CONFIG  # 导入配置信息
from metagpt.const import DEFAULT_LANGUAGE, DEFAULT_MAX_TOKENS, DEFAULT_TOKEN_SIZE  # 导入常量
from metagpt.logs import logger  # 导入日志模块
from metagpt.provider import MetaGPTLLM  # 导入MetaGPTLLM提供者
from metagpt.provider.base_llm import BaseLLM  # 导入BaseLLM提供者
from metagpt.schema import Message, SimpleMessage  # 导入消息模式
from metagpt.utils.redis import Redis  # 导入Redis模块

```