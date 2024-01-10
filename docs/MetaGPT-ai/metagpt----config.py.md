# `MetaGPT\metagpt\config.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Provide configuration, singleton
@Modified By: mashenquan, 2023/11/27.
        1. According to Section 2.2.3.11 of RFC 135, add git repository support.
        2. Add the parameter `src_workspace` for the old version project path.
"""
# 导入所需的模块
import datetime
import json
import os
import warnings
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

# 导入自定义模块
from metagpt.const import DEFAULT_WORKSPACE_ROOT, METAGPT_ROOT, OPTIONS
from metagpt.logs import logger
from metagpt.tools import SearchEngineType, WebBrowserEngineType
from metagpt.utils.common import require_python_version
from metagpt.utils.cost_manager import CostManager
from metagpt.utils.singleton import Singleton

# 自定义异常类，用于配置错误时抛出异常
class NotConfiguredException(Exception):
    """Exception raised for errors in the configuration.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message="The required configuration is not set"):
        self.message = message
        super().__init__(self.message)

# 枚举类，定义LLM提供者
class LLMProviderEnum(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    SPARK = "spark"
    ZHIPUAI = "zhipuai"
    FIREWORKS = "fireworks"
    OPEN_LLM = "open_llm"
    GEMINI = "gemini"
    METAGPT = "metagpt"
    AZURE_OPENAI = "azure_openai"
    OLLAMA = "ollama"

    def __missing__(self, key):
        return self.OPENAI

# 创建配置类的实例
CONFIG = Config()

```