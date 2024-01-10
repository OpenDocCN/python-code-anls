# `MetaGPT\metagpt\provider\zhipuai_api.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : zhipuai LLM from https://open.bigmodel.cn/dev/api#sdk
# 导入所需的模块
import json
from enum import Enum
import openai
import zhipuai
from requests import ConnectionError
from tenacity import (
    after_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from metagpt.config import CONFIG, LLMProviderEnum
from metagpt.logs import log_llm_stream, logger
from metagpt.provider.base_llm import BaseLLM
from metagpt.provider.llm_provider_registry import register_provider
from metagpt.provider.openai_api import log_and_reraise
from metagpt.provider.zhipuai.zhipu_model_api import ZhiPuModelAPI

# 定义 ZhiPuEvent 枚举类型，包含 ADD、ERROR、INTERRUPTED、FINISH 四个事件
class ZhiPuEvent(Enum):
    ADD = "add"
    ERROR = "error"
    INTERRUPTED = "interrupted"
    FINISH = "finish"

# 注册 ZHIPUAI 为 LLMProviderEnum 的提供者
@register_provider(LLMProviderEnum.ZHIPUAI)

```