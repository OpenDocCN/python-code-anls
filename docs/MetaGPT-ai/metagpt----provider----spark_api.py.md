# `MetaGPT\metagpt\provider\spark_api.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : spark_api.py
"""
# 导入所需的模块
import _thread as thread  # 导入_thread模块并重命名为thread
import base64  # 导入base64模块
import datetime  # 导入datetime模块
import hashlib  # 导入hashlib模块
import hmac  # 导入hmac模块
import json  # 导入json模块
import ssl  # 导入ssl模块
from time import mktime  # 从time模块中导入mktime函数
from urllib.parse import urlencode, urlparse  # 从urllib.parse模块中导入urlencode和urlparse函数
from wsgiref.handlers import format_date_time  # 从wsgiref.handlers模块中导入format_date_time函数
import websocket  # 导入websocket模块

from metagpt.config import CONFIG, LLMProviderEnum  # 从metagpt.config模块中导入CONFIG和LLMProviderEnum
from metagpt.logs import logger  # 从metagpt.logs模块中导入logger
from metagpt.provider.base_llm import BaseLLM  # 从metagpt.provider.base_llm模块中导入BaseLLM
from metagpt.provider.llm_provider_registry import register_provider  # 从metagpt.provider.llm_provider_registry模块中导入register_provider

# 将当前类注册为SparkLLM类型的提供者
@register_provider(LLMProviderEnum.SPARK)
class SparkLLM(BaseLLM):
    def __init__(self):
        # 初始化方法，输出警告信息
        logger.warning("当前方法无法支持异步运行。当你使用acompletion时，并不能并行访问。")

    # 获取选择文本的方法，返回文本内容
    def get_choice_text(self, rsp: dict) -> str:
        return rsp["payload"]["choices"]["text"][-1]["content"]

    # 异步获取文本的方法，返回文本内容
    async def acompletion_text(self, messages: list[dict], stream=False, timeout: int = 3) -> str:
        # 输出错误信息，该功能被禁用
        logger.error("该功能禁用。")
        # 创建GetMessageFromWeb对象并运行
        w = GetMessageFromWeb(messages)
        return w.run()

    # 异步获取方法，返回结果
    async def acompletion(self, messages: list[dict], timeout=3):
        # 输出错误信息，不支持异步
        logger.error("该功能禁用。")
        # 创建GetMessageFromWeb对象并运行
        w = GetMessageFromWeb(messages)
        return w.run()

```