# `MetaGPT\metagpt\provider\general_api_base.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : refs to openai 0.x sdk

# 导入所需的模块
import asyncio
import json
import os
import platform
import re
import sys
import threading
import time
from contextlib import asynccontextmanager
from enum import Enum
from typing import (
    AsyncGenerator,
    AsyncIterator,
    Dict,
    Iterator,
    Optional,
    Tuple,
    Union,
    overload,
)
from urllib.parse import urlencode, urlsplit, urlunsplit

import aiohttp
import requests

# 根据 Python 版本导入不同的模块
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

# 导入日志模块
import logging

# 导入 openai 模块
import openai
from openai import version

# 设置日志记录器
logger = logging.getLogger("openai")

# 设置一些常量
TIMEOUT_SECS = 600
MAX_SESSION_LIFETIME_SECS = 180
MAX_CONNECTION_RETRIES = 2

# 每个线程都有一个 'session' 属性
_thread_context = threading.local()

# 从环境变量中获取日志级别
LLM_LOG = os.environ.get("LLM_LOG", "debug")

# 定义 API 类型的枚举
class ApiType(Enum):
    AZURE = 1
    OPEN_AI = 2
    AZURE_AD = 3

    @staticmethod
    def from_str(label):
        if label.lower() == "azure":
            return ApiType.AZURE
        elif label.lower() in ("azure_ad", "azuread"):
            return ApiType.AZURE_AD
        elif label.lower() in ("open_ai", "openai"):
            return ApiType.OPEN_AI
        else:
            raise openai.OpenAIError(
                "The API type provided in invalid. Please select one of the supported API types: 'azure', 'azure_ad', 'open_ai'"
            )

# 根据 API 类型和密钥生成请求头
api_key_to_header = (
    lambda api, key: {"Authorization": f"Bearer {key}"}
    if api in (ApiType.OPEN_AI, ApiType.AZURE_AD)
    else {"api-key": f"{key}"}
)

# 获取控制台日志级别
def _console_log_level():
    if LLM_LOG in ["debug", "info"]:
        return LLM_LOG
    else:
        return None

# 记录调试日志
def log_debug(message, **params):
    msg = logfmt(dict(message=message, **params))
    if _console_log_level() == "debug":
        print(msg, file=sys.stderr)
    logger.debug(msg)

# 记录信息日志
def log_info(message, **params):
    msg = logfmt(dict(message=message, **params))
    if _console_log_level() in ["debug", "info"]:
        print(msg, file=sys.stderr)
    logger.info(msg)

# 记录警告日志
def log_warn(message, **params):
    msg = logfmt(dict(message=message, **params))
    print(msg, file=sys.stderr)
    logger.warning(msg)

# 格式化日志信息
def logfmt(props):
    def fmt(key, val):
        if hasattr(val, "decode"):
            val = val.decode("utf-8")
        if not isinstance(val, str):
            val = str(val)
        if re.search(r"\s", val):
            val = repr(val)
        if re.search(r"\s", key):
            key = repr(key)
        return "{key}={val}".format(key=key, val=val)

    return " ".join([fmt(key, val) for key, val in sorted(props.items())])

# 定义 OpenAI 响应类
class OpenAIResponse:
    def __init__(self, data, headers):
        self._headers = headers
        self.data = data

    @property
    def request_id(self) -> Optional[str]:
        return self._headers.get("request-id")

    @property
    def retry_after(self) -> Optional[int]:
        try:
            return int(self._headers.get("retry-after"))
        except TypeError:
            return None

    @property
    def operation_location(self) -> Optional[str]:
        return self._headers.get("operation-location")

    @property
    def organization(self) -> Optional[str]:
        return self._headers.get("LLM-Organization")

    @property
    def response_ms(self) -> Optional[int]:
        h = self._headers.get("Openai-Processing-Ms")
        return None if h is None else round(float(h))

# 构建 API URL
def _build_api_url(url, query):
    scheme, netloc, path, base_query, fragment = urlsplit(url)

    if base_query:
        query = "%s&%s" % (base_query, query)

    return urlunsplit((scheme, netloc, path, query, fragment))

# 构建 requests 模块的代理参数
def _requests_proxies_arg(proxy) -> Optional[Dict[str, str]]:
    if proxy is None:
        return None
    elif isinstance(proxy, str):
        return {"http": proxy, "https": proxy}
    elif isinstance(proxy, dict):
        return proxy.copy()
    else:
        raise ValueError(
            "'openai.proxy' must be specified as either a string URL or a dict with string URL under the https and/or http keys."
        )

# 构建 aiohttp 模块的代理参数
def _aiohttp_proxies_arg(proxy) -> Optional[str]:
    if proxy is None:
        return None
    elif isinstance(proxy, str):
        return proxy
    elif isinstance(proxy, dict):
        return proxy["https"] if "https" in proxy else proxy["http"]
    else:
        raise ValueError(
            "'openai.proxy' must be specified as either a string URL or a dict with string URL under the https and/or http keys."
        )

# 创建 requests 模块的会话
def _make_session() -> requests.Session:
    s = requests.Session()
    s.mount(
        "https://",
        requests.adapters.HTTPAdapter(max_retries=MAX_CONNECTION_RETRIES),
    )
    return s

# 解析流的辅助函数
def parse_stream_helper(line: bytes) -> Optional[str]:
    if line:
        if line.strip() == b"data: [DONE]":
            return None
        if line.startswith(b"data: "):
            line = line[len(b"data: ") :]
            return line.decode("utf-8")
        else:
            return None
    return None

# 解析流
def parse_stream(rbody: Iterator[bytes]) -> Iterator[str]:
    for line in rbody:
        _line = parse_stream_helper(line)
        if _line is not None:
            yield _line

# 异步解析流
async def parse_stream_async(rbody: aiohttp.StreamReader):
    async for line in rbody:
        _line = parse_stream_helper(line)
        if _line is not None:
            yield _line

# 异步上下文管理器，用于 aiohttp 客户端会话
@asynccontextmanager
async def aiohttp_session() -> AsyncIterator[aiohttp.ClientSession]:
    async with aiohttp.ClientSession() as session:
        yield session

```