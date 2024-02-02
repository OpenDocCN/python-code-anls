# `MetaGPT\tests\metagpt\provider\test_general_api_base.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : 用于指定 Python 解释器和编码方式

import os  # 导入操作系统模块
from typing import AsyncGenerator, Generator, Iterator, Tuple, Union  # 导入类型提示模块

import aiohttp  # 异步 HTTP 客户端/服务器框架
import pytest  # 测试框架
import requests  # 发送 HTTP 请求
from openai import OpenAIError  # 导入 OpenAI 错误类

from metagpt.provider.general_api_base import (  # 导入自定义模块
    APIRequestor,  # API 请求类
    ApiType,  # API 类型
    OpenAIResponse,  # OpenAI 响应类
    _aiohttp_proxies_arg,  # 异步 HTTP 代理参数
    _build_api_url,  # 构建 API URL
    _make_session,  # 创建会话
    _requests_proxies_arg,  # HTTP 请求代理参数
    log_debug,  # 调试日志
    log_info,  # 信息日志
    log_warn,  # 警告日志
    logfmt,  # 格式化日志
    parse_stream,  # 解析流
    parse_stream_helper,  # 解析流辅助函数
)


def test_basic():
    _ = ApiType.from_str("azure")  # 测试从字符串创建 ApiType 枚举
    _ = ApiType.from_str("azuread")  # 测试从字符串创建 ApiType 枚举
    _ = ApiType.from_str("openai")  # 测试从字符串创建 ApiType 枚举
    with pytest.raises(OpenAIError):  # 测试是否抛出 OpenAIError 异常
        _ = ApiType.from_str("xx")  # 测试从无效字符串创建 ApiType 枚举

    os.environ.setdefault("LLM_LOG", "debug")  # 设置环境变量 LLM_LOG 为 debug
    log_debug("debug")  # 输出调试日志
    log_warn("warn")  # 输出警告日志
    log_info("info")  # 输出信息日志

    logfmt({"k1": b"v1", "k2": 1, "k3": "a b"})  # 格式化输出日志

    _build_api_url(url="http://www.baidu.com/s?wd=", query="baidu")  # 构建 API URL


def test_openai_response():
    resp = OpenAIResponse(data=[], headers={"retry-after": 3})  # 创建 OpenAI 响应对象
    assert resp.request_id is None  # 断言请求 ID 为 None
    assert resp.retry_after == 3  # 断言重试时间为 3
    assert resp.operation_location is None  # 断言操作位置为 None
    assert resp.organization is None  # 断言组织为 None
    assert resp.response_ms is None  # 断言响应时间为 None


def test_proxy():
    assert _requests_proxies_arg(proxy=None) is None  # 断言请求代理参数为 None

    proxy = "127.0.0.1:80"  # 设置代理地址
    assert _requests_proxies_arg(proxy=proxy) == {"http": proxy, "https": proxy}  # 断言请求代理参数为指定代理
    proxy_dict = {"http": proxy}  # 设置代理字典
    assert _requests_proxies_arg(proxy=proxy_dict) == proxy_dict  # 断言请求代理参数为指定代理字典
    assert _aiohttp_proxies_arg(proxy_dict) == proxy  # 断言异步 HTTP 代理参数为指定代理
    proxy_dict = {"https": proxy}  # 设置 HTTPS 代理字典
    assert _requests_proxies_arg(proxy=proxy_dict) == proxy_dict  # 断言请求代理参数为指定 HTTPS 代理字典
    assert _aiohttp_proxies_arg(proxy_dict) == proxy  # 断言异步 HTTP 代理参数为指定 HTTPS 代理

    assert _make_session() is not None  # 断言创建会话不为 None

    assert _aiohttp_proxies_arg(None) is None  # 断言异步 HTTP 代理参数为 None
    assert _aiohttp_proxies_arg("test") == "test"  # 断言异步 HTTP 代理参数为指定字符串
    with pytest.raises(ValueError):  # 断言是否抛出 ValueError 异常
        _aiohttp_proxies_arg(-1)  # 测试无效的异步 HTTP 代理参数


def test_parse_stream():
    assert parse_stream_helper(None) is None  # 断言解析流辅助函数为 None
    assert parse_stream_helper(b"data: [DONE]") is None  # 断言解析流辅助函数为 None
    assert parse_stream_helper(b"data: test") == "test"  # 断言解析流辅助函数为指定字符串
    assert parse_stream_helper(b"test") is None  # 断言解析流辅助函数为 None
    for line in parse_stream([b"data: test"]):  # 遍历解析流的每一行
        assert line == "test"  # 断言每一行为指定字符串


api_requestor = APIRequestor(base_url="http://www.baidu.com")  # 创建 API 请求对象


def mock_interpret_response(
    self, result: requests.Response, stream: bool
) -> Tuple[Union[bytes, Iterator[Generator]], bytes]:
    return b"baidu", False  # 模拟解释响应函数返回值


async def mock_interpret_async_response(
    self, result: aiohttp.ClientResponse, stream: bool
) -> Tuple[Union[OpenAIResponse, AsyncGenerator[OpenAIResponse, None]], bool]:
    return b"baidu", True  # 模拟解释异步响应函数返回值


def test_requestor_headers():
    # validate_headers
    headers = api_requestor._validate_headers(None)  # 验证请求头为 None
    assert not headers  # 断言请求头为空
    with pytest.raises(Exception):  # 断言是否抛出异常
        api_requestor._validate_headers(-1)  # 测试无效的请求头
    with pytest.raises(Exception):  # 断言是否抛出异常
        api_requestor._validate_headers({1: 2})  # 测试无效的请求头
    with pytest.raises(Exception):  # 断言是否抛出异常
        api_requestor._validate_headers({"test": 1})  # 测试无效的请求头
    supplied_headers = {"test": "test"}  # 设置指定请求头
    assert api_requestor._validate_headers(supplied_headers) == supplied_headers  # 断言验证后的请求头为指定请求头

    api_requestor.organization = "test"  # 设置组织
    api_requestor.api_version = "test123"  # 设置 API 版本
    api_requestor.api_type = ApiType.OPEN_AI  # 设置 API 类型
    request_id = "test123"  # 设置请求 ID
    headers = api_requestor.request_headers(method="post", extra={}, request_id=request_id)  # 获取请求头
    assert headers["LLM-Organization"] == api_requestor.organization  # 断言组织请求头为指定组织
    assert headers["LLM-Version"] == api_requestor.api_version  # 断言版本请求头为指定版本
    assert headers["X-Request-Id"] == request_id  # 断言请求 ID 请求头为指定请求 ID


def test_api_requestor(mocker):
    mocker.patch("metagpt.provider.general_api_base.APIRequestor._interpret_response", mock_interpret_response)  # 模拟解释响应函数
    resp, _, _ = api_requestor.request(method="get", url="/s?wd=baidu")  # 发送 GET 请求

    resp, _, _ = api_requestor.request(method="post", url="/s?wd=baidu")  # 发送 POST 请求


@pytest.mark.asyncio
async def test_async_api_requestor(mocker):
    mocker.patch(
        "metagpt.provider.general_api_base.APIRequestor._interpret_async_response", mock_interpret_async_response
    )  # 模拟解释异步响应函数
    resp, _, _ = await api_requestor.arequest(method="get", url="/s?wd=baidu")  # 发送异步 GET 请求
    resp, _, _ = await api_requestor.arequest(method="post", url="/s?wd=baidu")  # 发送异步 POST 请求

```