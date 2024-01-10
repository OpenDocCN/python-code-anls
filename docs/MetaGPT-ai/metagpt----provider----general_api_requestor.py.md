# `MetaGPT\metagpt\provider\general_api_requestor.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : General Async API for http-based LLM model

# 引入必要的库
import asyncio
from typing import AsyncGenerator, Generator, Iterator, Tuple, Union
import aiohttp
import requests
from metagpt.logs import logger
from metagpt.provider.general_api_base import APIRequestor

# 定义一个辅助函数，用于解析流数据
def parse_stream_helper(line: bytes) -> Union[bytes, None]:
    if line and line.startswith(b"data:"):
        if line.startswith(b"data: "):
            # 如果 SSE 事件包含空格，则可能是有效的
            line = line[len(b"data: ") :]
        else:
            line = line[len(b"data:") :]
        if line.strip() == b"[DONE]":
            # 如果是 [DONE] 则返回 None，关闭 HTTP 连接
            return None
        else:
            return line
    return None

# 定义一个函数，用于解析流数据
def parse_stream(rbody: Iterator[bytes]) -> Iterator[bytes]:
    for line in rbody:
        _line = parse_stream_helper(line)
        if _line is not None:
            yield _line

# 定义一个类，用于处理通用 API 请求
class GeneralAPIRequestor(APIRequestor):
    """
    usage
        # full_url = "{base_url}{url}"
        requester = GeneralAPIRequestor(base_url=base_url)
        result, _, api_key = await requester.arequest(
            method=method,
            url=url,
            headers=headers,
            stream=stream,
            params=kwargs,
            request_timeout=120
        )
    """

    # 解释响应行的函数，返回原始数据
    def _interpret_response_line(self, rbody: bytes, rcode: int, rheaders, stream: bool) -> bytes:
        return rbody

    # 解释响应的函数，返回响应和是否为流的标志
    def _interpret_response(
        self, result: requests.Response, stream: bool
    ) -> Tuple[Union[bytes, Iterator[Generator]], bytes]:
        if stream and "text/event-stream" in result.headers.get("Content-Type", ""):
            return (
                self._interpret_response_line(line, result.status_code, result.headers, stream=True)
                for line in parse_stream(result.iter_lines())
            ), True
        else:
            return (
                self._interpret_response_line(
                    result.content,  # 让调用者解码消息
                    result.status_code,
                    result.headers,
                    stream=False,
                ),
                False,
            )

    # 解释异步响应的函数，返回响应和是否为流的标志
    async def _interpret_async_response(
        self, result: aiohttp.ClientResponse, stream: bool
    ) -> Tuple[Union[bytes, AsyncGenerator[bytes, None]], bool]:
        if stream and (
            "text/event-stream" in result.headers.get("Content-Type", "")
            or "application/x-ndjson" in result.headers.get("Content-Type", "")
        ):
            return (
                self._interpret_response_line(line, result.status, result.headers, stream=True)
                async for line in result.content
            ), True
        else:
            try:
                await result.read()
            except (aiohttp.ServerTimeoutError, asyncio.TimeoutError) as e:
                raise TimeoutError("Request timed out") from e
            except aiohttp.ClientError as exp:
                logger.warning(f"response: {result.content}, exp: {exp}")
            return (
                self._interpret_response_line(
                    await result.read(),  # 让调用者解码消息
                    result.status,
                    result.headers,
                    stream=False,
                ),
                False,
            )

```