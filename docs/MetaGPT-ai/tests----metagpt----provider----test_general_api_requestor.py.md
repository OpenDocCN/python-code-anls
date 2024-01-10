# `MetaGPT\tests\metagpt\provider\test_general_api_requestor.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : the unittest of APIRequestor

# 导入 pytest 模块
import pytest

# 从 metagpt.provider.general_api_requestor 模块中导入 GeneralAPIRequestor、parse_stream、parse_stream_helper
from metagpt.provider.general_api_requestor import (
    GeneralAPIRequestor,
    parse_stream,
    parse_stream_helper,
)

# 创建 GeneralAPIRequestor 对象，指定 base_url 为 "http://www.baidu.com"
api_requestor = GeneralAPIRequestor(base_url="http://www.baidu.com")

# 测试 parse_stream_helper 函数
def test_parse_stream():
    # 断言 parse_stream_helper(None) 的返回值为 None
    assert parse_stream_helper(None) is None
    # 断言 parse_stream_helper(b"data: [DONE]") 的返回值为 None
    assert parse_stream_helper(b"data: [DONE]") is None
    # 断言 parse_stream_helper(b"data: test") 的返回值为 b"test"
    assert parse_stream_helper(b"data: test") == b"test"
    # 断言 parse_stream_helper(b"test") 的返回值为 None
    assert parse_stream_helper(b"test") is None
    # 遍历 parse_stream([b"data: test"]) 的返回值，断言每个元素为 b"test"
    for line in parse_stream([b"data: test"]):
        assert line == b"test"

# 测试 api_requestor.request 方法
def test_api_requestor():
    # 发送 GET 请求，获取响应数据、响应头和状态码
    resp, _, _ = api_requestor.request(method="get", url="/s?wd=baidu")
    # 断言 b"baidu" 在响应数据中
    assert b"baidu" in resp

# 异步测试 async_api_requestor 方法
@pytest.mark.asyncio
async def test_async_api_requestor():
    # 发送异步 GET 请求，获取响应数据、响应头和状态码
    resp, _, _ = await api_requestor.arequest(method="get", url="/s?wd=baidu")
    # 断言 b"baidu" 在响应数据中
    assert b"baidu" in resp

```