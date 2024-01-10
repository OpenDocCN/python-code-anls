# `MetaGPT\tests\metagpt\provider\zhipuai\test_async_sse_client.py`

```

#!/usr/bin/env python
# 指定解释器为 python

# -*- coding: utf-8 -*-
# 指定文件编码格式为 utf-8

# @Desc   :
# 代码描述

import pytest
# 导入 pytest 模块

from metagpt.provider.zhipuai.async_sse_client import AsyncSSEClient
# 从 metagpt.provider.zhipuai.async_sse_client 模块中导入 AsyncSSEClient 类

@pytest.mark.asyncio
# 使用 pytest 的 asyncio 标记
async def test_async_sse_client():
    # 定义测试函数 test_async_sse_client

    class Iterator(object):
        # 定义迭代器类
        async def __aiter__(self):
            # 异步迭代器方法
            yield b"data: test_value"
            # 生成测试数据

    async_sse_client = AsyncSSEClient(event_source=Iterator())
    # 创建 AsyncSSEClient 实例，传入迭代器作为事件源
    async for event in async_sse_client.async_events():
        # 异步遍历事件流
        assert event.data, "test_value"
        # 断言事件数据为 "test_value"

    class InvalidIterator(object):
        # 定义无效的迭代器类
        async def __aiter__(self):
            # 异步迭代器方法
            yield b"invalid: test_value"
            # 生成无效的测试数据

    async_sse_client = AsyncSSEClient(event_source=InvalidIterator())
    # 创建另一个 AsyncSSEClient 实例，传入无效的迭代器作为事件源
    async for event in async_sse_client.async_events():
        # 异步遍历事件流
        assert not event
        # 断言事件为空

```