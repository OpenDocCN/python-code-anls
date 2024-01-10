# `MetaGPT\tests\metagpt\provider\test_zhipuai_api.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : the unittest of ZhiPuAILLM

# 导入所需的模块
import pytest
from zhipuai.utils.sse_client import Event

from metagpt.config import CONFIG
from metagpt.provider.zhipuai_api import ZhiPuAILLM

# 设置 zhipuai_api_key
CONFIG.zhipuai_api_key = "xxx.xxx"

# 设置测试用的消息和响应内容
prompt_msg = "who are you"
messages = [{"role": "user", "content": prompt_msg}]

resp_content = "I'm chatglm-turbo"
default_resp = {
    "code": 200,
    "data": {
        "choices": [{"role": "assistant", "content": resp_content}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 20},
    },
}

# 定义模拟的 zhipuai_invoke 函数
def mock_zhipuai_invoke(**kwargs) -> dict:
    return default_resp

# 定义模拟的 zhipuai_ainvoke 函数
async def mock_zhipuai_ainvoke(**kwargs) -> dict:
    return default_resp

# 定义模拟的 zhipuai_asse_invoke 函数
async def mock_zhipuai_asse_invoke(**kwargs):
    class MockResponse(object):
        async def _aread(self):
            class Iterator(object):
                events = [
                    Event(id="xxx", event="add", data=resp_content, retry=0),
                    Event(
                        id="xxx",
                        event="finish",
                        data="",
                        meta='{"usage": {"completion_tokens": 20,"prompt_tokens": 20}}',
                    ),
                ]

                async def __aiter__(self):
                    for event in self.events:
                        yield event

            async for chunk in Iterator():
                yield chunk

        async def async_events(self):
            async for chunk in self._aread():
                yield chunk

    return MockResponse()

# 定义测试用例
@pytest.mark.asyncio
async def test_zhipuai_acompletion(mocker):
    # 使用 mocker.patch 替换原有的函数调用，使用模拟函数
    mocker.patch("metagpt.provider.zhipuai.zhipu_model_api.ZhiPuModelAPI.invoke", mock_zhipuai_invoke)
    mocker.patch("metagpt.provider.zhipuai.zhipu_model_api.ZhiPuModelAPI.ainvoke", mock_zhipuai_ainvoke)
    mocker.patch("metagpt.provider.zhipuai.zhipu_model_api.ZhiPuModelAPI.asse_invoke", mock_zhipuai_asse_invoke)

    # 创建 ZhiPuAILLM 实例
    zhipu_gpt = ZhiPuAILLM()

    # 测试 acompletion 方法
    resp = await zhipu_gpt.acompletion(messages)
    assert resp["data"]["choices"][0]["content"] == resp_content

    # 测试 aask 方法
    resp = await zhipu_gpt.aask(prompt_msg, stream=False)
    assert resp == resp_content

    # 测试 acompletion_text 方法
    resp = await zhipu_gpt.acompletion_text(messages, stream=False)
    assert resp == resp_content

    # 测试 acompletion_text 方法
    resp = await zhipu_gpt.acompletion_text(messages, stream=True)
    assert resp == resp_content

    # 测试 aask 方法
    resp = await zhipu_gpt.aask(prompt_msg)
    assert resp == resp_content

# 定义测试 zhipuai_proxy 方法
def test_zhipuai_proxy():
    # 创建 ZhiPuAILLM 实例
    _ = ZhiPuAILLM()
    # 断言 openai.proxy 等于 CONFIG.openai_proxy
    assert openai.proxy == CONFIG.openai_proxy

```