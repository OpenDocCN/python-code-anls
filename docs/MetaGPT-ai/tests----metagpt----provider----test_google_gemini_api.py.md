# `MetaGPT\tests\metagpt\provider\test_google_gemini_api.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : the unittest of google gemini api

# 导入需要的模块
from abc import ABC
from dataclasses import dataclass

import pytest
from google.ai import generativelanguage as glm
from google.generativeai.types import content_types

from metagpt.config import CONFIG
from metagpt.provider.google_gemini_api import GeminiLLM

# 设置 gemini_api_key
CONFIG.gemini_api_key = "xx"

# 定义一个 MockGeminiResponse 类，用于模拟 Gemini 的响应
@dataclass
class MockGeminiResponse(ABC):
    text: str

# 定义一些测试用的数据
prompt_msg = "who are you"
messages = [{"role": "user", "parts": prompt_msg}]
resp_content = "I'm gemini from google"
default_resp = MockGeminiResponse(text=resp_content)

# 定义模拟的 Gemini 方法
def mock_gemini_count_tokens(self, contents: content_types.ContentsType) -> glm.CountTokensResponse:
    return glm.CountTokensResponse(total_tokens=20)

async def mock_gemini_count_tokens_async(self, contents: content_types.ContentsType) -> glm.CountTokensResponse:
    return glm.CountTokensResponse(total_tokens=20)

def mock_gemini_generate_content(self, **kwargs) -> MockGeminiResponse:
    return default_resp

async def mock_gemini_generate_content_async(self, stream: bool = False, **kwargs) -> MockGeminiResponse:
    if stream:
        # 如果是流式处理，则返回一个迭代器
        class Iterator(object):
            async def __aiter__(self):
                yield default_resp
        return Iterator()
    else:
        return default_resp

# 定义测试用例
@pytest.mark.asyncio
async def test_gemini_acompletion(mocker):
    # 使用 mocker.patch 方法替换原始方法，实现模拟
    mocker.patch("metagpt.provider.google_gemini_api.GeminiGenerativeModel.count_tokens", mock_gemini_count_tokens)
    mocker.patch(
        "metagpt.provider.google_gemini_api.GeminiGenerativeModel.count_tokens_async", mock_gemini_count_tokens_async
    )
    mocker.patch("google.generativeai.generative_models.GenerativeModel.generate_content", mock_gemini_generate_content)
    mocker.patch(
        "google.generativeai.generative_models.GenerativeModel.generate_content_async",
        mock_gemini_generate_content_async,
    )

    # 创建 GeminiLLM 实例
    gemini_gpt = GeminiLLM()

    # 测试 GeminiLLM 的一些方法
    assert gemini_gpt._user_msg(prompt_msg) == {"role": "user", "parts": [prompt_msg]}
    assert gemini_gpt._assistant_msg(prompt_msg) == {"role": "model", "parts": [prompt_msg]}

    # 测试获取 usage
    usage = gemini_gpt.get_usage(messages, resp_content)
    assert usage == {"prompt_tokens": 20, "completion_tokens": 20}

    # 测试生成 completion
    resp = gemini_gpt.completion(messages)
    assert resp == default_resp

    # 测试异步生成 completion
    resp = await gemini_gpt.acompletion(messages)
    assert resp.text == default_resp.text

    # 测试异步询问
    resp = await gemini_gpt.aask(prompt_msg, stream=False)
    assert resp == resp_content

    # 测试异步生成 completion 文本
    resp = await gemini_gpt.acompletion_text(messages, stream=False)
    assert resp == resp_content

    # 测试异步生成 completion 文本（流式处理）
    resp = await gemini_gpt.acompletion_text(messages, stream=True)
    assert resp == resp_content

    # 测试异步询问
    resp = await gemini_gpt.aask(prompt_msg)
    assert resp == resp_content

```