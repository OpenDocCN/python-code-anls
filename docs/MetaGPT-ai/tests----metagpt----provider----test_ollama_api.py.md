# `MetaGPT\tests\metagpt\provider\test_ollama_api.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : the unittest of ollama api

# 导入所需的模块
import json
from typing import Any, Tuple

import pytest

from metagpt.config import CONFIG
from metagpt.provider.ollama_api import OllamaLLM

# 设置测试用的消息和响应内容
prompt_msg = "who are you"
messages = [{"role": "user", "content": prompt_msg}]
resp_content = "I'm ollama"
default_resp = {"message": {"role": "assistant", "content": resp_content}}

# 设置 ollama_api_base 和 max_budget 的值
CONFIG.ollama_api_base = "http://xxx"
CONFIG.max_budget = 10

# 定义一个模拟的 ollama 请求函数
async def mock_ollama_arequest(self, stream: bool = False, **kwargs) -> Tuple[Any, Any, bool):
    if stream:
        # 如果是流式请求，返回模拟的迭代器
        class Iterator(object):
            events = [
                b'{"message": {"role": "assistant", "content": "I\'m ollama"}, "done": false}',
                b'{"prompt_eval_count": 20, "eval_count": 20, "done": true}',
            ]

            async def __aiter__(self):
                for event in self.events:
                    yield event

        return Iterator(), None, None
    else:
        # 如果不是流式请求，返回模拟的响应内容
        raw_default_resp = default_resp.copy()
        raw_default_resp.update({"prompt_eval_count": 20, "eval_count": 20})
        return json.dumps(raw_default_resp).encode(), None, None

# 定义测试函数
@pytest.mark.asyncio
async def test_gemini_acompletion(mocker):
    # 使用 mocker.patch 替换真实的请求函数为模拟的请求函数
    mocker.patch("metagpt.provider.general_api_requestor.GeneralAPIRequestor.arequest", mock_ollama_arequest)

    # 创建 OllamaLLM 实例
    ollama_gpt = OllamaLLM()

    # 测试 acompletion 方法
    resp = await ollama_gpt.acompletion(messages)
    assert resp["message"]["content"] == default_resp["message"]["content"]

    # 测试 aask 方法
    resp = await ollama_gpt.aask(prompt_msg, stream=False)
    assert resp == resp_content

    # 测试 acompletion_text 方法
    resp = await ollama_gpt.acompletion_text(messages, stream=False)
    assert resp == resp_content

    # 测试 acompletion_text 方法（流式请求）
    resp = await ollama_gpt.acompletion_text(messages, stream=True)
    assert resp == resp_content

    # 测试 aask 方法（不指定流式请求）
    resp = await ollama_gpt.aask(prompt_msg)
    assert resp == resp_content

```