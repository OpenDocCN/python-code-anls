# `MetaGPT\tests\metagpt\provider\test_anthropic_api.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : the unittest of Claude2
# 指定 Python 解释器和编码方式，以及描述测试的用途

import pytest
from anthropic.resources.completions import Completion
# 导入需要测试的模块和类

from metagpt.config import CONFIG
from metagpt.provider.anthropic_api import Claude2
# 导入配置和需要测试的类

CONFIG.anthropic_api_key = "xxx"
# 设置配置的 API 密钥

prompt = "who are you"
resp = "I'am Claude2"
# 设置测试用的提示和预期响应

def mock_anthropic_completions_create(self, model: str, prompt: str, max_tokens_to_sample: int) -> Completion:
    return Completion(id="xx", completion=resp, model="claude-2", stop_reason="stop_sequence", type="completion")
# 创建模拟的完成函数，返回预设的响应

async def mock_anthropic_acompletions_create(self, model: str, prompt: str, max_tokens_to_sample: int) -> Completion:
    return Completion(id="xx", completion=resp, model="claude-2", stop_reason="stop_sequence", type="completion")
# 创建模拟的异步完成函数，返回预设的响应

def test_claude2_ask(mocker):
    mocker.patch("anthropic.resources.completions.Completions.create", mock_anthropic_completions_create)
    assert resp == Claude2().ask(prompt)
# 测试同步的 ask 函数，使用模拟的完成函数进行测试

@pytest.mark.asyncio
async def test_claude2_aask(mocker):
    mocker.patch("anthropic.resources.completions.AsyncCompletions.create", mock_anthropic_acompletions_create)
    assert resp == await Claude2().aask(prompt)
# 测试异步的 aask 函数，使用模拟的异步完成函数进行测试

```