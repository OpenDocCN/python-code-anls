# `MetaGPT\tests\metagpt\test_llm.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 14:45
@Author  : alexanderwu
@File    : test_llm.py
@Modified By: mashenquan, 2023/8/20. Remove global configuration `CONFIG`, enable configuration support for business isolation.
"""

# 导入 pytest 模块
import pytest

# 导入 LLM 类
from metagpt.provider.openai_api import OpenAILLM as LLM

# 定义一个 pytest fixture，返回一个 LLM 实例
@pytest.fixture()
def llm():
    return LLM()

# 定义一个异步测试函数，测试 llm.aask 方法
@pytest.mark.asyncio
async def test_llm_aask(llm):
    # 调用 llm.aask 方法，传入参数 "hello world"，stream=False
    rsp = await llm.aask("hello world", stream=False)
    # 断言返回结果的长度大于 0
    assert len(rsp) > 0

# 定义一个异步测试函数，测试 llm.acompletion 方法
@pytest.mark.asyncio
async def test_llm_acompletion(llm):
    # 定义一个消息列表
    hello_msg = [{"role": "user", "content": "hello"}]
    # 调用 llm.acompletion 方法，传入参数 hello_msg
    rsp = await llm.acompletion(hello_msg)
    # 断言返回结果的第一个选择的消息内容长度大于 0
    assert len(rsp.choices[0].message.content) > 0

# 如果当前文件被直接执行，则运行 pytest 测试
if __name__ == "__main__":
    pytest.main([__file__, "-s"])

```