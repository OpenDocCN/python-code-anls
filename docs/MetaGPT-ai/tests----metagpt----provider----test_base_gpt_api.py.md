# `MetaGPT\tests\metagpt\provider\test_base_gpt_api.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/7 17:40
@Author  : alexanderwu
@File    : test_base_llm.py
"""

import pytest

from metagpt.provider.base_llm import BaseLLM  # 导入BaseLLM类
from metagpt.schema import Message  # 导入Message类

default_chat_resp = {  # 定义默认的聊天响应
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "I'am GPT",
            },
            "finish_reason": "stop",
        }
    ]
}
prompt_msg = "who are you"  # 定义提示消息
resp_content = default_chat_resp["choices"][0]["message"]["content"]  # 获取默认聊天响应的内容

class MockBaseLLM(BaseLLM):  # 定义MockBaseLLM类，继承自BaseLLM类
    def completion(self, messages: list[dict], timeout=3):  # 定义completion方法
        return default_chat_resp  # 返回默认聊天响应

    async def acompletion(self, messages: list[dict], timeout=3):  # 定义acompletion方法
        return default_chat_resp  # 返回默认聊天响应

    async def acompletion_text(self, messages: list[dict], stream=False, timeout=3) -> str:  # 定义acompletion_text方法
        return resp_content  # 返回默认聊天响应的内容

    async def close(self):  # 定义close方法
        return default_chat_resp  # 返回默认聊天响应

def test_base_llm():  # 定义测试方法test_base_llm
    message = Message(role="user", content="hello")  # 创建Message对象
    assert "role" in message.to_dict()  # 断言role属性在消息对象的字典表示中
    assert "user" in str(message)  # 断言"user"在消息对象的字符串表示中

    base_llm = MockBaseLLM()  # 创建MockBaseLLM对象

    openai_funccall_resp = {  # 定义OpenAI函数调用的响应
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "test",
                    "tool_calls": [
                        {
                            "id": "call_Y5r6Ddr2Qc2ZrqgfwzPX5l72",
                            "type": "function",
                            "function": {
                                "name": "execute",
                                "arguments": '{\n  "language": "python",\n  "code": "print(\'Hello, World!\')"\n}',
                            },
                        }
                    ],
                },
                "finish_reason": "stop",
            }
        ]
    }
    func: dict = base_llm.get_choice_function(openai_funccall_resp)  # 获取函数调用信息
    assert func == {
        "name": "execute",
        "arguments": '{\n  "language": "python",\n  "code": "print(\'Hello, World!\')"\n}',
    }

    func_args: dict = base_llm.get_choice_function_arguments(openai_funccall_resp)  # 获取函数调用的参数
    assert func_args == {"language": "python", "code": "print('Hello, World!')"}

    choice_text = base_llm.get_choice_text(openai_funccall_resp)  # 获取选择的文本
    assert choice_text == openai_funccall_resp["choices"][0]["message"]["content"]

    # resp = base_llm.ask(prompt_msg)
    # assert resp == resp_content

    # resp = base_llm.ask_batch([prompt_msg])
    # assert resp == resp_content

    # resp = base_llm.ask_code([prompt_msg])
    # assert resp == resp_content

@pytest.mark.asyncio
async def test_async_base_llm():  # 定义异步测试方法test_async_base_llm
    base_llm = MockBaseLLM()  # 创建MockBaseLLM对象

    resp = await base_llm.aask(prompt_msg)  # 调用aask方法
    assert resp == resp_content  # 断言响应内容与预期一致

    resp = await base_llm.aask_batch([prompt_msg])  # 调用aask_batch方法
    assert resp == resp_content  # 断言响应内容与预期一致

    resp = await base_llm.aask_code([prompt_msg])  # 调用aask_code方法
    assert resp == resp_content  # 断言响应内容与预期一致

```