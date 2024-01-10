# `MetaGPT\tests\metagpt\memory\test_brain_memory.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/8/27
@Author  : mashenquan
@File    : test_brain_memory.py
"""

import pytest  # 导入 pytest 模块

from metagpt.config import LLMProviderEnum  # 从 metagpt.config 模块导入 LLMProviderEnum
from metagpt.llm import LLM  # 从 metagpt.llm 模块导入 LLM
from metagpt.memory.brain_memory import BrainMemory  # 从 metagpt.memory.brain_memory 模块导入 BrainMemory
from metagpt.schema import Message  # 从 metagpt.schema 模块导入 Message


@pytest.mark.asyncio  # 使用 pytest 的 asyncio 标记
async def test_memory():  # 定义名为 test_memory 的测试函数
    memory = BrainMemory()  # 创建 BrainMemory 实例
    memory.add_talk(Message(content="talk"))  # 向 memory 中添加一条对话消息
    assert memory.history[0].role == "user"  # 断言第一条消息的角色为 "user"
    memory.add_answer(Message(content="answer"))  # 向 memory 中添加一条回答消息
    assert memory.history[1].role == "assistant"  # 断言第二条消息的角色为 "assistant"
    redis_key = BrainMemory.to_redis_key("none", "user_id", "chat_id")  # 生成 redis key
    await memory.dumps(redis_key=redis_key)  # 将 memory 数据转储到 redis 中
    assert memory.exists("talk")  # 断言 "talk" 是否存在于 memory 中
    assert 1 == memory.to_int("1", 0)  # 断言将 "1" 转换为整数是否等于 1
    memory.last_talk = "AAA"  # 设置最后一条对话为 "AAA"
    assert memory.pop_last_talk() == "AAA"  # 断言弹出最后一条对话是否为 "AAA"
    assert memory.last_talk is None  # 断言最后一条对话是否为 None
    assert memory.is_history_available  # 断言历史记录是否可用
    assert memory.history_text  # 断言历史记录文本是否存在

    memory = await BrainMemory.loads(redis_key=redis_key)  # 从 redis 中加载 memory 数据
    assert memory  # 断言 memory 是否存在


@pytest.mark.parametrize(  # 使用 pytest 的参数化标记
    ("input", "tag", "val"),  # 参数名称
    [("[TALK]:Hello", "TALK", "Hello"), ("Hello", None, "Hello"), ("[TALK]Hello", None, "[TALK]Hello")],  # 参数取值
)
def test_extract_info(input, tag, val):  # 定义名为 test_extract_info 的测试函数
    t, v = BrainMemory.extract_info(input)  # 调用 extract_info 函数
    assert tag == t  # 断言标签是否相等
    assert val == v  # 断言值是否相等


@pytest.mark.asyncio  # 使用 pytest 的 asyncio 标记
@pytest.mark.parametrize("llm", [LLM(provider=LLMProviderEnum.OPENAI), LLM(provider=LLMProviderEnum.METAGPT)])  # 参数化测试
async def test_memory_llm(llm):  # 定义名为 test_memory_llm 的测试函数
    memory = BrainMemory()  # 创建 BrainMemory 实例
    for i in range(500):  # 循环 500 次
        memory.add_talk(Message(content="Lily is a girl.\n"))  # 向 memory 中添加对话消息

    res = await memory.is_related("apple", "moon", llm)  # 调用 is_related 方法
    assert not res  # 断言结果为 False

    res = await memory.rewrite(sentence="apple Lily eating", context="", llm=llm)  # 调用 rewrite 方法
    assert "Lily" in res  # 断言 "Lily" 是否在结果中

    res = await memory.summarize(llm=llm)  # 调用 summarize 方法
    assert res  # 断言结果存在

    res = await memory.get_title(llm=llm)  # 调用 get_title 方法
    assert res  # 断言结果存在
    assert "Lily" in res  # 断言 "Lily" 是否在结果中
    assert memory.history or memory.historical_summary  # 断言历史记录或历史摘要存在


if __name__ == "__main__":  # 如果当前模块被直接执行
    pytest.main([__file__, "-s"])  # 运行 pytest 测试并输出结果

```