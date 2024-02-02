# `MetaGPT\tests\metagpt\memory\test_memory.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : the unittest of Memory

# 导入需要的模块和类
from metagpt.actions import UserRequirement
from metagpt.memory.memory import Memory
from metagpt.schema import Message

# 定义测试函数
def test_memory():
    # 创建 Memory 对象
    memory = Memory()

    # 创建测试消息
    message1 = Message(content="test message1", role="user1")
    message2 = Message(content="test message2", role="user2")
    message3 = Message(content="test message3", role="user1")

    # 添加消息到 Memory 对象中，并进行断言检查
    memory.add(message1)
    assert memory.count() == 1

    # 删除最新的消息，并进行断言检查
    memory.delete_newest()
    assert memory.count() == 0

    # 批量添加消息到 Memory 对象中，并进行断言检查
    memory.add_batch([message1, message2])
    assert memory.count() == 2
    assert len(memory.index.get(message1.cause_by)) == 2

    # 根据角色获取消息，并进行断言检查
    messages = memory.get_by_role("user1")
    assert messages[0].content == message1.content

    # 根据内容获取消息，并进行断言检查
    messages = memory.get_by_content("test message")
    assert len(messages) == 2

    # 根据动作获取消息，并进行断言检查
    messages = memory.get_by_action(UserRequirement)
    assert len(messages) == 2

    # 根据动作列表获取消息，并进行断言检查
    messages = memory.get_by_actions([UserRequirement])
    assert len(messages) == 2

    # 尝试记住消息，并进行断言检查
    messages = memory.try_remember("test message")
    assert len(messages) == 2

    # 获取指定数量的消息，并进行断言检查
    messages = memory.get(k=1)
    assert len(messages) == 1

    messages = memory.get(k=5)
    assert len(messages) == 2

    # 查找新消息，并进行断言检查
    messages = memory.find_news([message3])
    assert len(messages) == 1

    # 删除指定消息，并进行断言检查
    memory.delete(message1)
    assert memory.count() == 1
    messages = memory.get_by_role("user2")
    assert messages[0].content == message2.content

    # 清空 Memory 对象，并进行断言检查
    memory.clear()
    assert memory.count() == 0
    assert len(memory.index) == 0

```