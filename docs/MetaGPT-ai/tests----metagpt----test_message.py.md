# `MetaGPT\tests\metagpt\test_message.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/16 10:57
@Author  : alexanderwu
@File    : test_message.py
@Modified By: mashenquan, 2023-11-1. Modify coding style.
"""
# 导入 pytest 模块
import pytest

# 从 metagpt.schema 模块中导入 AIMessage, Message, SystemMessage, UserMessage 类
from metagpt.schema import AIMessage, Message, SystemMessage, UserMessage

# 定义测试函数 test_message
def test_message():
    # 创建一个 Message 对象
    msg = Message(role="User", content="WTF")
    # 断言 Message 对象的 role 属性为 "User"
    assert msg.to_dict()["role"] == "User"
    # 断言 "User" 在 Message 对象的字符串表示中
    assert "User" in str(msg)

# 定义测试函数 test_all_messages
def test_all_messages():
    # 定义测试内容
    test_content = "test_message"
    # 创建不同类型的消息对象并放入列表中
    msgs = [
        UserMessage(test_content),
        SystemMessage(test_content),
        AIMessage(test_content),
        Message(content=test_content, role="QA"),
    ]
    # 遍历消息对象列表
    for msg in msgs:
        # 断言消息对象的内容属性与测试内容相等
        assert msg.content == test_content

# 如果当前脚本为主程序
if __name__ == "__main__":
    # 运行 pytest 测试，并输出详细信息
    pytest.main([__file__, "-s"])

```