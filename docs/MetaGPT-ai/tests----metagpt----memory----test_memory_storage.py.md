# `MetaGPT\tests\metagpt\memory\test_memory_storage.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Desc   : the unittests of metagpt/memory/memory_storage.py
"""

# 导入所需的模块
import os
import shutil
from pathlib import Path
from typing import List

# 导入自定义模块
from metagpt.actions import UserRequirement, WritePRD
from metagpt.actions.action_node import ActionNode
from metagpt.config import CONFIG
from metagpt.const import DATA_PATH
from metagpt.memory.memory_storage import MemoryStorage
from metagpt.schema import Message

# 设置环境变量
os.environ.setdefault("OPENAI_API_KEY", CONFIG.openai_api_key)

# 测试用例1：测试 idea_message
def test_idea_message():
    # 定义初始想法和角色ID
    idea = "Write a cli snake game"
    role_id = "UTUser1(Product Manager)"
    # 创建消息对象
    message = Message(role="User", content=idea, cause_by=UserRequirement)

    # 清空指定路径下的文件夹
    shutil.rmtree(Path(DATA_PATH / f"role_mem/{role_id}/"), ignore_errors=True)

    # 创建内存存储对象
    memory_storage: MemoryStorage = MemoryStorage()
    # 从内存中恢复消息
    messages = memory_storage.recover_memory(role_id)
    assert len(messages) == 0

    # 添加消息到内存中
    memory_storage.add(message)
    assert memory_storage.is_initialized is True

    # 测试相似的想法消息
    sim_idea = "Write a game of cli snake"
    sim_message = Message(role="User", content=sim_idea, cause_by=UserRequirement)
    new_messages = memory_storage.search_dissimilar(sim_message)
    assert len(new_messages) == 0  # similar, return []

    # 测试不相似的想法消息
    new_idea = "Write a 2048 web game"
    new_message = Message(role="User", content=new_idea, cause_by=UserRequirement)
    new_messages = memory_storage.search_dissimilar(new_message)
    assert new_messages[0].content == message.content

    # 清空内存存储
    memory_storage.clean()
    assert memory_storage.is_initialized is False

# 测试用例2：测试 actionout_message
def test_actionout_message():
    # 定义输出映射和输出数据
    out_mapping = {"field1": (str, ...), "field2": (List[str], ...)}
    out_data = {"field1": "field1 value", "field2": ["field2 value1", "field2 value2"]}
    ic_obj = ActionNode.create_model_class("prd", out_mapping)

    # 定义角色ID和消息内容
    role_id = "UTUser2(Architect)"
    content = "The user has requested the creation of a command-line interface (CLI) snake game"
    message = Message(
        content=content, instruct_content=ic_obj(**out_data), role="user", cause_by=WritePRD
    )  # WritePRD as test action

    # 清空指定路径下的文件夹
    shutil.rmtree(Path(DATA_PATH / f"role_mem/{role_id}/"), ignore_errors=True)

    # 创建内存存储对象
    memory_storage: MemoryStorage = MemoryStorage()
    # 从内存中恢复消息
    messages = memory_storage.recover_memory(role_id)
    assert len(messages) == 0

    # 添加消息到内存中
    memory_storage.add(message)
    assert memory_storage.is_initialized is True

    # 测试相似的消息内容
    sim_conent = "The request is command-line interface (CLI) snake game"
    sim_message = Message(content=sim_conent, instruct_content=ic_obj(**out_data), role="user", cause_by=WritePRD)
    new_messages = memory_storage.search_dissimilar(sim_message)
    assert len(new_messages) == 0  # similar, return []

    # 测试不相似的消息内容
    new_conent = "Incorporate basic features of a snake game such as scoring and increasing difficulty"
    new_message = Message(content=new_conent, instruct_content=ic_obj(**out_data), role="user", cause_by=WritePRD)
    new_messages = memory_storage.search_dissimilar(new_message)
    assert new_messages[0].content == message.content

    # 清空内存存储
    memory_storage.clean()
    assert memory_storage.is_initialized is False

```