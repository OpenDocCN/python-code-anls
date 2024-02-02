# `MetaGPT\tests\metagpt\serialize_deserialize\test_memory.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : unittest of memory

# 导入需要的模块
from pydantic import BaseModel
from metagpt.actions.action_node import ActionNode
from metagpt.actions.add_requirement import UserRequirement
from metagpt.actions.design_api import WriteDesign
from metagpt.memory.memory import Memory
from metagpt.schema import Message
from metagpt.utils.common import any_to_str
from tests.metagpt.serialize_deserialize.test_serdeser_base import serdeser_path

# 测试内存序列化和反序列化的函数
def test_memory_serdeser():
    # 创建消息对象
    msg1 = Message(role="Boss", content="write a snake game", cause_by=UserRequirement)

    # 定义输出映射和数据
    out_mapping = {"field2": (list[str], ...)}
    out_data = {"field2": ["field2 value1", "field2 value2"]}
    ic_obj = ActionNode.create_model_class("system_design", out_mapping)
    # 创建消息对象
    msg2 = Message(
        role="Architect", instruct_content=ic_obj(**out_data), content="system design content", cause_by=WriteDesign
    )

    # 创建内存对象
    memory = Memory()
    # 添加消息到内存
    memory.add_batch([msg1, msg2])
    # 序列化内存对象
    ser_data = memory.model_dump()

    # 反序列化内存对象
    new_memory = Memory(**ser_data)
    # 断言内存对象中消息数量为2
    assert new_memory.count() == 2
    # 获取新内存对象中的消息
    new_msg2 = new_memory.get(2)[0]
    # 断言消息对象是BaseModel的实例
    assert isinstance(new_msg2, BaseModel)
    # 断言内存对象中最后一个消息对象是BaseModel的实例
    assert isinstance(new_memory.storage[-1], BaseModel)
    # 断言最后一个消息对象的cause_by属性为字符串类型
    assert new_memory.storage[-1].cause_by == any_to_str(WriteDesign)
    # 断言新消息对象的role属性为"Boss"
    assert new_msg2.role == "Boss"

    # 创建内存对象
    memory = Memory(storage=[msg1, msg2], index={msg1.cause_by: [msg1], msg2.cause_by: [msg2]})
    # 断言内存对象中消息数量为2
    assert memory.count() == 2

# 测试内存序列化和保存的函数
def test_memory_serdeser_save():
    # 创建消息对象
    msg1 = Message(role="User", content="write a 2048 game", cause_by=UserRequirement)

    # 定义输出映射和数据
    out_mapping = {"field1": (list[str], ...)}
    out_data = {"field1": ["field1 value1", "field1 value2"]}
    ic_obj = ActionNode.create_model_class("system_design", out_mapping)
    # 创建消息对象
    msg2 = Message(
        role="Architect", instruct_content=ic_obj(**out_data), content="system design content", cause_by=WriteDesign
    )

    # 创建内存对象
    memory = Memory()
    # 添加消息到内存
    memory.add_batch([msg1, msg2])

    # 定义序列化路径
    stg_path = serdeser_path.joinpath("team", "environment")
    # 序列化内存对象到指定路径
    memory.serialize(stg_path)
    # 断言序列化后的文件存在
    assert stg_path.joinpath("memory.json").exists()

    # 反序列化内存对象
    new_memory = Memory.deserialize(stg_path)
    # 断言内存对象中消息数量为2
    assert new_memory.count() == 2
    # 获取新内存对象中的消息
    new_msg2 = new_memory.get(1)[0]
    # 断言新消息对象中的instruct_content属性的field1值为["field1 value1", "field1 value2"]
    assert new_msg2.instruct_content.field1 == ["field1 value1", "field1 value2"]
    # 断言新消息对象的cause_by属性为字符串类型
    assert new_msg2.cause_by == any_to_str(WriteDesign)
    # 断言新内存对象中的索引数量为2
    assert len(new_memory.index) == 2

    # 删除序列化后的文件
    stg_path.joinpath("memory.json").unlink()

```