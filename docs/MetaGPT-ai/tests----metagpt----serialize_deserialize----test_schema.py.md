# `MetaGPT\tests\metagpt\serialize_deserialize\test_schema.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : unittest of schema ser&deser
# 导入需要的模块
from metagpt.actions.action_node import ActionNode
from metagpt.actions.write_code import WriteCode
from metagpt.schema import Document, Documents, Message
from metagpt.utils.common import any_to_str
from tests.metagpt.serialize_deserialize.test_serdeser_base import (
    MockICMessage,
    MockMessage,
)

# 测试消息的序列化和反序列化
def test_message_serdeser():
    # 定义输出映射和数据
    out_mapping = {"field3": (str, ...), "field4": (list[str], ...)}
    out_data = {"field3": "field3 value3", "field4": ["field4 value1", "field4 value2"]}
    # 创建模型类
    ic_obj = ActionNode.create_model_class("code", out_mapping)

    # 创建消息对象
    message = Message(content="code", instruct_content=ic_obj(**out_data), role="engineer", cause_by=WriteCode)
    # 序列化消息对象
    ser_data = message.model_dump()
    # 断言序列化后的数据
    assert ser_data["cause_by"] == "metagpt.actions.write_code.WriteCode"
    assert ser_data["instruct_content"]["class"] == "code"

    # 反序列化消息对象
    new_message = Message(**ser_data)
    # 断言反序列化后的数据
    assert new_message.cause_by == any_to_str(WriteCode)
    assert new_message.cause_by in [any_to_str(WriteCode)]
    assert new_message.instruct_content != ic_obj(**out_data)  # TODO find why `!=`
    assert new_message.instruct_content.model_dump() == ic_obj(**out_data).model_dump()

    # 创建消息对象
    message = Message(content="test_ic", instruct_content=MockICMessage())
    # 序列化消息对象
    ser_data = message.model_dump()
    # 反序列化消息对象
    new_message = Message(**ser_data)
    # 断言反序列化后的数据
    assert new_message.instruct_content != MockICMessage()  # TODO

    # 创建消息对象
    message = Message(content="test_documents", instruct_content=Documents(docs={"doc1": Document(content="test doc")}))
    # 序列化消息对象
    ser_data = message.model_dump()
    # 断言序列化后的数据
    assert "class" in ser_data["instruct_content"]

# 测试没有后处理的消息
def test_message_without_postprocess():
    """to explain `instruct_content` should be postprocessed"""
    # 定义输出映射和数据
    out_mapping = {"field1": (list[str], ...)}
    out_data = {"field1": ["field1 value1", "field1 value2"]}
    # 创建模型类
    ic_obj = ActionNode.create_model_class("code", out_mapping)
    # 创建消息对象
    message = MockMessage(content="code", instruct_content=ic_obj(**out_data))
    # 序列化消息对象
    ser_data = message.model_dump()
    # 断言序列化后的数据
    assert ser_data["instruct_content"] == {}

    # 修改序列化后的数据
    ser_data["instruct_content"] = None
    # 反序列化消息对象
    new_message = MockMessage(**ser_data)
    # 断言反序列化后的数据
    assert new_message.instruct_content != ic_obj(**out_data)

```