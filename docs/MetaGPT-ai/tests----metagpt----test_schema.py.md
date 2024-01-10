# `MetaGPT\tests\metagpt\test_schema.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/20 10:40
@Author  : alexanderwu
@File    : test_schema.py
@Modified By: mashenquan, 2023-11-1. In line with Chapter 2.2.1 and 2.2.2 of RFC 116, introduce unit tests for
            the utilization of the new feature of `Message` class.
"""

import json  # 导入json模块

import pytest  # 导入pytest模块

from metagpt.actions import Action  # 从metagpt.actions模块导入Action类
from metagpt.actions.action_node import ActionNode  # 从metagpt.actions.action_node模块导入ActionNode类
from metagpt.actions.write_code import WriteCode  # 从metagpt.actions.write_code模块导入WriteCode类
from metagpt.config import CONFIG  # 从metagpt.config模块导入CONFIG对象
from metagpt.const import SYSTEM_DESIGN_FILE_REPO, TASK_FILE_REPO  # 从metagpt.const模块导入SYSTEM_DESIGN_FILE_REPO, TASK_FILE_REPO常量
from metagpt.schema import (  # 从metagpt.schema模块导入以下类
    AIMessage,
    ClassAttribute,
    ClassMethod,
    ClassView,
    CodeSummarizeContext,
    Document,
    Message,
    MessageQueue,
    SystemMessage,
    UserMessage,
)
from metagpt.utils.common import any_to_str  # 从metagpt.utils.common模块导入any_to_str函数

# 测试消息
def test_messages():
    test_content = "test_message"
    msgs = [
        UserMessage(content=test_content),  # 创建UserMessage对象
        SystemMessage(content=test_content),  # 创建SystemMessage对象
        AIMessage(content=test_content),  # 创建AIMessage对象
        Message(content=test_content, role="QA"),  # 创建Message对象
    ]
    text = str(msgs)  # 将消息列表转换为字符串
    roles = ["user", "system", "assistant", "QA"]
    assert all([i in text for i in roles])  # 断言所有角色在文本中

# 测试消息
def test_message():
    Message("a", role="v1")  # 创建Message对象

    m = Message(content="a", role="v1")  # 创建Message对象
    v = m.dump()  # 转储消息
    d = json.loads(v)  # 从JSON字符串解码为Python对象
    assert d  # 断言d存在
    assert d.get("content") == "a"  # 断言内容为"a"
    assert d.get("role") == "v1"  # 断言角色为"v1"
    m.role = "v2"  # 修改角色为"v2"
    v = m.dump()  # 转储消息
    assert v  # 断言v存在
    m = Message.load(v)  # 从JSON字符串加载消息
    assert m.content == "a"  # 断言内容为"a"
    assert m.role == "v2"  # 断言角色为"v2"

    m = Message(content="a", role="b", cause_by="c", x="d", send_to="c")  # 创建Message对象
    assert m.content == "a"  # 断言内容为"a"
    assert m.role == "b"  # 断言角色为"b"
    assert m.send_to == {"c"}  # 断言发送给"c"
    assert m.cause_by == "c"  # 断言原因为"c"
    m.sent_from = "e"  # 设置发送者为"e"
    assert m.sent_from == "e"  # 断言发送者为"e"

    m.cause_by = "Message"  # 修改原因为"Message"
    assert m.cause_by == "Message"  # 断言原因为"Message"
    m.cause_by = Action  # 修改原因为Action类
    assert m.cause_by == any_to_str(Action)  # 断言原因为Action类的字符串表示
    m.cause_by = Action()  # 修改原因为Action对象
    assert m.cause_by == any_to_str(Action)  # 断言原因为Action类的字符串表示
    m.content = "b"  # 修改内容为"b"
    assert m.content == "b"  # 断言内容为"b"

# 测试路由
def test_routes():
    m = Message(content="a", role="b", cause_by="c", x="d", send_to="c")  # 创建Message对象
    m.send_to = "b"  # 设置发送给"b"
    assert m.send_to == {"b"}  # 断言发送给"b"
    m.send_to = {"e", Action}  # 设置发送给{"e", Action}
    assert m.send_to == {"e", any_to_str(Action)}  # 断言发送给{"e", Action的字符串表示}

# 测试消息序列化和反序列化
def test_message_serdeser():
    out_mapping = {"field3": (str, ...), "field4": (list[str], ...)}  # 创建映射
    out_data = {"field3": "field3 value3", "field4": ["field4 value1", "field4 value2"]}  # 创建数据
    ic_obj = ActionNode.create_model_class("code", out_mapping)  # 创建模型类

    message = Message(content="code", instruct_content=ic_obj(**out_data), role="engineer", cause_by=WriteCode)  # 创建Message对象
    message_dict = message.model_dump()  # 转储消息
    assert message_dict["cause_by"] == "metagpt.actions.write_code.WriteCode"  # 断言原因为WriteCode类
    assert message_dict["instruct_content"] == {
        "class": "code",
        "mapping": {"field3": "(<class 'str'>, Ellipsis)", "field4": "(list[str], Ellipsis)"},
        "value": {"field3": "field3 value3", "field4": ["field4 value1", "field4 value2"]},
    }  # 断言指令内容
    new_message = Message.model_validate(message_dict)  # 验证消息
    assert new_message.content == message.content  # 断言内容相同
    assert new_message.instruct_content.model_dump() == message.instruct_content.model_dump()  # 断言指令内容相同
    assert new_message.instruct_content != message.instruct_content  # TODO
    assert new_message.cause_by == message.cause_by  # 断言原因相同
    assert new_message.instruct_content.field3 == out_data["field3"]  # 断言指令内容的field3属性为out_data["field3"]

    message = Message(content="code")  # 创建Message对象
    message_dict = message.model_dump()  # 转储消息
    new_message = Message(**message_dict)  # 从字典创建Message对象
    assert new_message.instruct_content is None  # 断言指令内容为None
    assert new_message.cause_by == "metagpt.actions.add_requirement.UserRequirement"  # 断言原因为"metagpt.actions.add_requirement.UserRequirement"
    assert not Message.load("{")  # 断言加载失败

# 测试文档
def test_document():
    doc = Document(root_path="a", filename="b", content="c")  # 创建Document对象
    meta_doc = doc.get_meta()  # 获取元数据文档
    assert doc.root_path == meta_doc.root_path  # 断言根路径相同
    assert doc.filename == meta_doc.filename  # 断言文件名相同
    assert meta_doc.content == ""  # 断言内容为空字符串

    assert doc.full_path == str(CONFIG.git_repo.workdir / doc.root_path / doc.filename)  # 断言完整路径

# 异步测试消息队列
@pytest.mark.asyncio
async def test_message_queue():
    mq = MessageQueue()  # 创建消息队列
    val = await mq.dump()  # 转储消息队列
    assert val == "[]"  # 断言值为"[]"
    mq.push(Message(content="1"))  # 推送消息
    mq.push(Message(content="2中文测试aaa"))  # 推送消息
    msg = mq.pop()  # 弹出消息
    assert msg.content == "1"  # 断言消息内容为"1"

    val = await mq.dump()  # 转储消息队列
    assert val  # 断言值存在
    new_mq = MessageQueue.load(val)  # 从值加载消息队列
    assert new_mq.pop_all() == mq.pop_all()  # 断言弹出所有消息相同

# 参数化测试CodeSummarizeContext
@pytest.mark.parametrize(
    ("file_list", "want"),
    [
        (
            [f"{SYSTEM_DESIGN_FILE_REPO}/a.txt", f"{TASK_FILE_REPO}/b.txt"],
            CodeSummarizeContext(
                design_filename=f"{SYSTEM_DESIGN_FILE_REPO}/a.txt", task_filename=f"{TASK_FILE_REPO}/b.txt"
            ),
        )
    ],
)
def test_CodeSummarizeContext(file_list, want):
    ctx = CodeSummarizeContext.loads(file_list)  # 从文件列表加载CodeSummarizeContext
    assert ctx == want  # 断言相等
    m = {ctx: ctx}  # 创建字典
    assert want in m  # 断言want在字典中

# 测试类视图
def test_class_view():
    attr_a = ClassAttribute(name="a", value_type="int", default_value="0", visibility="+", abstraction=True)  # 创建类属性
    assert attr_a.get_mermaid(align=1) == "\t+int a=0*"  # 断言mermaid格式正确
    attr_b = ClassAttribute(name="b", value_type="str", default_value="0", visibility="#", static=True)  # 创建类属性
    assert attr_b.get_mermaid(align=0) == '#str b="0"$'  # 断言mermaid格式正确
    class_view = ClassView(name="A")  # 创建类视图
    class_view.attributes = [attr_a, attr_b]  # 设置属性列表

    method_a = ClassMethod(name="run", visibility="+", abstraction=True)  # 创建类方法
    assert method_a.get_mermaid(align=1) == "\t+run()*"  # 断言mermaid格式正确
    method_b = ClassMethod(
        name="_test",
        visibility="#",
        static=True,
        args=[ClassAttribute(name="a", value_type="str"), ClassAttribute(name="b", value_type="int")],
        return_type="str",
    )  # 创建类方法
    assert method_b.get_mermaid(align=0) == "#_test(str a,int b):str$"  # 断言mermaid格式正确
    class_view.methods = [method_a, method_b]  # 设置方法列表
    assert (
        class_view.get_mermaid(align=0)
        == 'class A{\n\t+int a=0*\n\t#str b="0"$\n\t+run()*\n\t#_test(str a,int b):str$\n}\n'
    )  # 断言mermaid格式正确

# 执行测试
if __name__ == "__main__":
    pytest.main([__file__, "-s"])  # 执行pytest测试

```