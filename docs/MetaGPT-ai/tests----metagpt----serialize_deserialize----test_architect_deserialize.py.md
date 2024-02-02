# `MetaGPT\tests\metagpt\serialize_deserialize\test_architect_deserialize.py`

```py

# -*- coding: utf-8 -*-  # 设置文件编码格式为 UTF-8
# @Date    : 11/26/2023 2:04 PM  # 代码编写日期和时间
# @Author  : stellahong (stellahong@fuzhi.ai)  # 作者信息
# @Desc    :  # 代码描述

import pytest  # 导入 pytest 模块

from metagpt.actions.action import Action  # 从 metagpt.actions.action 模块导入 Action 类
from metagpt.roles.architect import Architect  # 从 metagpt.roles.architect 模块导入 Architect 类


def test_architect_serialize():  # 定义测试函数 test_architect_serialize
    role = Architect()  # 创建 Architect 类的实例对象 role
    ser_role_dict = role.model_dump(by_alias=True)  # 调用 model_dump 方法将 role 对象序列化为字典
    assert "name" in ser_role_dict  # 断言字典中包含键 "name"
    assert "states" in ser_role_dict  # 断言字典中包含键 "states"
    assert "actions" in ser_role_dict  # 断言字典中包含键 "actions"


@pytest.mark.asyncio  # 使用 pytest.mark.asyncio 装饰器标记异步测试函数
async def test_architect_deserialize():  # 定义异步测试函数 test_architect_deserialize
    role = Architect()  # 创建 Architect 类的实例对象 role
    ser_role_dict = role.model_dump(by_alias=True)  # 调用 model_dump 方法将 role 对象序列化为字典
    new_role = Architect(**ser_role_dict)  # 使用字典内容创建新的 Architect 类实例对象 new_role
    # new_role = Architect.deserialize(ser_role_dict)  # 另一种方式使用 deserialize 方法创建新的 Architect 类实例对象 new_role
    assert new_role.name == "Bob"  # 断言 new_role 的 name 属性为 "Bob"
    assert len(new_role.actions) == 1  # 断言 new_role 的 actions 属性长度为 1
    assert isinstance(new_role.actions[0], Action)  # 断言 new_role 的 actions 属性的第一个元素是 Action 类的实例
    await new_role.actions[0].run(with_messages="write a cli snake game")  # 调用 new_role 的 actions 属性的第一个元素的 run 方法，并传入参数

```