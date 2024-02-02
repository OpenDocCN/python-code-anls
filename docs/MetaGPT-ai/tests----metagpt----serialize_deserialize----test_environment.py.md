# `MetaGPT\tests\metagpt\serialize_deserialize\test_environment.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : 用于指定 Python 解释器的路径和文件编码

import shutil  # 导入 shutil 模块，用于高级文件操作

from metagpt.actions.action_node import ActionNode  # 从 metagpt.actions.action_node 模块中导入 ActionNode 类
from metagpt.actions.add_requirement import UserRequirement  # 从 metagpt.actions.add_requirement 模块中导入 UserRequirement 类
from metagpt.actions.project_management import WriteTasks  # 从 metagpt.actions.project_management 模块中导入 WriteTasks 类
from metagpt.environment import Environment  # 从 metagpt.environment 模块中导入 Environment 类
from metagpt.roles.project_manager import ProjectManager  # 从 metagpt.roles.project_manager 模块中导入 ProjectManager 类
from metagpt.schema import Message  # 从 metagpt.schema 模块中导入 Message 类
from metagpt.utils.common import any_to_str  # 从 metagpt.utils.common 模块中导入 any_to_str 函数
from tests.metagpt.serialize_deserialize.test_serdeser_base import (  # 从 tests.metagpt.serialize_deserialize.test_serdeser_base 模块中导入 ActionOK, ActionRaise, RoleC, serdeser_path
    ActionOK,
    ActionRaise,
    RoleC,
    serdeser_path,
)


def test_env_serialize():
    env = Environment()  # 创建 Environment 实例
    ser_env_dict = env.model_dump()  # 调用 model_dump 方法，将环境序列化为字典
    assert "roles" in ser_env_dict  # 断言字典中包含 "roles" 键
    assert len(ser_env_dict["roles"]) == 0  # 断言 "roles" 键对应的值的长度为 0


def test_env_deserialize():
    env = Environment()  # 创建 Environment 实例
    env.publish_message(message=Message(content="test env serialize"))  # 发布消息
    ser_env_dict = env.model_dump()  # 调用 model_dump 方法，将环境序列化为字典
    new_env = Environment(**ser_env_dict)  # 使用序列化后的字典创建新的 Environment 实例
    assert len(new_env.roles) == 0  # 断言新环境的角色数量为 0
    assert len(new_env.history) == 25  # 断言新环境的历史记录长度为 25


def test_environment_serdeser():
    out_mapping = {"field1": (list[str], ...)}  # 定义输出映射
    out_data = {"field1": ["field1 value1", "field1 value2"]}  # 定义输出数据
    ic_obj = ActionNode.create_model_class("prd", out_mapping)  # 创建模型类

    message = Message(  # 创建消息对象
        content="prd", instruct_content=ic_obj(**out_data), role="product manager", cause_by=any_to_str(UserRequirement)
    )

    environment = Environment()  # 创建 Environment 实例
    role_c = RoleC()  # 创建 RoleC 实例
    environment.add_role(role_c)  # 将 RoleC 实例添加到环境中
    environment.publish_message(message)  # 发布消息

    ser_data = environment.model_dump()  # 调用 model_dump 方法，将环境序列化为字典
    assert ser_data["roles"]["Role C"]["name"] == "RoleC"  # 断言序列化后的字典中的角色名称为 "RoleC"

    new_env: Environment = Environment(**ser_data)  # 使用序列化后的字典创建新的 Environment 实例
    assert len(new_env.roles) == 1  # 断言新环境的角色数量为 1

    assert list(new_env.roles.values())[0].states == list(environment.roles.values())[0].states  # 断言新环境的角色状态与原环境的角色状态相同
    assert isinstance(list(environment.roles.values())[0].actions[0], ActionOK)  # 断言原环境的第一个角色的第一个动作是 ActionOK 类的实例
    assert type(list(new_env.roles.values())[0].actions[0]) == ActionOK  # 断言新环境的第一个角色的第一个动作是 ActionOK 类的实例
    assert type(list(new_env.roles.values())[0].actions[1]) == ActionRaise  # 断言新环境的第一个角色的第二个动作是 ActionRaise 类的实例


def test_environment_serdeser_v2():
    environment = Environment()  # 创建 Environment 实例
    pm = ProjectManager()  # 创建 ProjectManager 实例
    environment.add_role(pm)  # 将 ProjectManager 实例添加到环境中

    ser_data = environment.model_dump()  # 调用 model_dump 方法，将环境序列化为字典

    new_env: Environment = Environment(**ser_data)  # 使用序列化后的字典创建新的 Environment 实例
    role = new_env.get_role(pm.profile)  # 获取新环境中的指定角色
    assert isinstance(role, ProjectManager)  # 断言该角色是 ProjectManager 类的实例
    assert isinstance(role.actions[0], WriteTasks)  # 断言该角色的第一个动作是 WriteTasks 类的实例
    assert isinstance(list(new_env.roles.values())[0].actions[0], WriteTasks)  # 断言新环境的第一个角色的第一个动作是 WriteTasks 类的实例


def test_environment_serdeser_save():
    environment = Environment()  # 创建 Environment 实例
    role_c = RoleC()  # 创建 RoleC 实例

    shutil.rmtree(serdeser_path.joinpath("team"), ignore_errors=True)  # 删除指定路径下的文件夹及其内容

    stg_path = serdeser_path.joinpath("team", "environment")  # 定义存储路径
    environment.add_role(role_c)  # 将 RoleC 实例添加到环境中
    environment.serialize(stg_path)  # 将环境序列化并保存到指定路径

    new_env: Environment = Environment.deserialize(stg_path)  # 从指定路径反序列化环境
    assert len(new_env.roles) == 1  # 断言新环境的角色数量为 1
    assert type(list(new_env.roles.values())[0].actions[0]) == ActionOK  # 断言新环境的第一个角色的第一个动作是 ActionOK 类的实例

```