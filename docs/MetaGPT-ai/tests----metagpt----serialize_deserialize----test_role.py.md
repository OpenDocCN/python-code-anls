# `MetaGPT\tests\metagpt\serialize_deserialize\test_role.py`

```

# -*- coding: utf-8 -*-
# @Date    : 11/23/2023 4:49 PM
# @Author  : stellahong (stellahong@fuzhi.ai)
# @Desc    :

# 导入 shutil 模块
import shutil

# 导入 pytest 模块
import pytest
# 导入 pydantic 模块中的 BaseModel 和 SerializeAsAny
from pydantic import BaseModel, SerializeAsAny

# 导入 metagpt.actions 模块中的 WriteCode
from metagpt.actions import WriteCode
# 导入 metagpt.actions.add_requirement 模块中的 UserRequirement
from metagpt.actions.add_requirement import UserRequirement
# 导入 metagpt.const 模块中的 SERDESER_PATH
from metagpt.const import SERDESER_PATH
# 导入 metagpt.logs 模块中的 logger
from metagpt.logs import logger
# 导入 metagpt.roles.engineer 模块中的 Engineer
from metagpt.roles.engineer import Engineer
# 导入 metagpt.roles.product_manager 模块中的 ProductManager
from metagpt.roles.product_manager import ProductManager
# 导入 metagpt.roles.role 模块中的 Role
from metagpt.roles.role import Role
# 导入 metagpt.schema 模块中的 Message
from metagpt.schema import Message
# 导入 metagpt.utils.common 模块中的 format_trackback_info
from metagpt.utils.common import format_trackback_info
# 导入 tests.metagpt.serialize_deserialize.test_serdeser_base 模块中的 ActionOK, RoleA, RoleB, RoleC, RoleD, serdeser_path
from tests.metagpt.serialize_deserialize.test_serdeser_base import (
    ActionOK,
    RoleA,
    RoleB,
    RoleC,
    RoleD,
    serdeser_path,
)

# 测试函数，测试 RoleA 和 RoleB 的功能
def test_roles():
    role_a = RoleA()
    assert len(role_a.rc.watch) == 1
    role_b = RoleB()
    assert len(role_a.rc.watch) == 1
    assert len(role_b.rc.watch) == 1

    role_d = RoleD(actions=[ActionOK()])
    assert len(role_d.actions) == 1

# 测试函数，测试 RoleA 和 RoleB 的子类
def test_role_subclasses():
    """test subclasses of role with same fields in ser&deser"""

    class RoleSubClasses(BaseModel):
        roles: list[SerializeAsAny[Role]] = []

    role_subcls = RoleSubClasses(roles=[RoleA(), RoleB()])
    role_subcls_dict = role_subcls.model_dump()

    new_role_subcls = RoleSubClasses(**role_subcls_dict)
    assert isinstance(new_role_subcls.roles[0], RoleA)
    assert isinstance(new_role_subcls.roles[1], RoleB)

# 测试函数，测试 Role 的序列化
def test_role_serialize():
    role = Role()
    ser_role_dict = role.model_dump()
    assert "name" in ser_role_dict
    assert "states" in ser_role_dict
    assert "actions" in ser_role_dict

# 测试函数，测试 Engineer 的序列化
def test_engineer_serialize():
    role = Engineer()
    ser_role_dict = role.model_dump()
    assert "name" in ser_role_dict
    assert "states" in ser_role_dict
    assert "actions" in ser_role_dict

# 异步测试函数，测试 Engineer 的反序列化
@pytest.mark.asyncio
async def test_engineer_deserialize():
    role = Engineer(use_code_review=True)
    ser_role_dict = role.model_dump()

    new_role = Engineer(**ser_role_dict)
    assert new_role.name == "Alex"
    assert new_role.use_code_review is True
    assert len(new_role.actions) == 1
    assert isinstance(new_role.actions[0], WriteCode)
    # await new_role.actions[0].run(context="write a cli snake game", filename="test_code")

# 测试函数，测试 Role 的序列化和反序列化保存
def test_role_serdeser_save():
    stg_path_prefix = serdeser_path.joinpath("team", "environment", "roles")
    shutil.rmtree(serdeser_path.joinpath("team"), ignore_errors=True)

    pm = ProductManager()
    role_tag = f"{pm.__class__.__name__}_{pm.name}"
    stg_path = stg_path_prefix.joinpath(role_tag)
    pm.serialize(stg_path)

    new_pm = Role.deserialize(stg_path)
    assert new_pm.name == pm.name
    assert len(new_pm.get_memories(1)) == 0

# 异步测试函数，测试 Role 的序列化和反序列化中断
@pytest.mark.asyncio
async def test_role_serdeser_interrupt():
    role_c = RoleC()
    shutil.rmtree(SERDESER_PATH.joinpath("team"), ignore_errors=True)

    stg_path = SERDESER_PATH.joinpath("team", "environment", "roles", f"{role_c.__class__.__name__}_{role_c.name}")
    try:
        await role_c.run(with_message=Message(content="demo", cause_by=UserRequirement))
    except Exception:
        logger.error(f"Exception in `role_a.run`, detail: {format_trackback_info()}")
        role_c.serialize(stg_path)

    assert role_c.rc.memory.count() == 1

    new_role_a: Role = Role.deserialize(stg_path)
    assert new_role_a.rc.state == 1

    with pytest.raises(Exception):
        await new_role_a.run(with_message=Message(content="demo", cause_by=UserRequirement))

# 如果当前脚本被直接执行，则执行测试
if __name__ == "__main__":
    pytest.main([__file__, "-s"])

```