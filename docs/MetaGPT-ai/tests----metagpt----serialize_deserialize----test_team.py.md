# `MetaGPT\tests\metagpt\serialize_deserialize\test_team.py`

```

# -*- coding: utf-8 -*-
# @Date    : 11/27/2023 10:07 AM
# @Author  : stellahong (stellahong@fuzhi.ai)
# @Desc    :

# 导入 shutil 模块
import shutil

# 导入 pytest 模块
import pytest

# 从 metagpt.const 模块中导入 SERDESER_PATH 常量
from metagpt.const import SERDESER_PATH

# 从 metagpt.logs 模块中导入 logger 对象
from metagpt.logs import logger

# 从 metagpt.roles 模块中导入 Architect, ProductManager, ProjectManager 类
from metagpt.roles import Architect, ProductManager, ProjectManager

# 从 metagpt.team 模块中导入 Team 类
from metagpt.team import Team

# 从 tests.metagpt.serialize_deserialize.test_serdeser_base 模块中导入 ActionOK, RoleA, RoleB, RoleC, serdeser_path 变量
from tests.metagpt.serialize_deserialize.test_serdeser_base import (
    ActionOK,
    RoleA,
    RoleB,
    RoleC,
    serdeser_path,
)

# 定义测试函数 test_team_deserialize
def test_team_deserialize():
    # 创建 Team 对象
    company = Team()

    # 创建 ProductManager 对象
    pm = ProductManager()

    # 创建 Architect 对象
    arch = Architect()

    # 雇佣 ProductManager, Architect 和 ProjectManager 三个角色
    company.hire(
        [
            pm,
            arch,
            ProjectManager(),
        ]
    )

    # 断言公司环境中角色的数量为 3
    assert len(company.env.get_roles()) == 3

    # 对公司进行序列化
    ser_company = company.model_dump()

    # 从序列化数据中恢复出新的 Team 对象
    new_company = Team.model_validate(ser_company)

    # 断言新公司环境中角色的数量为 3
    assert len(new_company.env.get_roles()) == 3

    # 断言新公司环境中存在 ProductManager 角色
    assert new_company.env.get_role(pm.profile) is not None

    # 获取新公司环境中的 ProductManager 角色
    new_pm = new_company.env.get_role(pm.profile)

    # 断言新公司环境中的 ProductManager 角色类型为 ProductManager
    assert type(new_pm) == ProductManager

    # 断言新公司环境中存在 ProductManager 和 Architect 角色
    assert new_company.env.get_role(pm.profile) is not None
    assert new_company.env.get_role(arch.profile) is not None

# 定义测试函数 test_team_serdeser_save
def test_team_serdeser_save():
    # 创建 Team 对象
    company = Team()

    # 雇佣 RoleC 角色
    company.hire([RoleC()])

    # 设置存储路径
    stg_path = serdeser_path.joinpath("team")

    # 删除存储路径下的文件夹
    shutil.rmtree(stg_path, ignore_errors=True)

    # 将公司对象序列化并保存到指定路径
    company.serialize(stg_path=stg_path)

    # 从指定路径反序列化出新的 Team 对象
    new_company = Team.deserialize(stg_path)

    # 断言新公司环境中角色的数量为 1
    assert len(new_company.env.roles) == 1

# 定义异步测试函数 test_team_recover
@pytest.mark.asyncio
async def test_team_recover():
    # 定义初始想法
    idea = "write a snake game"

    # 设置存储路径
    stg_path = SERDESER_PATH.joinpath("team")

    # 删除存储路径下的文件夹
    shutil.rmtree(stg_path, ignore_errors=True)

    # 创建 Team 对象
    company = Team()

    # 创建 RoleC 角色
    role_c = RoleC()

    # 雇佣 RoleC 角色
    company.hire([role_c])

    # 运行项目
    company.run_project(idea)

    # 运行多轮
    await company.run(n_round=4)

    # 对公司进行序列化
    ser_data = company.model_dump()

    # 从序列化数据中恢复出新的 Team 对象
    new_company = Team(**ser_data)

    # 获取新公司环境中的 RoleC 角色
    new_company.env.get_role(role_c.profile)

    # 断言新公司环境中的 RoleC 角色的行为类型为 ActionOK
    assert type(list(new_company.env.roles.values())[0].actions[0]) == ActionOK

    # 运行项目
    new_company.run_project(idea)

    # 运行多轮
    await new_company.run(n_round=4)

# 定义异步测试函数 test_team_recover_save
@pytest.mark.asyncio
async def test_team_recover_save():
    # 定义初始想法
    idea = "write a 2048 web game"

    # 设置存储路径
    stg_path = SERDESER_PATH.joinpath("team")

    # 删除存储路径下的文件夹
    shutil.rmtree(stg_path, ignore_errors=True)

    # 创建 Team 对象
    company = Team()

    # 创建 RoleC 角色
    role_c = RoleC()

    # 雇佣 RoleC 角色
    company.hire([role_c])

    # 运行项目
    company.run_project(idea)

    # 运行多轮
    await company.run(n_round=4)

    # 从指定路径反序列化出新的 Team 对象
    new_company = Team.deserialize(stg_path)

    # 获取新公司环境中的 RoleC 角色
    new_role_c = new_company.env.get_role(role_c.profile)

    # 断言新公司环境中的 RoleC 角色的 recovered 属性与原角色的 recovered 属性不同
    assert new_role_c.recovered != role_c.recovered

    # 断言新公司环境中的 RoleC 角色的 rc.todo 属性与原角色的 rc.todo 属性不同
    assert new_role_c.rc.todo != role_c.rc.todo

    # 断言新公司环境中的 RoleC 角色的 rc.news 属性与原角色的 rc.news 属性不同
    assert new_role_c.rc.news != role_c.rc.news

    # 运行项目
    new_company.run_project(idea)

    # 运行多轮
    await new_company.run(n_round=4)

# 定义异步测试函数 test_team_recover_multi_roles_save
@pytest.mark.asyncio
async def test_team_recover_multi_roles_save():
    # 定义初始想法
    idea = "write a snake game"

    # 设置存储路径
    stg_path = SERDESER_PATH.joinpath("team")

    # 删除存储路径下的文件夹
    shutil.rmtree(stg_path, ignore_errors=True)

    # 创建 RoleA 角色
    role_a = RoleA()

    # 创建 RoleB 角色
    role_b = RoleB()

    # 创建 Team 对象
    company = Team()

    # 雇佣 RoleA 和 RoleB 两个角色
    company.hire([role_a, role_b])

    # 运行项目
    company.run_project(idea)

    # 运行多轮
    await company.run(n_round=4)

    # 从指定路径反序列化出新的 Team 对象
    new_company = Team.deserialize(stg_path)

    # 运行项目
    new_company.run_project(idea)

    # 断言新公司环境中的 RoleB 角色的 rc.state 属性为 1
    assert new_company.env.get_role(role_b.profile).rc.state == 1

    # 运行多轮
    await new_company.run(n_round=4)

```