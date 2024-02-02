# `MetaGPT\tests\metagpt\test_environment.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/12 00:47
@Author  : alexanderwu
@File    : test_environment.py
@Modified By: mashenquan, 2023/8/20. Remove global configuration `CONFIG`, enable configuration support for business isolation.
"""

# 导入模块
from pathlib import Path
import pytest
from metagpt.actions import UserRequirement
from metagpt.config import CONFIG
from metagpt.environment import Environment
from metagpt.logs import logger
from metagpt.roles import Architect, ProductManager, Role
from metagpt.schema import Message

# 定义文件路径
serdeser_path = Path(__file__).absolute().parent.joinpath("../data/serdeser_storage")

# 定义测试环境
@pytest.fixture
def env():
    return Environment()

# 测试添加角色
def test_add_role(env: Environment):
    role = ProductManager(
        name="Alice", profile="product manager", goal="create a new product", constraints="limited resources"
    )
    env.add_role(role)
    assert env.get_role(role.profile) == role

# 测试获取角色
def test_get_roles(env: Environment):
    role1 = Role(name="Alice", profile="product manager", goal="create a new product", constraints="limited resources")
    role2 = Role(name="Bob", profile="engineer", goal="develop the new product", constraints="short deadline")
    env.add_role(role1)
    env.add_role(role2)
    roles = env.get_roles()
    assert roles == {role1.profile: role1, role2.profile: role2}

# 异步测试发布和处理消息
@pytest.mark.asyncio
async def test_publish_and_process_message(env: Environment, new_filename):
    # 如果存在 git 仓库，则删除并置为 None
    if CONFIG.git_repo:
        CONFIG.git_repo.delete_repository()
        CONFIG.git_repo = None

    # 定义产品经理和架构师角色
    product_manager = ProductManager(name="Alice", profile="Product Manager", goal="做AI Native产品", constraints="资源有限")
    architect = Architect(
        name="Bob", profile="Architect", goal="设计一个可用、高效、较低成本的系统，包括数据结构与接口", constraints="资源有限，需要节省成本"
    )

    # 添加角色
    env.add_roles([product_manager, architect])

    # 发布消息
    env.publish_message(Message(role="User", content="需要一个基于LLM做总结的搜索引擎", cause_by=UserRequirement))
    # 运行环境
    await env.run(k=2)
    logger.info(f"{env.history=}")
    assert len(env.history) > 10

# 执行测试
if __name__ == "__main__":
    pytest.main([__file__, "-s"])

```