# `MetaGPT\tests\metagpt\memory\test_longterm_memory.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Desc   : unittest of `metagpt/memory/longterm_memory.py`
@Modified By: mashenquan, 2023/8/20. Remove global configuration `CONFIG`, enable configuration support for business isolation.
"""

import os  # 导入操作系统模块

import pytest  # 导入 pytest 测试框架

from metagpt.actions import UserRequirement  # 从 metagpt.actions 模块导入 UserRequirement 类
from metagpt.config import CONFIG  # 从 metagpt.config 模块导入 CONFIG 配置
from metagpt.memory.longterm_memory import LongTermMemory  # 从 metagpt.memory.longterm_memory 模块导入 LongTermMemory 类
from metagpt.roles.role import RoleContext  # 从 metagpt.roles.role 模块导入 RoleContext 类
from metagpt.schema import Message  # 从 metagpt.schema 模块导入 Message 类


def test_ltm_search():  # 定义测试函数 test_ltm_search
    assert hasattr(CONFIG, "long_term_memory") is True  # 断言 CONFIG 中存在 long_term_memory 属性
    os.environ.setdefault("OPENAI_API_KEY", CONFIG.openai_api_key)  # 设置环境变量 OPENAI_API_KEY 为 CONFIG 中的 openai_api_key
    assert len(CONFIG.openai_api_key) > 20  # 断言 openai_api_key 的长度大于 20

    role_id = "UTUserLtm(Product Manager)"  # 定义 role_id
    from metagpt.environment import Environment  # 从 metagpt.environment 模块导入 Environment 类

    Environment  # 实例化 Environment 类
    RoleContext.model_rebuild()  # 重建 RoleContext 模型
    rc = RoleContext(watch={"metagpt.actions.add_requirement.UserRequirement"})  # 创建 RoleContext 实例
    ltm = LongTermMemory()  # 创建 LongTermMemory 实例
    ltm.recover_memory(role_id, rc)  # 从存储中恢复记忆

    idea = "Write a cli snake game"  # 定义 idea
    message = Message(role="User", content=idea, cause_by=UserRequirement)  # 创建 Message 实例
    news = ltm.find_news([message])  # 在记忆中查找消息
    assert len(news) == 1  # 断言消息长度为1
    ltm.add(message)  # 将消息添加到记忆中

    sim_idea = "Write a game of cli snake"  # 定义 sim_idea
    sim_message = Message(role="User", content=sim_idea, cause_by=UserRequirement)  # 创建 sim_message 实例
    news = ltm.find_news([sim_message])  # 在记忆中查找 sim_message
    assert len(news) == 0  # 断言消息长度为0
    ltm.add(sim_message)  # 将 sim_message 添加到记忆中

    new_idea = "Write a 2048 web game"  # 定义 new_idea
    new_message = Message(role="User", content=new_idea, cause_by=UserRequirement)  # 创建 new_message 实例
    news = ltm.find_news([new_message])  # 在记忆中查找 new_message
    assert len(news) == 1  # 断言消息长度为1
    ltm.add(new_message)  # 将 new_message 添加到记忆中

    # restore from local index
    ltm_new = LongTermMemory()  # 创建新的 LongTermMemory 实例
    ltm_new.recover_memory(role_id, rc)  # 从存储中恢复记忆
    news = ltm_new.find_news([message])  # 在新的记忆中查找消息
    assert len(news) == 0  # 断言消息长度为0

    ltm_new.recover_memory(role_id, rc)  # 从存储中恢复记忆
    news = ltm_new.find_news([sim_message])  # 在新的记忆中查找 sim_message
    assert len(news) == 0  # 断言消息长度为0

    new_idea = "Write a Battle City"  # 重新定义 new_idea
    new_message = Message(role="User", content=new_idea, cause_by=UserRequirement)  # 创建新的 new_message 实例
    news = ltm_new.find_news([new_message])  # 在新的记忆中查找 new_message
    assert len(news) == 1  # 断言消息长度为1

    ltm_new.clear()  # 清空新的记忆


if __name__ == "__main__":
    pytest.main([__file__, "-s"])  # 运行测试用例

```