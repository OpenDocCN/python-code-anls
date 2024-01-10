# `MetaGPT\tests\metagpt\planner\test_action_planner.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/9/16 20:03
@Author  : femto Zheng
@File    : test_basic_planner.py
@Modified By: mashenquan, 2023-11-1. In accordance with Chapter 2.2.1 and 2.2.2 of RFC 116, utilize the new message
        distribution feature for message handling.
"""
# 导入 pytest 模块
import pytest
# 从 semantic_kernel.core_skills 模块中导入 FileIOSkill, MathSkill, TextSkill, TimeSkill 类
from semantic_kernel.core_skills import FileIOSkill, MathSkill, TextSkill, TimeSkill
# 从 semantic_kernel.planning.action_planner.action_planner 模块中导入 ActionPlanner 类
from semantic_kernel.planning.action_planner.action_planner import ActionPlanner

# 从 metagpt.actions 模块中导入 UserRequirement 类
from metagpt.actions import UserRequirement
# 从 metagpt.roles.sk_agent 模块中导入 SkAgent 类
from metagpt.roles.sk_agent import SkAgent
# 从 metagpt.schema 模块中导入 Message 类
from metagpt.schema import Message

# 使用 pytest 的 asyncio 标记
@pytest.mark.asyncio
# 定义测试函数 test_action_planner
async def test_action_planner():
    # 创建 SkAgent 对象，使用 ActionPlanner 类作为 planner_cls
    role = SkAgent(planner_cls=ActionPlanner)
    # 给 agent 添加 4 个技能
    role.import_skill(MathSkill(), "math")
    role.import_skill(FileIOSkill(), "fileIO")
    role.import_skill(TimeSkill(), "time")
    role.import_skill(TextSkill(), "text")
    # 定义任务
    task = "What is the sum of 110 and 990?"

    # 将任务作为消息传递给 agent
    role.put_message(Message(content=task, cause_by=UserRequirement))
    # 观察消息
    await role._observe()
    # 让 agent 思考，选择合适的技能
    await role._think()  # it will choose mathskill.Add
    # 断言 agent 执行后的结果
    assert "1100" == (await role._act()).content

```