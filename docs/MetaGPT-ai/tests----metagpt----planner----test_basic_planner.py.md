# `MetaGPT\tests\metagpt\planner\test_basic_planner.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/9/16 20:03
@Author  : femto Zheng
@File    : test_basic_planner.py
@Modified By: mashenquan, 2023-11-1. In accordance with Chapter 2.2.1 and 2.2.2 of RFC 116, utilize the new message
        distribution feature for message handling.
"""
# 导入所需的模块
import pytest
from semantic_kernel.core_skills import TextSkill

from metagpt.actions import UserRequirement
from metagpt.const import SKILL_DIRECTORY
from metagpt.roles.sk_agent import SkAgent
from metagpt.schema import Message

# 使用 pytest 的异步测试标记
@pytest.mark.asyncio
async def test_basic_planner():
    # 定义任务
    task = """
        Tomorrow is Valentine's day. I need to come up with a few date ideas. She speaks French so write it in French.
        Convert the text to uppercase"""
    # 创建 SkAgent 角色
    role = SkAgent()

    # 给角色添加一些技能
    role.import_semantic_skill_from_directory(SKILL_DIRECTORY, "SummarizeSkill")
    role.import_semantic_skill_from_directory(SKILL_DIRECTORY, "WriterSkill")
    role.import_skill(TextSkill(), "TextSkill")
    # 使用 BasicPlanner
    role.put_message(Message(content=task, cause_by=UserRequirement))
    # 观察角色的行为
    await role._observe()
    # 角色思考
    await role._think()
    # 假设 sk_agent 认为他需要 WriterSkill.Brainstorm 和 WriterSkill.Translate
    assert "WriterSkill.Brainstorm" in role.plan.generated_plan.result
    assert "WriterSkill.Translate" in role.plan.generated_plan.result
    # 假设 (await role._act()).content 中包含 "SALUT"，content 将是一些法语
    # assert "SALUT" in (await role._act()).content 

```