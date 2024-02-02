# `MetaGPT\examples\sk_agent.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/9/13 12:36
@Author  : femto Zheng
@File    : sk_agent.py
"""
# 导入 asyncio 模块
import asyncio

# 从 semantic_kernel.core_skills 模块中导入 FileIOSkill, MathSkill, TextSkill, TimeSkill
from semantic_kernel.core_skills import FileIOSkill, MathSkill, TextSkill, TimeSkill
# 从 semantic_kernel.planning 模块中导入 SequentialPlanner
from semantic_kernel.planning import SequentialPlanner
# 从 semantic_kernel.planning.action_planner.action_planner 模块中导入 ActionPlanner
from semantic_kernel.planning.action_planner.action_planner import ActionPlanner
# 从 metagpt.actions 模块中导入 UserRequirement
from metagpt.actions import UserRequirement
# 从 metagpt.const 模块中导入 SKILL_DIRECTORY
from metagpt.const import SKILL_DIRECTORY
# 从 metagpt.roles.sk_agent 模块中导入 SkAgent
from metagpt.roles.sk_agent import SkAgent
# 从 metagpt.schema 模块中导入 Message
from metagpt.schema import Message
# 从 metagpt.tools.search_engine 模块中导入 SkSearchEngine
from metagpt.tools.search_engine import SkSearchEngine

# 定义异步函数 main
async def main():
    # 调用 basic_planner_web_search_example 函数
    await basic_planner_web_search_example()

# 定义异步函数 basic_planner_web_search_example
async def basic_planner_web_search_example():
    # 定义任务
    task = """
    Question: Who made the 1989 comic book, the film version of which Jon Raymond Polito appeared in?"""
    # 创建 SkAgent 对象
    role = SkAgent()
    # 导入 SkSearchEngine 技能
    role.import_skill(SkSearchEngine(), "WebSearchSkill")
    # 运行角色
    await role.run(Message(content=task, cause_by=UserRequirement))

# 如果当前脚本为主程序
if __name__ == "__main__":
    # 运行 main 函数
    asyncio.run(main())

```