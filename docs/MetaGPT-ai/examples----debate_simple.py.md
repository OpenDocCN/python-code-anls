# `MetaGPT\examples\debate_simple.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/22
@Author  : alexanderwu
@File    : debate_simple.py
"""
# 导入 asyncio 模块
import asyncio

# 导入自定义模块
from metagpt.actions import Action
from metagpt.environment import Environment
from metagpt.roles import Role
from metagpt.team import Team

# 创建动作对象
action1 = Action(name="AlexSay", instruction="Express your opinion with emotion and don't repeat it")
action2 = Action(name="BobSay", instruction="Express your opinion with emotion and don't repeat it")

# 创建角色对象
alex = Role(name="Alex", profile="Democratic candidate", goal="Win the election", actions=[action1], watch=[action2])
bob = Role(name="Bob", profile="Republican candidate", goal="Win the election", actions=[action2], watch=[action1])

# 创建环境对象
env = Environment(desc="US election live broadcast")

# 创建团队对象
team = Team(investment=10.0, env=env, roles=[alex, bob])

# 运行团队的异步方法
asyncio.run(team.run(idea="Topic: climate change. Under 80 words per message.", send_to="Alex", n_round=5))

```