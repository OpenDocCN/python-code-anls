# `MetaGPT\metagpt\roles\assistant.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/8/7
@Author  : mashenquan
@File    : assistant.py
@Desc   : I am attempting to incorporate certain symbol concepts from UML into MetaGPT, enabling it to have the
            ability to freely construct flows through symbol concatenation. Simultaneously, I am also striving to
            make these symbols configurable and standardized, making the process of building flows more convenient.
            For more about `fork` node in activity diagrams, see: `https://www.uml-diagrams.org/activity-diagrams.html`
          This file defines a `fork` style meta role capable of generating arbitrary roles at runtime based on a
            configuration file.
@Modified By: mashenquan, 2023/8/22. A definition has been provided for the return value of _think: returning false
            indicates that further reasoning cannot continue.

"""
# 导入需要的模块
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field

# 导入自定义模块
from metagpt.actions.skill_action import ArgumentsParingAction, SkillAction
from metagpt.actions.talk_action import TalkAction
from metagpt.config import CONFIG
from metagpt.learn.skill_loader import SkillsDeclaration
from metagpt.logs import logger
from metagpt.memory.brain_memory import BrainMemory
from metagpt.roles import Role
from metagpt.schema import Message

# 定义消息类型的枚举
class MessageType(Enum):
    Talk = "TALK"
    Skill = "SKILL"

```