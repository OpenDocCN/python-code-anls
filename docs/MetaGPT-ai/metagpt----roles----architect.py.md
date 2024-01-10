# `MetaGPT\metagpt\roles\architect.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 14:43
@Author  : alexanderwu
@File    : architect.py
"""

# 从metagpt.actions中导入WritePRD和WriteDesign
from metagpt.actions import WritePRD
from metagpt.actions.design_api import WriteDesign
from metagpt.roles.role import Role

# 定义Architect类，继承自Role类
class Architect(Role):
    """
    Represents an Architect role in a software development process.

    Attributes:
        name (str): Name of the architect.
        profile (str): Role profile, default is 'Architect'.
        goal (str): Primary goal or responsibility of the architect.
        constraints (str): Constraints or guidelines for the architect.
    """

    # 初始化属性
    name: str = "Bob"
    profile: str = "Architect"
    goal: str = "design a concise, usable, complete software system"
    constraints: str = (
        "make sure the architecture is simple enough and use  appropriate open source "
        "libraries. Use same language as user requirement"
    )

    # 构造函数
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # 初始化特定于Architect角色的操作
        self._init_actions([WriteDesign])

        # 设置Architect应该关注或了解的事件或操作
        self._watch({WritePRD})

```