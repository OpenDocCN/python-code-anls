# `MetaGPT\metagpt\roles\project_manager.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 15:04
@Author  : alexanderwu
@File    : project_manager.py
"""

# 导入需要的模块
from metagpt.actions import WriteTasks
from metagpt.actions.design_api import WriteDesign
from metagpt.roles.role import Role

# 定义一个名为 ProjectManager 的类，继承自 Role 类
class ProjectManager(Role):
    """
    Represents a Project Manager role responsible for overseeing project execution and team efficiency.

    Attributes:
        name (str): Name of the project manager.
        profile (str): Role profile, default is 'Project Manager'.
        goal (str): Goal of the project manager.
        constraints (str): Constraints or limitations for the project manager.
    """

    # 初始化类属性
    name: str = "Eve"
    profile: str = "Project Manager"
    goal: str = (
        "break down tasks according to PRD/technical design, generate a task list, and analyze task "
        "dependencies to start with the prerequisite modules"
    )
    constraints: str = "use same language as user requirement"

    # 初始化方法
    def __init__(self, **kwargs) -> None:
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 初始化项目经理的行为
        self._init_actions([WriteTasks])
        # 监控项目经理的行为
        self._watch([WriteDesign])

```