# `MetaGPT\metagpt\roles\product_manager.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 14:43
@Author  : alexanderwu
@File    : product_manager.py
@Modified By: mashenquan, 2023/11/27. Add `PrepareDocuments` action according to Section 2.2.3.5.1 of RFC 135.
"""

# 导入需要的模块
from metagpt.actions import UserRequirement, WritePRD
from metagpt.actions.prepare_documents import PrepareDocuments
from metagpt.config import CONFIG
from metagpt.roles.role import Role
from metagpt.utils.common import any_to_name

# 定义产品经理角色类
class ProductManager(Role):
    """
    Represents a Product Manager role responsible for product development and management.

    Attributes:
        name (str): Name of the product manager.
        profile (str): Role profile, default is 'Product Manager'.
        goal (str): Goal of the product manager.
        constraints (str): Constraints or limitations for the product manager.
    """

    # 初始化产品经理的属性
    name: str = "Alice"
    profile: str = "Product Manager"
    goal: str = "efficiently create a successful product that meets market demands and user expectations"
    constraints: str = "utilize the same language as the user requirements for seamless communication"
    todo_action: str = ""

    # 初始化产品经理角色
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # 初始化产品经理的行动列表
        self._init_actions([PrepareDocuments, WritePRD])
        # 监听用户需求和准备文档的行动
        self._watch([UserRequirement, PrepareDocuments])
        # 设置待办行动为准备文档
        self.todo_action = any_to_name(PrepareDocuments)

    # 决定下一步要做什么
    async def _think(self) -> bool:
        """Decide what to do"""
        if CONFIG.git_repo and not CONFIG.git_reinit:
            self._set_state(1)
        else:
            self._set_state(0)
            CONFIG.git_reinit = False
            self.todo_action = any_to_name(WritePRD)
        return bool(self.rc.todo)

    # 观察环境
    async def _observe(self, ignore_memory=False) -> int:
        return await super()._observe(ignore_memory=True)

    # 返回待办行动
    @property
    def todo(self) -> str:
        """AgentStore uses this attribute to display to the user what actions the current role should take."""
        return self.todo_action

```