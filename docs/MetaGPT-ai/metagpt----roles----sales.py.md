# `MetaGPT\metagpt\roles\sales.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/25 17:21
@Author  : alexanderwu
@File    : sales.py
"""

# 导入必要的模块
from typing import Optional
from pydantic import Field
from metagpt.actions import SearchAndSummarize, UserRequirement
from metagpt.document_store.base_store import BaseStore
from metagpt.roles import Role
from metagpt.tools import SearchEngineType

# 定义一个销售角色类，继承自 Role 类
class Sales(Role):
    # 定义销售角色的属性
    name: str = "John Smith"
    profile: str = "Retail Sales Guide"
    desc: str = (
        "As a Retail Sales Guide, my name is John Smith. I specialize in addressing customer inquiries with "
        "expertise and precision. My responses are based solely on the information available in our knowledge"
        " base. In instances where your query extends beyond this scope, I'll honestly indicate my inability "
        "to provide an answer, rather than speculate or assume. Please note, each of my replies will be "
        "delivered with the professionalism and courtesy expected of a seasoned sales guide."
    )

    # 定义一个可选的属性，表示销售角色所属的商店
    store: Optional[BaseStore] = Field(default=None, exclude=True)

    # 初始化销售角色对象
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_store(self.store)

    # 设置销售角色所属的商店
    def _set_store(self, store):
        if store:
            # 如果有商店，创建一个搜索和总结的动作
            action = SearchAndSummarize(name="", engine=SearchEngineType.CUSTOM_ENGINE, search_func=store.asearch)
        else:
            # 如果没有商店，创建一个默认的搜索和总结的动作
            action = SearchAndSummarize()
        # 初始化销售角色的动作
        self._init_actions([action])
        # 监听用户需求
        self._watch([UserRequirement])

```