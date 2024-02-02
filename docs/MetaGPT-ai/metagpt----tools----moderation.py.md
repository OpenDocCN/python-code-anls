# `MetaGPT\metagpt\tools\moderation.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/9/26 14:27
@Author  : zhanglei
@File    : moderation.py
"""
# 导入 Union 类型
from typing import Union

# 从metagpt.llm模块中导入LLM类
from metagpt.llm import LLM

# 定义Moderation类
class Moderation:
    # 初始化方法
    def __init__(self):
        # 实例化LLM类
        self.llm = LLM()

    # 处理审核结果的方法
    def handle_moderation_results(self, results):
        # 初始化响应列表
        resp = []
        # 遍历结果
        for item in results:
            # 获取分类字典
            categories = item.categories.dict()
            # 获取被标记为True的分类
            true_categories = [category for category, item_flagged in categories.items() if item_flagged]
            # 将结果添加到响应列表中
            resp.append({"flagged": item.flagged, "true_categories": true_categories})
        # 返回响应列表
        return resp

    # 带分类的审核方法
    async def amoderation_with_categories(self, content: Union[str, list[str]]):
        # 初始化响应列表
        resp = []
        # 如果内容不为空
        if content:
            # 进行审核并获取结果
            moderation_results = await self.llm.amoderation(content=content)
            # 处理审核结果
            resp = self.handle_moderation_results(moderation_results.results)
        # 返回响应列表
        return resp

    # 审核方法
    async def amoderation(self, content: Union[str, list[str]]):
        # 初始化响应列表
        resp = []
        # 如果内容不为空
        if content:
            # 进行审核并获取结果
            moderation_results = await self.llm.amoderation(content=content)
            # 获取结果列表
            results = moderation_results.results
            # 遍历结果列表
            for item in results:
                # 将标记结果添加到响应列表中
                resp.append(item.flagged)
        # 返回响应列表
        return resp

```