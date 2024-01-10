# `MetaGPT\metagpt\actions\write_prd_review.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 17:45
@Author  : alexanderwu
@File    : write_prd_review.py
"""

# 导入必要的模块
from typing import Optional
from metagpt.actions.action import Action

# 定义一个名为WritePRDReview的类，继承自Action类
class WritePRDReview(Action):
    # 定义类属性
    name: str = ""
    context: Optional[str] = None
    prd: Optional[str] = None
    desc: str = "Based on the PRD, conduct a PRD Review, providing clear and detailed feedback"
    prd_review_prompt_template: str = """
Given the following Product Requirement Document (PRD):
{prd}

As a project manager, please review it and provide your feedback and suggestions.
"""

    # 定义一个异步方法run，接受一个名为prd的参数
    async def run(self, prd):
        # 将传入的prd赋值给类属性prd
        self.prd = prd
        # 根据prd_review_prompt_template生成提示信息
        prompt = self.prd_review_prompt_template.format(prd=self.prd)
        # 调用_aask方法，等待用户输入
        review = await self._aask(prompt)
        # 返回用户输入的review
        return review

```