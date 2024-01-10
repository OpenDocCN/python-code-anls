# `MetaGPT\metagpt\actions\design_api_review.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 19:31
@Author  : alexanderwu
@File    : design_api_review.py
"""

# 导入必要的模块
from typing import Optional
from metagpt.actions.action import Action

# 定义一个名为DesignReview的类，继承自Action类
class DesignReview(Action):
    # 定义类属性
    name: str = "DesignReview"
    context: Optional[str] = None

    # 定义异步方法run，接受两个参数prd和api_design
    async def run(self, prd, api_design):
        # 构建提示信息
        prompt = (
            f"Here is the Product Requirement Document (PRD):\n\n{prd}\n\nHere is the list of APIs designed "
            f"based on this PRD:\n\n{api_design}\n\nPlease review whether this API design meets the requirements"
            f" of the PRD, and whether it complies with good design practices."
        )
        # 调用_aask方法，等待用户输入
        api_review = await self._aask(prompt)
        # 返回API设计的审查结果
        return api_review

```