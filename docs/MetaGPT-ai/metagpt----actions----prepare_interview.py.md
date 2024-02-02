# `MetaGPT\metagpt\actions\prepare_interview.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/9/19 15:02
@Author  : DevXiaolan
@File    : prepare_interview.py
"""
# 导入必要的模块和类
from metagpt.actions import Action
from metagpt.actions.action_node import ActionNode

# 创建一个名为QUESTIONS的ActionNode对象
QUESTIONS = ActionNode(
    key="Questions",
    expected_type=list[str],
    instruction="""Role: You are an interviewer of our company who is well-knonwn in frontend or backend develop;
Requirement: Provide a list of questions for the interviewer to ask the interviewee, by reading the resume of the interviewee in the context.
Attention: Provide as markdown block as the format above, at least 10 questions.""",
    example=["1. What ...", "2. How ..."],
)

# 创建一个名为PrepareInterview的Action类
class PrepareInterview(Action):
    name: str = "PrepareInterview"

    # 异步运行方法
    async def run(self, context):
        # 调用QUESTIONS对象的fill方法，填充上下文和llm参数
        return await QUESTIONS.fill(context=context, llm=self.llm)

```