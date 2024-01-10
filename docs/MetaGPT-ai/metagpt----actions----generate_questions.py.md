# `MetaGPT\metagpt\actions\generate_questions.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/9/12 17:45
@Author  : fisherdeng
@File    : generate_questions.py
"""
# 导入必要的模块和类
from metagpt.actions import Action
from metagpt.actions.action_node import ActionNode

# 创建一个名为QUESTIONS的ActionNode对象，用于生成问题
QUESTIONS = ActionNode(
    key="Questions",
    expected_type=list[str],
    instruction="Task: Refer to the context to further inquire about the details that interest you, within a word limit"
    " of 150 words. Please provide the specific details you would like to inquire about here",
    example=["1. What ...", "2. How ...", "3. ..."],
)

# 创建一个名为GenerateQuestions的Action类，用于深入讨论并挖掘值得注意的细节
class GenerateQuestions(Action):
    """This class allows LLM to further mine noteworthy details based on specific "##TOPIC"(discussion topic) and
    "##RECORD" (discussion records), thereby deepening the discussion."""

    name: str = "GenerateQuestions"

    # 异步运行方法，用于填充问题并返回结果
    async def run(self, context):
        return await QUESTIONS.fill(context=context, llm=self.llm)

```